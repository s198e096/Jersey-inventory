
import os
import io
import re
import json
import base64
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5-mini")

def get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

SUPABASE_URL = get_secret("SUPABASE_URL")
SUPABASE_KEY = get_secret("SUPABASE_KEY")


# ----------------------------
# Supabase database helpers
# ----------------------------
def get_supabase():
    try:
        from supabase import create_client
    except ImportError:
        raise RuntimeError("The supabase package is not installed. Run: pip install -r requirements.txt")
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY are not configured.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def init_db():
    sb = get_supabase()
    try:
        sb.table("inventory").select("id").limit(1).execute()
        sb.table("sales").select("id").limit(1).execute()
    except Exception as e:
        raise RuntimeError("Could not access Supabase tables. Run supabase_schema.sql first.") from e


def read_inventory():
    sb = get_supabase()
    inv_rows = sb.table("inventory").select("*").order("id", desc=True).execute().data or []
    sales_rows = sb.table("sales").select("inventory_id,quantity").execute().data or []
    inv = pd.DataFrame(inv_rows)
    if inv.empty:
        return pd.DataFrame(columns=["id","created_at","import_batch","source_file","order_number","version","team","player_name","jersey_number","size","quantity_received","unit_cost","line_cost","notes","quantity_sold","quantity_in_stock"])
    sold = pd.DataFrame(sales_rows)
    if sold.empty:
        inv["quantity_sold"] = 0
    else:
        totals = sold.groupby("inventory_id", as_index=False)["quantity"].sum().rename(columns={"quantity":"quantity_sold"})
        inv = inv.merge(totals, left_on="id", right_on="inventory_id", how="left").drop(columns=["inventory_id"], errors="ignore")
        inv["quantity_sold"] = inv["quantity_sold"].fillna(0).astype(int)
    inv["quantity_in_stock"] = inv["quantity_received"].astype(int) - inv["quantity_sold"].astype(int)
    return inv.sort_values("id", ascending=False).reset_index(drop=True)


def read_sales():
    sb = get_supabase()
    sales = pd.DataFrame(sb.table("sales").select("*").order("id", desc=True).execute().data or [])
    inv = pd.DataFrame(sb.table("inventory").select("*").execute().data or [])
    cols=["id","sold_at","inventory_id","team","player_name","jersey_number","version","size","quantity","unit_cost","sale_price_each","fees","shipping_cost","revenue","cogs","profit","notes"]
    if sales.empty:
        return pd.DataFrame(columns=cols)
    invj=inv[["id","team","player_name","jersey_number","version","size","unit_cost"]].rename(columns={"id":"inventory_id"})
    df=sales.merge(invj,on="inventory_id",how="left")
    df["revenue"]=df["quantity"]*df["sale_price_each"]
    df["cogs"]=df["quantity"]*df["unit_cost"]
    df["profit"]=df["revenue"]-df["cogs"]-df["fees"]-df["shipping_cost"]
    return df[cols].sort_values("id",ascending=False).reset_index(drop=True)


def add_inventory_rows(df, source_file):
    sb=get_supabase(); batch=datetime.now().strftime("%Y%m%d-%H%M%S"); now=datetime.now().isoformat(timespec="seconds")
    payload=[]
    for _,row in df.iterrows():
        qty=int(row.get("quantity",0) or 0); line_cost=float(row.get("line_cost",0) or 0); unit_cost=float(row.get("unit_cost",0) or 0)
        if qty>0 and unit_cost<=0 and line_cost>0: unit_cost=line_cost/qty
        if qty>0 and line_cost<=0 and unit_cost>0: line_cost=unit_cost*qty
        payload.append({"created_at":now,"import_batch":batch,"source_file":source_file,"order_number":str(row.get("order_number","") or ""),"version":str(row.get("version","") or ""),"team":str(row.get("team","") or ""),"player_name":str(row.get("player_name","") or ""),"jersey_number":str(row.get("jersey_number","") or ""),"size":str(row.get("size","") or ""),"quantity_received":qty,"unit_cost":unit_cost,"line_cost":line_cost,"notes":str(row.get("notes","") or "")})
    if payload: sb.table("inventory").insert(payload).execute()


def delete_inventory_row(inv_id):
    sb=get_supabase()
    if sb.table("sales").select("id").eq("inventory_id",int(inv_id)).limit(1).execute().data:
        raise ValueError("This inventory lot already has sales and cannot be deleted.")
    sb.table("inventory").delete().eq("id",int(inv_id)).execute()


def normalize_size(value): return str(value or "").strip().upper()

def split_size_tokens(value):
    raw=normalize_size(value)
    if not raw: return []
    raw=raw.replace("2XL","XXL").replace("3XL","XXXL")
    tokens=[t.strip() for t in re.split(r"[/,;|]+|\s+",raw) if t.strip()]
    valid={"XS","S","M","L","XL","XXL","XXXL"}
    return tokens if len(tokens)>1 and all(t in valid for t in tokens) else []


def expand_evenly_split_sizes(df):
    expanded=[]
    for _,row in df.iterrows():
        row=row.copy(); sizes=split_size_tokens(row.get("size","")); qty=int(row.get("quantity",0) or 0)
        if sizes and qty==len(sizes):
            total=float(row.get("line_cost",0) or 0); unit=float(row.get("unit_cost",0) or 0)
            if unit<=0 and qty>0 and total>0: unit=total/qty
            for size in sizes:
                nr=row.copy(); nr["size"]=size; nr["quantity"]=1; nr["unit_cost"]=unit; nr["line_cost"]=unit
                nr["notes"]=(str(nr.get("notes","") or "").strip()+" Auto-split from combined size row.").strip(); expanded.append(nr)
        else: expanded.append(row)
    return pd.DataFrame(expanded,columns=df.columns)


def product_key_columns(): return ["team","player_name","jersey_number","version"]


def add_sale_for_product_size(product_row,size,quantity,sale_price_each,fees,shipping_cost,notes):
    sb=get_supabase(); inv=read_inventory(); mask=pd.Series(True,index=inv.index)
    for col in product_key_columns(): mask &= inv[col].fillna("").astype(str).eq(str(product_row[col] or ""))
    mask &= inv["size"].fillna("").astype(str).str.upper().eq(normalize_size(size))
    matches=inv[mask & (inv["quantity_in_stock"]>0)].copy()
    if matches.empty: raise ValueError(f"No {size} jerseys are currently in stock.")
    total=int(matches["quantity_in_stock"].sum())
    if quantity>total: raise ValueError(f"Only {total} unit(s) of size {size} are currently in stock.")
    matches["created_at_dt"]=pd.to_datetime(matches["created_at"],errors="coerce"); matches=matches.sort_values(["created_at_dt","id"])
    remaining=int(quantity); alloc=[]
    for _,lot in matches.iterrows():
        if remaining<=0: break
        take=min(remaining,int(lot["quantity_in_stock"])); alloc.append((int(lot["id"]),take)); remaining-=take
    now=datetime.now().isoformat(timespec="seconds"); payload=[]
    for iid,q in alloc:
        p=q/int(quantity); payload.append({"sold_at":now,"inventory_id":iid,"quantity":q,"sale_price_each":float(sale_price_each),"fees":float(fees)*p,"shipping_cost":float(shipping_cost)*p,"notes":notes})
    sb.table("sales").insert(payload).execute()


def split_existing_inventory_lot(inv_id,size_quantities):
    sb=get_supabase(); rows=sb.table("inventory").select("*").eq("id",int(inv_id)).execute().data or []
    if not rows: raise ValueError("Inventory lot not found.")
    row=rows[0]; sold=sb.table("sales").select("quantity").eq("inventory_id",int(inv_id)).execute().data or []
    if sum(int(r["quantity"]) for r in sold)>0: raise ValueError("This lot already has recorded sales. Only unsold lots can be split.")
    cleaned={normalize_size(s):int(q) for s,q in size_quantities.items() if normalize_size(s) and int(q)>0}
    oq=int(row["quantity_received"])
    if sum(cleaned.values())!=oq: raise ValueError(f"Size quantities must add up to {oq}.")
    unit=float(row["unit_cost"]); payload=[]
    for size,q in cleaned.items():
        payload.append({"created_at":row["created_at"],"import_batch":row.get("import_batch"),"source_file":row.get("source_file"),"order_number":row.get("order_number"),"version":row.get("version"),"team":row.get("team"),"player_name":row.get("player_name"),"jersey_number":row.get("jersey_number"),"size":size,"quantity_received":q,"unit_cost":unit,"line_cost":unit*q,"notes":((row.get("notes") or "")+f" Split from original inventory lot {inv_id}.").strip()})
    sb.table("inventory").insert(payload).execute(); sb.table("inventory").delete().eq("id",int(inv_id)).execute()


def inventory_by_size(inv):
    if inv.empty: return inv
    g=product_key_columns()+["size"]
    qty=inv.groupby(g,dropna=False,as_index=False).agg(quantity_received=("quantity_received","sum"),quantity_sold=("quantity_sold","sum"),quantity_in_stock=("quantity_in_stock","sum"))
    val=inv.assign(inventory_value=inv["quantity_in_stock"]*inv["unit_cost"]).groupby(g,dropna=False,as_index=False)["inventory_value"].sum()
    return qty.merge(val,on=g,how="left").sort_values(g)


def update_sale(sale_id,quantity,sale_price_each,fees,shipping_cost,notes):
    sb=get_supabase(); rows=sb.table("sales").select("*").eq("id",int(sale_id)).execute().data or []
    if not rows: raise ValueError("Sale not found.")
    sale=rows[0]; iid=int(sale["inventory_id"]); inv=sb.table("inventory").select("quantity_received").eq("id",iid).execute().data or []
    others=sb.table("sales").select("id,quantity").eq("inventory_id",iid).execute().data or []
    max_allowed=int(inv[0]["quantity_received"])-sum(int(r["quantity"]) for r in others if int(r["id"])!=int(sale_id))
    if int(quantity)>max_allowed: raise ValueError(f"This inventory lot can support at most {max_allowed} unit(s) for this sale.")
    sb.table("sales").update({"quantity":int(quantity),"sale_price_each":float(sale_price_each),"fees":float(fees),"shipping_cost":float(shipping_cost),"notes":notes}).eq("id",int(sale_id)).execute()


def delete_sale(sale_id):
    sb=get_supabase()
    if not sb.table("sales").select("id").eq("id",int(sale_id)).execute().data: raise ValueError("Sale not found.")
    sb.table("sales").delete().eq("id",int(sale_id)).execute()


# ----------------------------
# Voice sale helpers
# ----------------------------
def transcribe_sale_audio(audio_file):
    """Transcribe a short recorded sale description."""
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("The openai package is not installed. Run: pip install -r requirements.txt")

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    # st.audio_input returns a file-like UploadedFile (WAV).
    audio_file.seek(0)
    transcript = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe",
        file=audio_file,
    )
    return transcript.text.strip()


def parse_voice_sale(transcript, available_inventory):
    """
    Convert a voice transcript into sale fields, using current inventory as
    grounding so team/player/size names are matched to items that actually exist.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("The openai package is not installed. Run: pip install -r requirements.txt")

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    client = OpenAI(api_key=api_key)

    # Give the model only the inventory choices it needs for matching.
    choices = []
    grouped = (
        available_inventory[
            ["team", "player_name", "jersey_number", "version", "size", "quantity_in_stock"]
        ]
        .fillna("")
        .groupby(
            ["team", "player_name", "jersey_number", "version", "size"],
            as_index=False,
            dropna=False
        )["quantity_in_stock"]
        .sum()
    )

    for _, r in grouped.iterrows():
        if int(r["quantity_in_stock"]) > 0:
            choices.append({
                "team": str(r["team"]),
                "player_name": str(r["player_name"]),
                "jersey_number": str(r["jersey_number"]),
                "version": str(r["version"]),
                "size": str(r["size"]),
                "quantity_in_stock": int(r["quantity_in_stock"]),
            })

    prompt = f"""
You extract a jersey sale from a short voice transcript.

VOICE TRANSCRIPT:
{transcript}

CURRENT AVAILABLE INVENTORY:
{json.dumps(choices, ensure_ascii=False)}

Return JSON only in this exact shape:
{{
  "team": "",
  "player_name": "",
  "jersey_number": "",
  "version": "",
  "size": "",
  "quantity": 1,
  "sale_price_each": 0.0,
  "fees": 0.0,
  "shipping_cost": 0.0,
  "notes": "",
  "match_confidence": "high"
}}

Rules:
1. Match team/player/number/version/size to CURRENT AVAILABLE INVENTORY whenever possible.
2. The spoken price is the SALE PRICE PER JERSEY unless the speaker explicitly says it is a total.
3. If quantity is not spoken, use 1.
4. If fees or shipping are not spoken, use 0.
5. Never invent a team, player, size, or price that was not spoken or strongly matched to inventory.
6. If the transcript is ambiguous, choose the closest inventory match but set match_confidence to "low".
7. Normalize common size speech: small=S, medium=M, large=L, extra large=XL, double XL=XXL.
8. Return JSON only, no markdown.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    result = extract_json(response.output_text)

    # Defensive normalization
    result["quantity"] = int(result.get("quantity", 1) or 1)
    result["sale_price_each"] = float(result.get("sale_price_each", 0) or 0)
    result["fees"] = float(result.get("fees", 0) or 0)
    result["shipping_cost"] = float(result.get("shipping_cost", 0) or 0)
    return result


def voice_product_match_index(products, parsed):
    """Return the best exact-ish product index for the parsed voice sale."""
    if products.empty:
        return 0

    def norm(x):
        return str(x or "").strip().lower()

    best_idx = 0
    best_score = -1

    for idx, row in products.iterrows():
        score = 0
        for col, weight in [
            ("team", 4),
            ("player_name", 5),
            ("jersey_number", 3),
            ("version", 1),
        ]:
            p = norm(parsed.get(col, ""))
            r = norm(row[col])
            if p and r and p == r:
                score += weight
            elif p and r and (p in r or r in p):
                score += max(1, weight - 2)

        if score > best_score:
            best_score = score
            best_idx = idx

    return int(best_idx)


# ----------------------------
# OpenAI image extraction
# ----------------------------
def image_to_data_url(uploaded_file):
    data = uploaded_file.getvalue()
    mime = uploaded_file.type or "image/jpeg"
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end >= 0:
        text = text[start:end + 1]
    return json.loads(text)


def parse_order_sheet(uploaded_file):
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("The openai package is not installed. Run: pip install -r requirements.txt")

    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI(api_key=api_key)

    prompt = """
You are reading a soccer/football jersey supplier order sheet from an image.

Extract EVERY jersey line item that can be reasonably identified. The sheet may contain:
- an order/row number
- version such as FAN, Player, Retro
- a jersey photo
- size(s), sometimes several sizes listed vertically
- player name and jersey number
- quantity
- a line price at the far right

Important accounting rule:
The far-right price on sheets like this is usually the TOTAL COST FOR THAT ROW, not the per-unit cost.
For example, if QTY is 3 and Price is $39, line_cost=39 and unit_cost=13.

Return JSON only, in this exact shape:
{
  "rows": [
    {
      "order_number": "1",
      "version": "FAN",
      "team": "Liverpool",
      "player_name": "M. Salah",
      "jersey_number": "10",
      "size": "L",
      "quantity": 2,
      "line_cost": 26.0,
      "unit_cost": 13.0,
      "notes": ""
    }
  ]
}

Rules:
1. Use one row per distinct jersey/size lot.
2. Prefer ONE OUTPUT ROW PER SIZE. If multiple sizes are listed and quantity-per-size is visible, create a separate row for each size. If total quantity equals the number of listed sizes (for example S/M/L and QTY 3), treat that as one of each size and divide the line cost evenly. Only keep sizes combined when the split truly cannot be determined, and explain that in notes.
3. If the player name/number are visible, capture them.
4. If a team can be confidently recognized from visible text, crest, or jersey design, capture it; otherwise use "".
5. Never invent a price or quantity. Use 0 when unreadable.
6. Convert dollar values to numbers only, no "$".
7. line_cost is the total supplier cost for that line. unit_cost = line_cost / quantity when quantity > 0.
8. Preserve useful uncertainty in notes, e.g. "team inferred from jersey image".
9. Do not include summary/total rows.
10. Return JSON only and no markdown.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_to_data_url(uploaded_file)}
                ]
            }
        ]
    )
    payload = extract_json(response.output_text)
    rows = payload.get("rows", [])
    df = pd.DataFrame(rows)

    required_cols = [
        "order_number", "version", "team", "player_name", "jersey_number",
        "size", "quantity", "line_cost", "unit_cost", "notes"
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = "" if col not in {"quantity", "line_cost", "unit_cost"} else 0

    df = df[required_cols]
    for col in ["quantity", "line_cost", "unit_cost"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["quantity"] = df["quantity"].astype(int)
    missing_unit = (df["unit_cost"] <= 0) & (df["quantity"] > 0) & (df["line_cost"] > 0)
    df.loc[missing_unit, "unit_cost"] = (
        df.loc[missing_unit, "line_cost"] / df.loc[missing_unit, "quantity"]
    )
    df = expand_evenly_split_sizes(df)
    return df


# ----------------------------
# UI
# ----------------------------
st.set_page_config(page_title="Jersey Inventory Tracker", page_icon="⚽", layout="wide")
try:
    init_db()
except Exception as e:
    st.error(str(e))
    st.info("Configure Supabase secrets and run supabase_schema.sql first.")
    st.stop()

st.title("⚽ Jersey Inventory & Profit Tracker")
st.caption("Import supplier sheets from images, track stock, record sales, and see realized profit.")

tabs = st.tabs([
    "Dashboard",
    "Import Order Sheet",
    "Inventory",
    "Record Sale",
    "Sales History"
])

with tabs[0]:
    inv = read_inventory()
    sales = read_sales()

    units_in_stock = int(inv["quantity_in_stock"].sum()) if not inv.empty else 0
    inventory_value = float((inv["quantity_in_stock"] * inv["unit_cost"]).sum()) if not inv.empty else 0
    total_spend = float((inv["quantity_received"] * inv["unit_cost"]).sum()) if not inv.empty else 0
    revenue = float(sales["revenue"].sum()) if not sales.empty else 0
    profit = float(sales["profit"].sum()) if not sales.empty else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Units in stock", f"{units_in_stock:,}")
    c2.metric("Inventory value", f"${inventory_value:,.2f}")
    c3.metric("Total supplier spend", f"${total_spend:,.2f}")
    c4.metric("Sales revenue", f"${revenue:,.2f}")
    c5.metric("Realized profit", f"${profit:,.2f}")

    st.subheader("Profit by team")
    if sales.empty:
        st.info("No sales recorded yet.")
    else:
        by_team = (
            sales.assign(team=sales["team"].replace("", "Unknown"))
            .groupby("team", as_index=False)[["revenue", "profit"]]
            .sum()
            .sort_values("profit", ascending=False)
        )
        st.bar_chart(by_team.set_index("team")[["revenue", "profit"]])

    st.subheader("Low / sold-out stock")
    if inv.empty:
        st.info("No inventory yet.")
    else:
        low = inv[inv["quantity_in_stock"] <= 2][
            ["id", "team", "player_name", "jersey_number", "size", "quantity_in_stock", "unit_cost"]
        ]
        st.dataframe(low, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Upload a supplier order sheet")
    st.write(
        "Upload a screenshot/photo like your supplier sheet. The AI will read the jersey rows, "
        "then you can review and edit everything before it is saved."
    )

    uploaded = st.file_uploader(
        "Order-sheet image",
        type=["png", "jpg", "jpeg", "webp"],
        key="order_sheet"
    )

    if uploaded:
        try:
            img = Image.open(io.BytesIO(uploaded.getvalue()))
            st.image(img, caption=uploaded.name, use_container_width=True)
        except Exception:
            st.warning("Image preview could not be displayed.")

        if st.button("Read jersey sheet with AI", type="primary"):
            with st.spinner("Reading jersey rows..."):
                try:
                    st.session_state["parsed_order"] = parse_order_sheet(uploaded)
                    st.success("Extraction complete. Review the rows below before saving.")
                except Exception as e:
                    st.error(str(e))

    if "parsed_order" in st.session_state:
        st.subheader("Review extracted rows")
        st.caption("You can directly edit cells, add rows, or remove incorrect rows.")
        edited = st.data_editor(
            st.session_state["parsed_order"],
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "quantity": st.column_config.NumberColumn(min_value=0, step=1),
                "line_cost": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
                "unit_cost": st.column_config.NumberColumn(format="$%.2f", min_value=0.0),
            },
            key="parsed_editor"
        )

        total_units = int(pd.to_numeric(edited["quantity"], errors="coerce").fillna(0).sum())
        total_cost = float(pd.to_numeric(edited["line_cost"], errors="coerce").fillna(0).sum())
        a, b = st.columns(2)
        a.metric("Detected units", total_units)
        b.metric("Detected supplier cost", f"${total_cost:,.2f}")

        if st.button("Add reviewed rows to inventory"):
            cleaned = edited.copy()
            cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce").fillna(0).astype(int)
            cleaned["line_cost"] = pd.to_numeric(cleaned["line_cost"], errors="coerce").fillna(0.0)
            cleaned["unit_cost"] = pd.to_numeric(cleaned["unit_cost"], errors="coerce").fillna(0.0)
            cleaned = cleaned[cleaned["quantity"] > 0]
            if cleaned.empty:
                st.error("There are no rows with a quantity greater than zero.")
            else:
                add_inventory_rows(cleaned, uploaded.name if uploaded else "")
                del st.session_state["parsed_order"]
                st.success(f"Added {len(cleaned)} inventory lot(s).")
                st.rerun()

with tabs[2]:
    st.subheader("Current inventory")
    inv = read_inventory()

    if inv.empty:
        st.info("No inventory yet.")
    else:
        search = st.text_input("Search inventory", placeholder="e.g. Ronaldo, Barcelona, XL")
        view = inv.copy()
        if search:
            q = search.lower()
            mask = view.astype(str).apply(
                lambda row: row.str.lower().str.contains(q, regex=False).any(), axis=1
            )
            view = view[mask]

        st.markdown("#### Stock by size")
        size_summary = inventory_by_size(view)
        st.dataframe(
            size_summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "inventory_value": st.column_config.NumberColumn(format="$%.2f"),
            }
        )

        with st.expander("Show detailed purchase lots"):
            cols = [
                "id", "team", "player_name", "jersey_number", "version", "size",
                "quantity_received", "quantity_sold", "quantity_in_stock",
                "unit_cost", "line_cost", "source_file", "created_at"
            ]
            st.dataframe(
                view[cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "unit_cost": st.column_config.NumberColumn(format="$%.2f"),
                    "line_cost": st.column_config.NumberColumn(format="$%.2f"),
                }
            )

        csv = view[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download inventory CSV",
            data=csv,
            file_name="jersey_inventory.csv",
            mime="text/csv"
        )

        with st.expander("Split / Edit Sizes for existing inventory"):
            st.write(
                "Use this for older inventory that was saved with combined sizes such as "
                "`S/M/L`. The quantities you enter must add up to the original lot quantity."
            )

            unsold = view[view["quantity_sold"] == 0].copy()
            if unsold.empty:
                st.info("There are no unsold inventory lots available to split.")
            else:
                def split_label(r):
                    who = f"{r['player_name']} #{r['jersey_number']}".strip()
                    return (
                        f"ID {int(r['id'])} — {r['team']} — {who} — {r['version']} — "
                        f"Current size: {r['size']} — Qty {int(r['quantity_received'])}"
                    )

                split_options = {
                    split_label(r): int(r["id"])
                    for _, r in unsold.iterrows()
                }
                split_choice = st.selectbox(
                    "Inventory lot to split",
                    list(split_options.keys()),
                    key="split_inventory_lot"
                )
                split_id = split_options[split_choice]
                split_row = unsold[unsold["id"] == split_id].iloc[0]
                original_qty = int(split_row["quantity_received"])

                st.caption(
                    f"Original quantity: {original_qty}. "
                    "Enter the quantity you have for each size below."
                )

                size_cols = st.columns(4)
                split_sizes = ["XS", "S", "M", "L", "XL", "XXL", "XXXL"]

                entered = {}
                for idx, size in enumerate(split_sizes):
                    with size_cols[idx % 4]:
                        entered[size] = st.number_input(
                            size,
                            min_value=0,
                            max_value=original_qty,
                            value=0,
                            step=1,
                            key=f"split_{split_id}_{size}"
                        )

                entered_total = sum(int(v) for v in entered.values())
                st.write(f"**Entered total:** {entered_total} / {original_qty}")

                if entered_total == original_qty:
                    st.success("Size quantities match the original inventory quantity.")
                else:
                    st.warning(
                        f"You still need to allocate {original_qty - entered_total} unit(s)."
                        if entered_total < original_qty
                        else f"You allocated {entered_total - original_qty} too many unit(s)."
                    )

                if st.button(
                    "Save size split",
                    type="primary",
                    disabled=(entered_total != original_qty),
                    key="save_size_split"
                ):
                    try:
                        split_existing_inventory_lot(split_id, entered)
                        st.success("Inventory lot was split into size-specific inventory.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

        with st.expander("Delete an inventory lot"):
            st.warning("Only lots with no recorded sales can be deleted.")
            delete_id = st.number_input("Inventory ID to delete", min_value=1, step=1)
            if st.button("Delete inventory lot"):
                try:
                    delete_inventory_row(int(delete_id))
                    st.success("Inventory lot deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

with tabs[3]:
    st.subheader("Record a sale")
    inv = read_inventory()
    available = inv[inv["quantity_in_stock"] > 0].copy() if not inv.empty else pd.DataFrame()

    if available.empty:
        st.info("There are no items in stock.")
    else:
        pcols = product_key_columns()
        products = (
            available[pcols]
            .fillna("")
            .astype(str)
            .drop_duplicates()
            .reset_index(drop=True)
        )

        def product_label(r):
            who = str(r["player_name"]).strip()
            number = str(r["jersey_number"]).strip()
            if number:
                who = f"{who} #{number}".strip()
            parts = [str(r["team"]).strip(), who, str(r["version"]).strip()]
            return " — ".join([p for p in parts if p]) or "Unnamed jersey"

        # -------------------------------------------------
        # Voice sale
        # -------------------------------------------------
        st.markdown("### 🎙️ Record sale by voice")
        st.caption(
            'Example: "Sold one Barcelona Lamine Yamal, size medium, for 45 dollars."'
        )

        voice_audio = st.audio_input(
            "Tap the microphone and describe the sale",
            key="voice_sale_audio"
        )

        if voice_audio is not None:
            if st.button("Process voice sale", type="primary", key="process_voice_sale"):
                with st.spinner("Listening and matching it to your inventory..."):
                    try:
                        transcript = transcribe_sale_audio(voice_audio)
                        parsed = parse_voice_sale(transcript, available)
                        st.session_state["voice_sale_transcript"] = transcript
                        st.session_state["voice_sale_parsed"] = parsed
                    except Exception as e:
                        st.error(str(e))

        if "voice_sale_parsed" in st.session_state:
            parsed = st.session_state["voice_sale_parsed"]
            transcript = st.session_state.get("voice_sale_transcript", "")

            st.success(f'Heard: "{transcript}"')

            if str(parsed.get("match_confidence", "")).lower() == "low":
                st.warning("The inventory match may be uncertain. Please verify the fields below.")

            st.markdown("#### Confirm sale details")

            product_labels = [product_label(r) for _, r in products.iterrows()]
            default_product_idx = voice_product_match_index(products, parsed)

            voice_product_idx = st.selectbox(
                "Jersey",
                options=list(range(len(products))),
                index=min(default_product_idx, len(products) - 1),
                format_func=lambda i: product_labels[i],
                key="voice_confirm_product"
            )
            voice_product_row = products.iloc[int(voice_product_idx)]

            voice_product_mask = pd.Series(True, index=available.index)
            for col in pcols:
                voice_product_mask &= (
                    available[col].fillna("").astype(str)
                    == str(voice_product_row[col] or "")
                )
            voice_product_stock = available[voice_product_mask].copy()

            voice_size_stock = (
                voice_product_stock.assign(
                    size=voice_product_stock["size"].fillna("").astype(str).str.upper()
                )
                .groupby("size", dropna=False, as_index=False)["quantity_in_stock"]
                .sum()
            )
            voice_size_stock = voice_size_stock[voice_size_stock["quantity_in_stock"] > 0]

            voice_sizes = voice_size_stock["size"].tolist()
            parsed_size = normalize_size(parsed.get("size", ""))
            voice_size_index = (
                voice_sizes.index(parsed_size)
                if parsed_size in voice_sizes
                else 0
            )

            voice_size = st.selectbox(
                "Size",
                voice_sizes,
                index=voice_size_index,
                format_func=lambda s: (
                    f"{s or 'Unspecified'} — "
                    f"{int(voice_size_stock.loc[voice_size_stock['size'] == s, 'quantity_in_stock'].iloc[0])} in stock"
                ),
                key="voice_confirm_size"
            )

            voice_stock = int(
                voice_size_stock.loc[
                    voice_size_stock["size"] == voice_size,
                    "quantity_in_stock"
                ].iloc[0]
            )

            vc1, vc2 = st.columns(2)
            voice_qty = vc1.number_input(
                "Quantity sold",
                min_value=1,
                max_value=voice_stock,
                value=min(max(int(parsed.get("quantity", 1)), 1), voice_stock),
                step=1,
                key="voice_confirm_qty"
            )
            voice_price = vc2.number_input(
                "Sale price per jersey ($)",
                min_value=0.0,
                value=float(parsed.get("sale_price_each", 0) or 0),
                step=1.0,
                key="voice_confirm_price"
            )
            voice_fees = vc1.number_input(
                "Fees ($)",
                min_value=0.0,
                value=float(parsed.get("fees", 0) or 0),
                step=0.5,
                key="voice_confirm_fees"
            )
            voice_shipping = vc2.number_input(
                "Shipping ($)",
                min_value=0.0,
                value=float(parsed.get("shipping_cost", 0) or 0),
                step=0.5,
                key="voice_confirm_shipping"
            )
            voice_notes = st.text_input(
                "Notes",
                value=str(parsed.get("notes", "") or ""),
                key="voice_confirm_notes"
            )

            selected_voice_lots = voice_product_stock[
                voice_product_stock["size"].fillna("").astype(str).str.upper()
                == normalize_size(voice_size)
            ]
            voice_weighted_cost = float(
                (
                    selected_voice_lots["quantity_in_stock"]
                    * selected_voice_lots["unit_cost"]
                ).sum()
                / int(selected_voice_lots["quantity_in_stock"].sum())
            )

            voice_profit = (
                voice_qty * voice_price
                - voice_qty * voice_weighted_cost
                - voice_fees
                - voice_shipping
            )
            st.metric("Estimated profit", f"${voice_profit:,.2f}")

            confirm_col, cancel_col = st.columns(2)

            if confirm_col.button(
                "✅ Confirm & record sale",
                type="primary",
                key="confirm_voice_sale"
            ):
                try:
                    add_sale_for_product_size(
                        voice_product_row,
                        voice_size,
                        voice_qty,
                        voice_price,
                        voice_fees,
                        voice_shipping,
                        voice_notes
                    )
                    st.session_state.pop("voice_sale_parsed", None)
                    st.session_state.pop("voice_sale_transcript", None)
                    st.success(
                        f"Sale recorded: {voice_qty} × {voice_size} "
                        f"{product_label(voice_product_row)} at ${voice_price:.2f} each."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

            if cancel_col.button("Cancel voice sale", key="cancel_voice_sale"):
                st.session_state.pop("voice_sale_parsed", None)
                st.session_state.pop("voice_sale_transcript", None)
                st.rerun()

        st.markdown("---")
        st.markdown("### Manual sale entry")

        product_options = {
            f"{product_label(r)} [{idx + 1}]": idx
            for idx, r in products.iterrows()
        }
        selected_product_label = st.selectbox("Jersey", list(product_options.keys()))
        product_row = products.iloc[product_options[selected_product_label]]

        product_mask = pd.Series(True, index=available.index)
        for col in pcols:
            product_mask &= (
                available[col].fillna("").astype(str)
                == str(product_row[col] or "")
            )
        product_stock = available[product_mask].copy()

        size_stock = (
            product_stock.assign(size=product_stock["size"].fillna("").astype(str).str.upper())
            .groupby("size", dropna=False, as_index=False)["quantity_in_stock"]
            .sum()
        )
        size_stock = size_stock[size_stock["quantity_in_stock"] > 0]

        size_options = {
            f"{r['size'] or 'Unspecified'} — {int(r['quantity_in_stock'])} in stock": r["size"]
            for _, r in size_stock.iterrows()
        }
        selected_size_label = st.selectbox("Size sold", list(size_options.keys()))
        selected_size = size_options[selected_size_label]

        selected_size_lots = product_stock[
            product_stock["size"].fillna("").astype(str).str.upper()
            == normalize_size(selected_size)
        ]
        total_size_stock = int(selected_size_lots["quantity_in_stock"].sum())

        weighted_cost = float(
            (selected_size_lots["quantity_in_stock"] * selected_size_lots["unit_cost"]).sum()
            / total_size_stock
        )

        c1, c2 = st.columns(2)
        qty = c1.number_input(
            "Quantity sold",
            min_value=1,
            max_value=total_size_stock,
            value=1,
            step=1
        )
        sale_price_each = c2.number_input(
            "Sale price per jersey ($)",
            min_value=0.0,
            value=max(weighted_cost, 1.0),
            step=1.0
        )
        fees = c1.number_input("Marketplace/payment fees ($)", min_value=0.0, value=0.0, step=0.5)
        shipping = c2.number_input("Shipping cost you paid ($)", min_value=0.0, value=0.0, step=0.5)
        notes = st.text_input("Sale notes", placeholder="Buyer, marketplace, order #, etc.")

        expected_profit = (
            qty * sale_price_each
            - qty * weighted_cost
            - fees
            - shipping
        )

        a, b = st.columns(2)
        a.metric(f"Size {selected_size or 'Unspecified'} stock", total_size_stock)
        b.metric("Estimated profit from this sale", f"${expected_profit:,.2f}")

        st.caption(
            "If the same jersey and size exists in multiple purchase lots, "
            "the app deducts the oldest stock first (FIFO)."
        )

        if st.button("Record sale", type="primary"):
            try:
                add_sale_for_product_size(
                    product_row,
                    selected_size,
                    qty,
                    sale_price_each,
                    fees,
                    shipping,
                    notes
                )
                st.success(
                    f"Sale recorded. {qty} × size {selected_size or 'Unspecified'} "
                    "was removed from inventory."
                )
                st.rerun()
            except Exception as e:
                st.error(str(e))


with tabs[4]:
    st.subheader("Sales history")
    sales = read_sales()
    if sales.empty:
        st.info("No sales recorded yet.")
    else:
        st.dataframe(
            sales,
            use_container_width=True,
            hide_index=True,
            column_config={
                "unit_cost": st.column_config.NumberColumn(format="$%.2f"),
                "sale_price_each": st.column_config.NumberColumn(format="$%.2f"),
                "fees": st.column_config.NumberColumn(format="$%.2f"),
                "shipping_cost": st.column_config.NumberColumn(format="$%.2f"),
                "revenue": st.column_config.NumberColumn(format="$%.2f"),
                "cogs": st.column_config.NumberColumn(format="$%.2f"),
                "profit": st.column_config.NumberColumn(format="$%.2f"),
            }
        )

        csv = sales.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download sales CSV",
            data=csv,
            file_name="jersey_sales.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.subheader("Edit / Delete Sale")

        def sale_label(r):
            who = f"{r['player_name']} #{r['jersey_number']}".strip()
            return (
                f"Sale {int(r['id'])} — {r['sold_at']} — {r['team']} — {who} — "
                f"Size {r['size']} — Qty {int(r['quantity'])} — "
                f"${float(r['sale_price_each']):.2f} each"
            )

        sale_options = {
            sale_label(r): int(r["id"])
            for _, r in sales.iterrows()
        }

        selected_sale_label = st.selectbox(
            "Select sale",
            list(sale_options.keys()),
            key="edit_sale_select"
        )
        selected_sale_id = sale_options[selected_sale_label]
        selected_sale = sales[sales["id"] == selected_sale_id].iloc[0]

        st.caption(
            f"Inventory item: {selected_sale['team']} — "
            f"{selected_sale['player_name']} #{selected_sale['jersey_number']} — "
            f"Size {selected_sale['size']}"
        )

        c1, c2 = st.columns(2)
        edit_qty = c1.number_input(
            "Quantity",
            min_value=1,
            value=int(selected_sale["quantity"]),
            step=1,
            key=f"edit_qty_{selected_sale_id}"
        )
        edit_price = c2.number_input(
            "Sale price per jersey ($)",
            min_value=0.0,
            value=float(selected_sale["sale_price_each"]),
            step=1.0,
            key=f"edit_price_{selected_sale_id}"
        )
        edit_fees = c1.number_input(
            "Marketplace/payment fees ($)",
            min_value=0.0,
            value=float(selected_sale["fees"]),
            step=0.5,
            key=f"edit_fees_{selected_sale_id}"
        )
        edit_shipping = c2.number_input(
            "Shipping cost ($)",
            min_value=0.0,
            value=float(selected_sale["shipping_cost"]),
            step=0.5,
            key=f"edit_shipping_{selected_sale_id}"
        )
        edit_notes = st.text_input(
            "Notes",
            value=str(selected_sale["notes"] or ""),
            key=f"edit_notes_{selected_sale_id}"
        )

        edited_profit = (
            edit_qty * edit_price
            - edit_qty * float(selected_sale["unit_cost"])
            - edit_fees
            - edit_shipping
        )
        st.metric("Updated profit", f"${edited_profit:,.2f}")

        save_col, delete_col = st.columns(2)

        if save_col.button(
            "Save sale changes",
            type="primary",
            key=f"save_sale_{selected_sale_id}"
        ):
            try:
                update_sale(
                    selected_sale_id,
                    edit_qty,
                    edit_price,
                    edit_fees,
                    edit_shipping,
                    edit_notes
                )
                st.success("Sale updated. Inventory and profit totals were recalculated.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

        confirm_delete = delete_col.checkbox(
            "Confirm delete",
            key=f"confirm_delete_sale_{selected_sale_id}"
        )
        if delete_col.button(
            "Delete sale",
            disabled=not confirm_delete,
            key=f"delete_sale_{selected_sale_id}"
        ):
            try:
                delete_sale(selected_sale_id)
                st.success("Sale deleted. The sold quantity was returned to inventory.")
                st.rerun()
            except Exception as e:
                st.error(str(e))

