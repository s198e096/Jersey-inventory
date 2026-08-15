
import os
import io
import re
import json
import base64
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

DB_PATH = os.getenv("JERSEY_DB_PATH", "jersey_inventory.db")
OPENAI_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-5-mini")


# ----------------------------
# Database helpers
# ----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                import_batch TEXT,
                source_file TEXT,
                order_number TEXT,
                version TEXT,
                team TEXT,
                player_name TEXT,
                jersey_number TEXT,
                size TEXT,
                quantity_received INTEGER NOT NULL,
                unit_cost REAL NOT NULL,
                line_cost REAL NOT NULL,
                notes TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sold_at TEXT NOT NULL,
                inventory_id INTEGER NOT NULL,
                quantity INTEGER NOT NULL,
                sale_price_each REAL NOT NULL,
                fees REAL NOT NULL DEFAULT 0,
                shipping_cost REAL NOT NULL DEFAULT 0,
                notes TEXT,
                FOREIGN KEY(inventory_id) REFERENCES inventory(id)
            )
        """)
        conn.commit()


def read_inventory():
    with get_conn() as conn:
        inv = pd.read_sql_query("""
            SELECT
                i.*,
                COALESCE(SUM(s.quantity), 0) AS quantity_sold,
                i.quantity_received - COALESCE(SUM(s.quantity), 0) AS quantity_in_stock
            FROM inventory i
            LEFT JOIN sales s ON s.inventory_id = i.id
            GROUP BY i.id
            ORDER BY i.id DESC
        """, conn)
    return inv


def read_sales():
    with get_conn() as conn:
        df = pd.read_sql_query("""
            SELECT
                s.id,
                s.sold_at,
                s.inventory_id,
                i.team,
                i.player_name,
                i.jersey_number,
                i.version,
                i.size,
                s.quantity,
                i.unit_cost,
                s.sale_price_each,
                s.fees,
                s.shipping_cost,
                (s.quantity * s.sale_price_each) AS revenue,
                (s.quantity * i.unit_cost) AS cogs,
                ((s.quantity * s.sale_price_each)
                    - (s.quantity * i.unit_cost)
                    - s.fees
                    - s.shipping_cost) AS profit,
                s.notes
            FROM sales s
            JOIN inventory i ON i.id = s.inventory_id
            ORDER BY s.id DESC
        """, conn)
    return df


def add_inventory_rows(df, source_file):
    batch = datetime.now().strftime("%Y%m%d-%H%M%S")
    now = datetime.now().isoformat(timespec="seconds")
    with get_conn() as conn:
        for _, row in df.iterrows():
            qty = int(row.get("quantity", 0) or 0)
            line_cost = float(row.get("line_cost", 0) or 0)
            unit_cost = float(row.get("unit_cost", 0) or 0)
            if qty > 0 and unit_cost <= 0 and line_cost > 0:
                unit_cost = line_cost / qty
            if qty > 0 and line_cost <= 0 and unit_cost > 0:
                line_cost = unit_cost * qty

            conn.execute("""
                INSERT INTO inventory (
                    created_at, import_batch, source_file, order_number, version,
                    team, player_name, jersey_number, size,
                    quantity_received, unit_cost, line_cost, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now,
                batch,
                source_file,
                str(row.get("order_number", "") or ""),
                str(row.get("version", "") or ""),
                str(row.get("team", "") or ""),
                str(row.get("player_name", "") or ""),
                str(row.get("jersey_number", "") or ""),
                str(row.get("size", "") or ""),
                qty,
                unit_cost,
                line_cost,
                str(row.get("notes", "") or "")
            ))
        conn.commit()


def delete_inventory_row(inv_id):
    with get_conn() as conn:
        sale_count = conn.execute(
            "SELECT COUNT(*) FROM sales WHERE inventory_id = ?", (inv_id,)
        ).fetchone()[0]
        if sale_count:
            raise ValueError("This inventory lot already has sales and cannot be deleted.")
        conn.execute("DELETE FROM inventory WHERE id = ?", (inv_id,))
        conn.commit()



def normalize_size(value):
    return str(value or "").strip().upper()


def split_size_tokens(value):
    raw = normalize_size(value)
    if not raw:
        return []
    raw = raw.replace("2XL", "XXL").replace("3XL", "XXXL")
    tokens = [t.strip() for t in re.split(r"[/,;|]+|\s+", raw) if t.strip()]
    valid = {"XS", "S", "M", "L", "XL", "XXL", "XXXL"}
    return tokens if len(tokens) > 1 and all(t in valid for t in tokens) else []


def expand_evenly_split_sizes(df):
    expanded = []
    for _, row in df.iterrows():
        row = row.copy()
        sizes = split_size_tokens(row.get("size", ""))
        qty = int(row.get("quantity", 0) or 0)

        if sizes and qty == len(sizes):
            total_line_cost = float(row.get("line_cost", 0) or 0)
            unit_cost = float(row.get("unit_cost", 0) or 0)
            if unit_cost <= 0 and qty > 0 and total_line_cost > 0:
                unit_cost = total_line_cost / qty

            for size in sizes:
                new_row = row.copy()
                new_row["size"] = size
                new_row["quantity"] = 1
                new_row["unit_cost"] = unit_cost
                new_row["line_cost"] = unit_cost
                note = str(new_row.get("notes", "") or "").strip()
                new_row["notes"] = f"{note} Auto-split from combined size row.".strip()
                expanded.append(new_row)
        else:
            expanded.append(row)

    return pd.DataFrame(expanded, columns=df.columns)


def product_key_columns():
    return ["team", "player_name", "jersey_number", "version"]


def add_sale_for_product_size(product_row, size, quantity, sale_price_each, fees, shipping_cost, notes):
    inv = read_inventory()

    mask = pd.Series(True, index=inv.index)
    for col in product_key_columns():
        mask &= inv[col].fillna("").astype(str).eq(str(product_row[col] or ""))
    mask &= inv["size"].fillna("").astype(str).str.upper().eq(normalize_size(size))
    matches = inv[mask & (inv["quantity_in_stock"] > 0)].copy()

    if matches.empty:
        raise ValueError(f"No {size} jerseys are currently in stock.")

    total_available = int(matches["quantity_in_stock"].sum())
    if quantity > total_available:
        raise ValueError(f"Only {total_available} unit(s) of size {size} are currently in stock.")

    matches["created_at_dt"] = pd.to_datetime(matches["created_at"], errors="coerce")
    matches = matches.sort_values(["created_at_dt", "id"], ascending=[True, True])

    remaining = int(quantity)
    allocations = []
    for _, lot in matches.iterrows():
        if remaining <= 0:
            break
        take = min(remaining, int(lot["quantity_in_stock"]))
        allocations.append((int(lot["id"]), take))
        remaining -= take

    with get_conn() as conn:
        for inv_id, qty_take in allocations:
            proportion = qty_take / int(quantity)
            conn.execute("""
                INSERT INTO sales (
                    sold_at, inventory_id, quantity, sale_price_each,
                    fees, shipping_cost, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(timespec="seconds"),
                inv_id,
                qty_take,
                float(sale_price_each),
                float(fees) * proportion,
                float(shipping_cost) * proportion,
                notes
            ))
        conn.commit()



def split_existing_inventory_lot(inv_id, size_quantities):
    """
    Split an existing combined-size inventory lot into separate size-specific lots.
    Only unsold inventory can be split safely.
    """
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM inventory WHERE id = ?",
            (int(inv_id),)
        ).fetchone()

        if row is None:
            raise ValueError("Inventory lot not found.")

        sold_qty = conn.execute(
            "SELECT COALESCE(SUM(quantity), 0) FROM sales WHERE inventory_id = ?",
            (int(inv_id),)
        ).fetchone()[0]

        if int(sold_qty) > 0:
            raise ValueError(
                "This lot already has recorded sales. Split/Edit Sizes only supports unsold lots "
                "so previous sales remain historically accurate."
            )

        cleaned = {
            normalize_size(size): int(qty)
            for size, qty in size_quantities.items()
            if normalize_size(size) and int(qty) > 0
        }

        if not cleaned:
            raise ValueError("Enter at least one size with a quantity greater than zero.")

        original_qty = int(row["quantity_received"])
        new_qty = sum(cleaned.values())

        if new_qty != original_qty:
            raise ValueError(
                f"Size quantities must add up to the original quantity ({original_qty}). "
                f"Current total is {new_qty}."
            )

        original_line_cost = float(row["line_cost"])
        original_unit_cost = float(row["unit_cost"])

        # Remove the original combined-size lot and replace it with size-specific lots.
        conn.execute("DELETE FROM inventory WHERE id = ?", (int(inv_id),))

        for size, qty in cleaned.items():
            line_cost = original_unit_cost * qty
            conn.execute("""
                INSERT INTO inventory (
                    created_at, import_batch, source_file, order_number, version,
                    team, player_name, jersey_number, size,
                    quantity_received, unit_cost, line_cost, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["created_at"],
                row["import_batch"],
                row["source_file"],
                row["order_number"],
                row["version"],
                row["team"],
                row["player_name"],
                row["jersey_number"],
                size,
                qty,
                original_unit_cost,
                line_cost,
                (
                    (row["notes"] or "") +
                    f" Split from original inventory lot {inv_id}."
                ).strip()
            ))

        conn.commit()


def inventory_by_size(inv):
    if inv.empty:
        return inv

    group_cols = product_key_columns() + ["size"]
    qty = (
        inv.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            quantity_received=("quantity_received", "sum"),
            quantity_sold=("quantity_sold", "sum"),
            quantity_in_stock=("quantity_in_stock", "sum"),
        )
    )
    values = (
        inv.assign(inventory_value=inv["quantity_in_stock"] * inv["unit_cost"])
        .groupby(group_cols, dropna=False, as_index=False)["inventory_value"]
        .sum()
    )
    return qty.merge(values, on=group_cols, how="left").sort_values(group_cols)


def add_sale(inventory_id, quantity, sale_price_each, fees, shipping_cost, notes):
    inv = read_inventory()
    match = inv[inv["id"] == inventory_id]
    if match.empty:
        raise ValueError("Inventory item not found.")
    available = int(match.iloc[0]["quantity_in_stock"])
    if quantity > available:
        raise ValueError(f"Only {available} unit(s) are currently in stock.")

    with get_conn() as conn:
        conn.execute("""
            INSERT INTO sales (
                sold_at, inventory_id, quantity, sale_price_each,
                fees, shipping_cost, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(timespec="seconds"),
            int(inventory_id),
            int(quantity),
            float(sale_price_each),
            float(fees),
            float(shipping_cost),
            notes
        ))
        conn.commit()


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

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    client = OpenAI()

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
init_db()

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
