import os, sqlite3
from supabase import create_client
DB=os.getenv("SQLITE_DB","jersey_inventory.db")
sb=create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
con=sqlite3.connect(DB); con.row_factory=sqlite3.Row
inv=[dict(r) for r in con.execute("select * from inventory order by id")]
sales=[dict(r) for r in con.execute("select * from sales order by id")]
print(f"Migrating {len(inv)} inventory rows and {len(sales)} sales rows...")
if inv: sb.table("inventory").upsert(inv,on_conflict="id").execute()
if sales: sb.table("sales").upsert(sales,on_conflict="id").execute()
print("Migration complete. Verify counts in Supabase before changing anything else.")
