
# Jersey Inventory & Profit Tracker

A small Streamlit website for a jersey resale business.

## What it does

- Upload a supplier order-sheet screenshot/photo.
- Uses an OpenAI vision model to extract jersey rows.
- Lets you review/edit the extracted data before saving.
- Tracks inventory by lot, including size, player, version, quantity and cost.
- Tracks units sold and remaining stock.
- Records selling price, marketplace/payment fees and shipping cost.
- Calculates revenue, COGS and realized profit.
- Shows simple dashboard metrics and profit by team.
- Exports inventory and sales to CSV.
- Stores data locally in SQLite.

## Cost interpretation

The importer is specifically prompted for supplier sheets like the attached example, where the right-most "Price" appears to be a **line total**.

Example:
- QTY = 3
- Price = $39
- line cost = $39
- unit cost = $13

You can always correct the AI output in the review table before saving it.

## Run it

1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Create a virtual environment if desired.
4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Set your OpenAI API key.

macOS/Linux:
```bash
export OPENAI_API_KEY="your-key"
```

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
```

6. Start the website:

```bash
streamlit run app.py
```

Streamlit will show a local URL, normally `http://localhost:8501`.

## Files

- `app.py` — complete website
- `requirements.txt` — dependencies
- `.env.example` — environment-variable example
- `jersey_inventory.db` — created automatically on first run

## Data model

Each supplier import is stored as an inventory lot. A sale points to a specific lot, so profit uses the actual cost for that lot rather than an average guessed cost.

Realized profit:

`quantity sold × selling price - quantity sold × unit cost - fees - shipping`

## Good next upgrades

The current version is intentionally simple and local. Natural next upgrades are:
- user login
- cloud hosting
- PostgreSQL/Supabase instead of local SQLite
- barcode/SKU generation
- customer/order tracking
- pending/paid/shipped sale states
- automatic duplicate detection on supplier sheet uploads
- product photos attached to each inventory item
