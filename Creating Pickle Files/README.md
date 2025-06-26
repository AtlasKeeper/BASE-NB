# Historical Options and Index Data Extraction

This script is something we built to help extract options and index data directly from our PostgreSQL database and save it month-by-month in `.pkl` format. It was created specifically for Indian indices like:

* **NIFTY**
* **BANKNIFTY**
* **SENSEX**
* **FINNIFTY**

The goal was to make the entire data extraction process hands-free — it reads expiry mappings from Excel, grabs the strike range dynamically, pulls both index and options data, and stores everything cleanly month-by-month in a fast-to-load format.

---

##  What It Can Do

* Pulls index data from `<INDEX>_IDX` tables
* Pulls options data from `<INDEX>_OPT_<YEAR>` tables
* Automatically handles multiple years and months
* Aligns data using expiry dates from Excel
* Dynamically calculates the relevant strike price range
* Saves clean monthly `.pkl` files for later use

---

## What You’ll Need (Dependencies)

Install these Python packages:

```bash
pip install pandas numpy psycopg2 tqdm openpyxl
```

---

## Project Layout (Suggested Structure)

```
project_folder/
├── script.py
├── README.md
├── data/                     # where output .pkl files go
│   ├── NIFTY_202301.pkl
│   ├── BANKNIFTY_202212.pkl
│   └── ...
├── expiry_dates/            # expiry date Excel files
│   ├── NIFTY market dates.xlsx
│   ├── BANKNIFTY market dates.xlsx
│   └── ...
```

---

## ⚙Setup and Config (Things to Change)

### 1. Choose the Index to Process

```python
stock = 'NIFTY'  # Change to BANKNIFTY, FINNIFTY, or SENSEX if needed
```

### 2. Define File Paths

```python
expiry_file_path = '/home/yourname/NIFTY market dates.xlsx'
output_path = '/home/yourname/output_folder/'
```

> The Excel file should have `Date` and `ExpiryDate` columns.

### 3. Set the Time Period

```python
start_date = '2021-06-01'
end_date = '2025-04-30'
```

> The script loops through each month between these dates.

---

## How the DB Query Function Works

This function handles database connections and executes your SQL queries:

```python
def postgresql_query(query, input_tuples=None):
    # connects to DB, runs query, returns list of tuples
```

> Make sure to update your PostgreSQL credentials inside this function.

---

## How Options Data Is Pulled

```python
def get_option_data(table_name, mapped_days, low_strike, high_strike):
    # pulls CE/PE options data for each date from mapped_days within the given strike range
```

* Uses expiry mapping from Excel
* Warns if no data is found for a given day

---

## How Dates Are Generated

* Loops through all selected years and months
* Builds complete daily date list
* Filters based on the selected `start_date` and `end_date`
* Matches those dates to expiry mapping using Excel

---

## Monthly Workflow

For each month between the start and end dates:

1. Generates daily dates
2. Filters using `start_date` and `end_date`
3. Loads expiry mapping for filtered days
4. Pulls index data from `<STOCK>_IDX` (only weekdays)
5. Determines strike range using min/max prices
6. Pulls options data from `<STOCK>_OPT_<YEAR>`
7. Cleans and saves it as `STOCK_YYYYMM.pkl`

---

## Output Format

Each `.pkl` file is saved to the path you configured.
Example files:

```
NIFTY_202301.pkl
BANKNIFTY_202212.pkl
FINNIFTY_202401.pkl
```

Contents of each file:

```
['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'High', 'Low', 'Close', 'OI', 'Volume', 'Ticker']
```

> Use `pd.read_pickle()` to load these later for analysis or backtesting.

---

## Sample Config

```python
stock = 'BANKNIFTY'
expiry_file_path = '/home/user/BANKNIFTY market dates.xlsx'
output_path = '/home/user/bnf_data/'
start_date = '2022-01-01'
end_date = '2024-12-31'
```

---

## Notes

* Each year should have a separate options table like `BANKNIFTY_OPT_2023`, `NIFTY_OPT_2024`, etc.
* If you want to use second-week expiry, just point to a different column in your Excel
* Use strike steps of 50 for NIFTY/FINNIFTY and 100 for BANKNIFTY/SENSEX

---

## Credits

Built and maintained by Algo Department NewBerry to support backtesting, analytics, and strategy development. Feel free to fork, modify, or extend it.

---

## ⚠Final Tip

Always test this on a small date range first to confirm that your database tables and expiry mappings are correctly aligned. Then scale up to your full date range.
