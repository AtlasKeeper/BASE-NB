# Historical Options and Index Data Extraction

This Python project automates the extraction, transformation, and monthly storage of historical **options** and **index** data for:

* **NIFTY**
* **BANKNIFTY**
* **SENSEX**
* **FINNIFTY**

The data is pulled from a PostgreSQL database and saved as `.pkl` (Pickle) files after aligning with market expiry dates.

---

## Features

* Queries index and options data from PostgreSQL
* Dynamically handles multiple years and monthly partitions
* Maps expiry dates to trading days via Excel
* Supports weekly/monthly/second-week expiry mapping
* Saves clean `.pkl` files month-wise for faster access

---

## Requirements

Install dependencies:

```bash
pip install pandas numpy psycopg2 tqdm openpyxl
```

---

## Folder Structure

```
project/
├── script.py
├── README.md
├── data/
│   ├── NIFTY_202301.pkl
│   ├── BANKNIFTY_202212.pkl
│   └── ...
├── expiry_dates/
│   ├── NIFTY market dates.xlsx
│   ├── BANKNIFTY market dates.xlsx
│   └── ...
```

---

## ⚙Configuration

### 1. Select Stock Index

```python
stock = 'NIFTY'  # or BANKNIFTY / SENSEX / FINNIFTY
```

### 2. Paths

```python
expiry_file_path = '/home/user/NIFTY market dates.xlsx'
output_path = '/home/user/output_folder/'
```

### 3. Date Range

```python
start_date = '2021-06-01'
end_date = '2025-04-30'
```

> Make sure the expiry file has `Date`, `ExpiryDate` columns.

---

## Database Querying

### PostgreSQL Query Wrapper

```python
def postgresql_query(query, input_tuples=None):
    # Runs SELECT query and returns result as list of tuples
```

> Edit DB credentials before use.

### Fetching Options Data

```python
def get_option_data(table_name, mapped_days, low_strike, high_strike):
    # Returns filtered options data (CE + PE) from PostgreSQL
```

---

## How Dates are Processed

* Dates are generated for all months/years in a loop
* Filtered based on `start_date` and `end_date`
* Expiry mappings are read from Excel and aligned with dates

---

## Workflow Per Month

1. Generate dates
2. Load expiry mapping
3. Query index data from `<STOCK>_IDX`
4. Determine strike range from min/max prices
5. Query options data from `<STOCK>_OPT_<YYYY>`
6. Clean and save as `.pkl`

---

## Output

* Pickle files saved as:

```
NIFTY_202301.pkl
BANKNIFTY_202212.pkl
... etc.
```

Each file contains:

```
['Date', 'Time', 'ExpiryDate', 'StrikePrice', 'Type', 'Open', 'High', 'Low', 'Close', 'OI', 'Volume', 'Ticker']
```

---

## Example Configuration

```python
stock = 'BANKNIFTY'
expiry_file_path = '/home/user/BANKNIFTY market dates.xlsx'
output_path = '/home/user/bnf_data/'
start_date = '2022-01-01'
end_date = '2024-12-31'
```

---

## Notes

* Works best when each year has a separate options table (`*_OPT_2023`, `*_OPT_2024`, etc.)
* Supports second-week expiry via simple toggle
* Adjust strike range logic for 50 vs 100 intervals

---

## Author / Credits

Maintained by the Data Research Team.

---

## ⚠️ Disclaimer

Ensure correct PostgreSQL credentials and valid table structures before running the script. Always test on a small batch of data.
