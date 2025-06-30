# NIFTY/BANKNIFTY/FINNIFTY Options Strategy Backtester

This is the backtesting script designed to simulate option selling strategies (e.g., short straddles or strangles) using minute-level index and option chain data. It supports custom strike selection, premium calculation, expiry management, and performance filtering across date ranges and parameter combinations.

---

## Features

-  Supports **ATM/OTM** logic for CE/PE legs
-  Pulls index OHLC data from **PostgreSQL**
-  Reads options data from **Pickle files**
-  Applies **brokerage and lot size** logic
-  Generates detailed **trade sheets**
-  Creates **filter summaries** for strategy evaluation
-  Plug-and-play architecture with helper functions
-  Multiprocessing-ready for large-scale batch testing

---

##  File Structure
```
/
├── main.py # Parameter loop to generate trades
├── trade_sheet_creator.py # Strategy simulation logic
├── data/
│ ├── option_data/ # Pickle files of option chain data
│ └── mapped_days.csv # Trading calendar + expiry mapping
├── outputs/ # Trade sheet and filter CSVs
├── Common_Functions/
│ └── utils.py # Shared helper utilities
├── README.md # This file
```
---

## Requirements

- Python 3.8+
- `pandas`, `numpy`, `psycopg2`, `tqdm`
- `TA-Lib`
- PostgreSQL running with relevant index tables


## Core Logic
```
pull_options_data_d(...)
Loads .pkl option chain files filtered by month/year and ticker name.

pull_index_data(...)
Queries index OHLC data from PostgreSQL for a given symbol and date range.

get_strike(...)
Returns ATM premium and OTM strikes selected based on a target premium multiplier (e.g., 1.03x).

get_final_premium(...)
Merges CE/PE ATM and OTM premium legs, applies brokerage and lot sizing rules, and computes net premium.

trade_sheet_creator(...)
Simulates trades using strategy parameters, logs entry/exit, computes DTE buckets, and saves:
```

- Trade data CSV

- Filter summary CSV per parameter set

## Trade Sheet Format
```
Each trade is represented with both entry (Short) and exit (Long) legs.

Column	Description
Date	Trade date
Position	1 = Entry, 0 = Exit
Action	"Short" or "Long"
CE_Time / PE_Time	Entry/Exit timestamp
Index Value	Index open price at entry
CE_OTM / PE_OTM	Strike prices selected
CE/PE Premiums	Premium values at entry/exit
DaysToExpiry	DTE at time of trade
EXIT_TYPE	TSL, EOD, etc. (exit reason)
Premium	Signed P&L per leg
```

## Sample Strategy File Name
NIFTY_candle_5_strike_1.03_entry_09,30_exit_15,15.csv
This corresponds to:

5-minute candles

3% OTM strike selection

Entry at 09:30, Exit at 15:15

## Workflow Overview
```
flowchart TD;
    A[Mapped Trading Days] --> B[Loop over Days]
    B --> C[Pull Index and Option Data]
    C --> D[Select Strikes using get_strike]
    D --> E[Get Premiums at ENTRY and EXIT]
    E --> F[Apply TSL or EOD Exit]
    F --> G[Append Entry & Exit Rows]
    G --> H[Save Trade Sheet to CSV]
    G --> I[Update Filter DF with DTE]
```
## Batch Backtest Example
```bash
from trade_sheet_creator import trade_sheet_creator

params = [(5, 1.03, "09:30", "15:15"), (15, 1.05, "10:00", "15:20")]

for counter, (tf, strike, entry, exit) in enumerate(params):
    trade_sheet_creator(
        mapped_days, option_data, index_data,
        start_date="2023-01-01", end_date="2023-12-31",
        counter=counter,
        output_folder_path="./outputs/",
        resampled=False,
        TIME_FRAME=tf,
        STRIKE=strike,
        ENTRY=entry,
        EXIT=exit
    )
 ```

## Output Files
outputs/NIFTY_candle_5_strike_1.03_entry_09,30_exit_15,15.csv
Entry/exit data per trade (two rows per trade)

outputs/filter_df1.csv
Row per parameter set

DTE0–DTE4 column flags

Status = 1 if any DTE profitable

## Tips
- Remove expired or holiday dates in mapped_days manually if needed.
- Enable re-entry logic for advanced testing.
- Add rolling metrics (e.g., drawdown, win rate) to the output CSVs for deeper analysis.
- Externalize constants like brokerage, LOT_SIZE in a config file or .env.

# INPUTS PART

This module sets up and executes a **multi-parameter backtest** for index options strategies (e.g., plain vanilla short straddles). It uses a **multiprocessing architecture** to parallelize evaluation of parameter sets over multiple time ranges.

---

## Folder Structure

```bash
/home/newberry3/user/
├── STRATEGY_NAME/
│   └── plain_vanilla/
│       └── NIFTY/
│           └── ND/
│               ├── Filter_Sheets/
│               ├── Trade_Sheets/
│               └── new_done.txt
└── Common_Files/
    └── NIFTY market dates.xlsx
```

---

## Configuration

### trategy & Market Setup

```python
superset = 'plain_vanilla'
stock = 'NIFTY'
option_type = 'ND'
```

* `superset`: Strategy folder name (used in path generation).
* `stock`: One of `NIFTY`, `BANKNIFTY`, `FINNIFTY`, `SENSEX`.
* `option_type`: Custom label for differentiating strategy types.

---

### Market Constants

```python
roundoff = 50
brokerage = 4.5
LOT_SIZE = 25
```

* **Strike round-off**: Step size to determine ATM strikes.
* **Brokerage**: Per leg (flat).
* **Lot size**: Number of units per contract.

---

## Paths

Dynamically generated based on the above configuration.

| Variable             | Description                      |
| -------------------- | -------------------------------- |
| `root_path`          | Base folder for strategy outputs |
| `filter_df_path`     | CSVs of filtered parameter sets  |
| `expiry_file_path`   | Excel mapping of expiry and DTE  |
| `txt_file_path`      | Tracks completed parameter runs  |
| `output_folder_path` | Final output sheets for trades   |
| `option_data_path`   | Source folder of options data    |

---

## Date Ranges

```python
date_ranges = [
    ('2024-06-01', '2024-10-30'),
    ...
    ('2021-06-01', '2021-12-31')
]
```

Each range defines a **backtest window**. The script iterates through all of them in chronological order.

---

## Strategy Parameters

```python
candle_time_frame = ['5T']
entries = ['09:25']
exits = ['15:20']
strikes = [0]
```

* **Time frame**: Candle duration (e.g., `'5T'` for 5-min).
* **Entry/Exit**: Trade times per day.
* **Strike offset**: 0 = ATM; positive values = OTM calls/puts.

> BANKNIFTY/SENSEX strike spacing is doubled automatically.

---

## Parameter Processing Function

```python
def parameter_process(parameter, mapped_days, option_data, df, start_date, end_date, counter, output_folder_path)
```

Handles logic for one parameter set:

* Resamples index data.
* Passes it to `trade_sheet_creator` (user-defined core logic).
* Exports the output to sheet.

---

##  Execution Flow

### 1. Initialize

```python
counter = 0
start_date_idx = earliest start from date_ranges
end_date_idx = latest end from date_ranges
```

* Load expiry mapping Excel.
* Load index data for full period (to avoid re-pulling each time).

---

### 2. Loop Through Date Ranges

```python
for start_date, end_date in date_ranges:
```

For each range:

* Pull option data for `start_date → end_date + 30 days`.
* If it's the **first run**, generate all parameter combinations.
* If not, **load previous filter** (`filter_dfX.csv`) and reuse parameters.

---

### 3. Parameter Filtering

Checks `new_done.txt` to skip already processed combinations.

```python
existing_values = [line.strip() for line in open(txt_file_path)]
parameters = [p for p in parameters if p_string not in existing_values]
```

Each parameter is logged as:

```
NIFTY_candle_5T_strike_09:25_entry_15:20_exit_2024-06-01_2024-10-30
```

---

## Multiprocessing Execution

```python
from functools import partial
with multiprocessing.Pool(processes=12) as pool:
```

* Uses 12 parallel processes.
* Each parameter is mapped to `parameter_process()`.
* `tqdm` tracks progress.
* After completion, logs are appended to `new_done.txt`.

---

## Time Logging

At the end of each range:

```python
print('Time taken to get Initial Tradesheets:', elapsed_time)
```

Final statement confirms script completion.

---

## Output Summary

* Filtered parameter runs go in `Filter_Sheets/`.
* Completed tradesheets in `Trade_Sheets/`.
* Execution metadata tracked in `new_done.txt`.

---

## How to Run

Run directly:

```bash
python main_backtest.py
```

Ensure:

* PostgreSQL is up (if used in pull functions).
* `pull_index_data`, `pull_options_data_d`, `trade_sheet_creator`, and `resample_data` functions are defined.
* Data files exist in their respective folders.

---

## Extending

* Add more `entries`, `exits`, or `strikes` to increase search space.
* Insert custom filters or scoring into `parameter_process`.
* Modularize by moving I/O to separate `io_utils.py`.


