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

Trade data CSV

Filter summary CSV per parameter set

Trade Sheet Format

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

