# Options Backtesting Framework

This is a modular Python backtesting framework for options strategies on Indian indices like NIFTY and BANKNIFTY. The system loads index and option data from a PostgreSQL database and `.pkl` files, processes strike premiums, and computes position-level metrics such as net premium and days to expiry.

---

## Features

- Efficient multi-date option data loading from pickled files
- Index data pull from PostgreSQL
- Customizable strike selection and premium calculation logic
- Flexible support for CE/PE legs and lot/brokerage modeling
- Timeframe resampling, expiry filtering, and strategy utilities

---

## Installation

1. **Clone this repo and setup the environment:**
   ```bash
   git clone https://github.com/your_repo/options-backtester.git
   cd options-backtester
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt

- Dependencies
  -numpy
  -pandas
  -psycopg2
  -tqdm
  -talib
  -PostgreSQL client setup
  -Optionally: multiprocessing (comes with stdlib)

Project Structure

options-backtester/
│
├── Common_Functions/
│   └── utils.py                  # Contains reusable logic like TSL, resampling, expiry logic, etc.
│
├── data/
│   └── *.pkl                     # Option chain data in monthly pickle files
│
├── main_script.py               # Your main code file (shown above)
├── README.md                    # This file
└── requirements.txt             # All dependencies

## How to Use
1. Pull Option Chain Data
   ```bash
   option_data = pull_options_data_d(start_date, end_date, option_data_path, stock='NIFTY')

3. Pull Index Data from PostgreSQL
   ```bash
   index_data = pull_index_data(start_date_idx='20240101', end_date_idx='20240331', stock='NIFTY')
   
5. Strike Selection at Entry Time
   ```bash
   ce_atm, pe_atm, ce_otm, ce_otm_prem, pe_otm, pe_otm_prem = get_strike(ATM=22350, minute=dt.datetime(2024, 2, 1, 9, 30), daily_option_data=option_data)
   
7. Calculate Net Premiums
   ```bash
   result_df = get_final_premium(premium_data=df, option_data=option_data, RATIO=(1, 2))
   
##Core Functions
pull_options_data_d()	Loads pickled monthly option data filtered by date and symbol
pull_index_data()	Fetches 1-min OHLC index data from PostgreSQL for a date range
get_strike()	Selects CE/PE ATM and OTM strikes based on open price proximity to a multiplier
get_final_premium()	Computes premium for CE/PE ATM/OTM legs based on position and lot configuration
resample_data()	Aggregates OHLC data to a specified time frame like 5min/15min/hourly
nearest_multiple()	Utility to round prices to the nearest valid strike multiple (e.g., 50 for NIFTY)
postgresql_query()	Wrapper for safely querying the PostgreSQL database

##Configuration Notes
Lot Size and Brokerage:
These are assumed global variables:

LOT_SIZE = 50      # e.g., for NIFTY
brokerage = 4.5    # per leg
STRIKE = 0.7       # 70% rule for OTM leg premium

