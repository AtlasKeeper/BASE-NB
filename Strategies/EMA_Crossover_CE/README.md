# EMA Crossover Intraday Options Buying Strategy (CE)

This repository implements an intraday trading strategy that buys **Call Options (CE)** based on **EMA crossover signals** from the underlying index (e.g., NIFTY). It includes dynamic **trailing stop-loss (TSL)** logic to manage risk and lock in profits.

---

## Strategy Logic

### Entry Criteria

* Uses resampled intraday data (e.g., 5-minute candles).
* Calculates:

  * **Fast EMA** (e.g., 5-period)
  * **Slow EMA** (e.g., 20-period)
* Entry signal is triggered when:

  1. **Fast EMA crosses above Slow EMA**
  2. The candle **closes above both EMAs**
  3. The **next candle is bullish** (Close > Open)
  4. **Candle size** is small (e.g., less than 40 points)

On this signal, the strategy buys a **Call Option (CE)** near the ATM strike.

---

## Exit Strategy — TSL Logic

After entering a trade, the exit is handled by a **dynamic trailing stop-loss (TSL)** system:

* **Initial Stoploss & Target** are defined in premium points.
* If the premium hits the initial target, a **TSL is activated**:

  * The target and stop-loss keep increasing as the premium rises.
  * The TSL protects profits and locks in gains on reversals.
* The trade also exits if:

  * Index hits an **initial stop-level**,
  * **End-of-day (EOD)** time is reached,
  * Or any **reversal threshold** is triggered.

---

## Parameters

The strategy supports tuning multiple parameters:

| Parameter     | Description                                 |
| ------------- | ------------------------------------------- |
| `TIME_FRAME`  | Candle timeframe (e.g., 5, 15 min)          |
| `FAST_EMA`    | Fast EMA period                             |
| `SLOW_EMA`    | Slow EMA period                             |
| `LAST_ENTRY`  | Latest entry time allowed                   |
| `TARGET_TH`   | Initial premium target                      |
| `STOPLOSS_TH` | Index-based fallback stop-loss              |
| `STRIKE`      | ATM offset for strike price selection       |
| `I_T`         | Initial premium target in points            |
| `L_P`         | Base stop-loss below premium entry          |
| `INC_T`       | Target increment for each TSL trigger       |
| `INC_S`       | Stop-loss trail increment after each target |

---

## Output

Each trade is logged with:

* Entry/Exit Time
* Entry/Exit Premium
* Index Price at Entry
* Option Strike
* Trade Type (CE only)
* Exit Reason (`TSL`, `Stoploss`, `Target`, `Universal`)
* P\&L
* DTE (Days to Expiry)

---

## Folder Structure

```
.
├── utils/
│   └── indicators.py           # EMA calculation
│   └── trailing_stoploss.py    # Core TSL logic
├── data/
│   └── option_data.pkl         # Option OHLC data
│   └── index_data.pkl          # Nifty/BankNifty index OHLC
├── main.py                     # Entry point for backtest
├── results/
│   └── trade_sheet.csv         # Output trades log
└── README.md
```

---

##  Notes

* Strategy currently supports **only CE (Call Option)** logic.
* Can be easily extended to support PE entries.
* Designed for **index options (Nifty, BankNifty)**.
* Trades are **intraday only** — exits before EOD.

---

## Dependencies

* `pandas`
* `numpy`
* `ta-lib`
* `datetime`
* `matplotlib` (optional for plotting)
* `pickle` (for data I/O)
  
---

