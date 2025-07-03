# Strategy Analytics & Daily PnL Aggregation - README

This script is designed to process and analyze daily trade P\&L sheets generated from an options strategy backtesting engine. It performs trade aggregation, computes key metrics (drawdown, ROI, sortino ratio), and outputs a ranked analytics report.

---

## Folder Structure

```
/{root_dir}/
    └─ {superset}/
        ├─ {stock}/{option_type}/
        │   ├─ Trade_Sheets/       <- Raw trade data (daily level)
        │   ├─ dailypnl/             <- Computed daily P&L files
        │   └─ Analytics/            <- Final analytics output
```

---

## Configuration

Set the following configuration parameters at the top:

```python
stock = 'NIFTY'                       # Can be 'BANKNIFTY', 'FINNIFTY', 'SENSEX'
option_type = 'ND'                   # Strategy type
superset = 'STRATEGY_NAME/plain_vanilla(VERSION OF YOUR STRATEGY)'
root_dir = '/home/newberry3/user/'
```

Other important configuration constants:

```python
lot_size_dict = {'NIFTY': 75, 'BANKNIFTY': 30, 'FINNIFTY': 65, 'SENSEX': 20}
govt_tc_dict = {'NIFTY': 2.25, 'BANKNIFTY': 3, 'FINNIFTY': 2.25, 'SENSEX': 3}
```

---

## Part 1: Daily P\&L Calculation

This step reads all trade files (`.csv` or `.xls`) from `Trade_Sheets/`, computes:

* Per trade net P\&L using lot size and government charges
* Aggregates P\&L on daily basis
* Saves them to `dailypnl/` as CSV

### Function:

* `get_spread_from_range()` → Determines slippage spread cost based on premium bucket

> Output: `{stock}/{option_type}/dailypnl/*.csv`

---

## Part 2: Profitability & Drawdown Analysis

For each daily P\&L file, this section:

* Filters date range for multi-period windows (31M, 11M, 3M)
* Computes:

  * Total PnL
  * Max Drawdown, Time to Recover
  * Median/Max/Min PnL
  * ROI, Sortino Ratio
  * Win %, Profit/Loss counts
* Outputs result into `existing_df`

### Key Functions:

* `minPnl()`
* `get_drawdown()`
* `analytics()`

> Requires `base_investment` defined per instrument (e.g. `350000` for NIFTY).

---

## Part 3: Final Strategy Ranking (Z-Score Based)

* Computes:

  * Normalized Z-scores of:

    * 11M PnL / Max Investment
    * Max Drawdown / Max Investment (inverted)
    * Win %
    * Sortino Ratio
* Weighted composite final Z-score:

```python
Final_Z_Score = Sortino_Z + 11M_PnL_Z + (1 - DD_Z) + 0.5 * Win_Z
```

* Sorts strategies by `Final_Z_Score`
* Picks top strategy with `Sortino > 1`

### Output

Final ranking CSV:

```bash
{stock}/{option_type}/Analytics/Analytics.csv
```

---

## How to Run

1. Ensure all trade CSVs are placed in `Trade_Sheets/`
2. Set correct `stock`, `option_type`, and `superset`
3. Run the script with:

```bash
python this_script.py
```

4. Daily PnL files will be created in `dailypnl/`
5. Final analytics will be saved in `Analytics/`

---

## Metrics Generated Per Strategy

| Metric                 | Description                               |
| ---------------------- | ----------------------------------------- |
| Total PnL              | Sum of net PnLs                           |
| Max Drawdown           | Largest peak-to-trough loss               |
| Sortino Ratio          | Monthly return adjusted for downside risk |
| ROI %                  | Scaled return over 40-month window        |
| Median PnL/Profit/Loss | Robustness measures                       |
| Win % / Loss %         | Win/Loss trade count percentages          |
| SD Profit / Loss       | Risk metrics                              |
| 11M PnL                | Avg monthly return over last \~12 months  |
| Final Z Score          | Weighted rank to identify best strategy   |

---

## Dependencies

```bash
pip install pandas numpy tqdm scipy
```

---

## Notes

* The filtering thresholds (Sortino > 1, positive PnL) are hardcoded but can be adjusted.
* The date ranges for 3M, 11M, 31M are also customizable.
* Consider adding multiprocessing to daily PnL generation if needed.

---

## Output Summary

```
Analytics.csv: Final ranked strategy metrics
*.csv in dailypnl/: Daily aggregated PnL files
```
