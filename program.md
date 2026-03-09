# autoresearch — UK House Price Index Prediction
## Research Program v1.0

---

## Your role

You are an autonomous ML research agent. Your job is to improve the prediction
of UK average house prices (next-month forecast) by modifying `train.py`.

You run experiments in a loop:
1. Read this file and the current `train.py`
2. Propose ONE targeted change
3. Edit `train.py`
4. Run `python train.py` and record `val_rmse`
5. If `val_rmse` improved → keep the change, log it
6. If `val_rmse` did not improve → revert `train.py`, log the failure
7. Repeat

---

## Dataset context

- **Source**: UK HPI (Land Registry), 1990–2026
- **Granularity**: Monthly, national UK aggregate
- **Target**: `average_price` — raw £ average house price (next month)
- **Train period**: 1990–2022 approx
- **Val period**: 2022–2024 approx (chronological, no leakage)
- **Test set**: Held out entirely in `prepare.py` — you never touch it

### Key columns available as features
| Column | Description |
|--------|-------------|
| `price_lag_1/3/6/12` | Lagged average price (1, 3, 6, 12 months back) |
| `price_roll3_mean` | 3-month rolling average of price |
| `price_roll6_mean` | 6-month rolling average of price |
| `price_roll12_std` | 12-month rolling std of price |
| `pct_change_1m` | Month-on-month % change |
| `pct_change_annual` | Year-on-year % change |
| `hpi` | House Price Index (index-form, base 100 = Jan 2015) |
| `average_price_detached` | Detached house price (partial coverage) |
| `average_price_semi_detached` | Semi-detached (partial coverage) |
| `average_price_terraced` | Terraced (partial coverage) |
| `average_price_flat_maisonette` | Flat/maisonette (partial coverage) |
| `sales_volume` | Monthly transaction count |
| `year`, `month` | Calendar components |

---

## Metric

**Primary**: `val_rmse` — RMSE in £. Lower is better.  
**Secondary**: `val_mae`, `val_mape` — reported for context, not used to decide keep/revert.

Current baseline: Ridge Regression + StandardScaler + log(y)
Record your baseline `val_rmse` on first run before making any changes.

---

## Rules

### You MAY change in `train.py`
- Model class and all its hyperparameters
- Additional feature engineering applied to `X_train` and `X_val` **symmetrically**
- Target transformation (currently `log1p`) — try others or remove
- Preprocessing (scalers, power transforms, polynomial features)
- Ensembling (average multiple models)
- Import any library already in the environment (sklearn, numpy, scipy)

### You MUST NOT change
- `from prepare import load_data, get_splits, rmse, mae, mape`
- The train/val split (it lives in `prepare.py`)
- The final print lines: `val_rmse=`, `val_mae=`, `val_mape=`, `elapsed=`
- `prepare.py` — never touch it

### External libraries allowed
`sklearn`, `numpy`, `scipy`, `pandas`  
If you want `xgboost` or `lightgbm`, try to import and gracefully skip if not installed.

---

## Experiment log

Record every experiment here as you go. Format:

```
| # | Change | val_rmse | val_mae | kept? |
|---|--------|----------|---------|-------|
| 0 | Baseline: Ridge alpha=1.0 + log(y) | TBD | TBD | — |
```

---

## Research directions to explore (in rough priority order)

1. **Gradient boosted trees** — GradientBoostingRegressor or HistGradientBoostingRegressor
   - These handle tabular non-linear patterns well
   - Try with and without log(y) transform

2. **Richer lag features** — add lag 24, 36 months; rolling 24-month mean/std

3. **Cyclical month encoding** — encode `month` as sin/cos to capture seasonality

4. **Interaction features** — `price_lag_1 × pct_change_annual`, `hpi × sales_volume`

5. **Power transform on target** — try `QuantileTransformer` or `PowerTransformer(method='box-cox')`

6. **Stacking ensemble** — Ridge + GBT predictions as meta-features into a final linear model

7. **Feature selection** — try dropping low-variance or high-collinearity features

---

## What success looks like

| val_rmse | Interpretation |
|----------|----------------|
| > £15,000 | Worse than baseline |
| £10,000–£15,000 | Marginal — baseline territory |
| £5,000–£10,000 | Good — meaningful improvement |
| < £5,000 | Excellent — strong predictive model |
| < £2,000 | Near-perfect — check for data leakage |

Average UK house price in val period is approximately £250,000–£290,000.
A MAPE under 3% would be a strong result.

---

## How to run

```bash
# Single experiment
python train.py

# Check output format
python train.py | grep val_rmse
```

---

*program.md last updated: v1.0 — start here, iterate this file as you learn what works.*
