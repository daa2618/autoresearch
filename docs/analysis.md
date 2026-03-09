# UK House Price Index Prediction - Experiment Analysis

## Problem Statement

Predict next-month UK average house prices using historical Land Registry data (1990-2026). The dataset contains ~400 monthly observations with lagged prices, rolling statistics, percentage changes, HPI index values, and property-type breakdowns. The train/val split is strictly chronological (train: ~1990-2022, val: ~2022-2024) to prevent data leakage.

**Primary metric**: `val_rmse` (RMSE in GBP, lower is better)

---

## Algorithms Evaluated

### 1. Ridge Regression (sklearn.linear_model.Ridge)

L2-regularised linear regression. The final and best-performing model. Ridge excels here because:

- **Linear extrapolation**: House prices in the validation period (~250k-290k) exceed the training range. Linear models naturally extrapolate trends; tree models cannot predict beyond the training target range.
- **Strong regularisation needed**: With ~300 training samples and 19 features, overfitting is a real risk. Tuning alpha from the default 1.0 to 280 was the single largest improvement.
- **Log-space modelling**: Fitting on `log1p(y)` then inverting with `expm1()` handles the exponential price growth from ~50k (1990) to ~290k (2024).

### 2. HistGradientBoostingRegressor (sklearn.ensemble)

Histogram-based gradient boosting, a fast tree ensemble. Tested both with and without log-target transform.

- **Result**: ~31,900 RMSE (3x worse than baseline)
- **Why it failed**: Tree-based models partition the feature space and predict the mean target within each leaf. They cannot predict values outside the range seen during training. Since validation prices exceed training prices, trees systematically underpredict.

### 3. ElasticNet (sklearn.linear_model.ElasticNet)

Combined L1 + L2 regularisation. Grid-searched over alpha and l1_ratio.

- **Result**: Best configuration was l1_ratio=0.0 (pure Ridge), confirming L1 sparsity adds no value here.
- **Interpretation**: All features contribute meaningful signal; none should be driven to exactly zero.

### 4. Ridge + HGBR Residual Stacking

Two-stage approach: Ridge predicts the trend, then HistGradientBoostingRegressor corrects the residuals.

- **Result**: 3,786 RMSE (worse than Ridge alone)
- **Why it failed**: The HGBR overfits to training residuals given the small dataset (~300 samples). The residual corrections don't generalise.

### 5. Ridge Ensemble (averaging over alphas)

Average predictions from Ridge models at alpha values [100, 200, 290, 400, 600, 1000].

- **Result**: 4,009 RMSE
- **Why it failed**: Models with small alpha (under-regularised) overfit and drag down the ensemble average.

### 6. Polynomial Features (sklearn.preprocessing.PolynomialFeatures)

Degree-2 polynomial expansion (both full and interaction-only variants), with alpha re-tuned.

- **Result**: ~4,273 RMSE at best
- **Why it failed**: Polynomial expansion on 19+ features creates hundreds of terms. Even with heavy regularisation, this introduces more noise than signal on a small dataset.

### 7. Box-Cox Power Transform on Target

Used `PowerTransformer(method='box-cox')` instead of `log1p` for the target variable.

- **Result**: 136,360 RMSE (catastrophic)
- **Why it failed**: The fitted Box-Cox parameters optimise normality on the training distribution. When the model predicts values slightly outside the fitted range, the inverse transform amplifies errors severely.

---

## Feature Engineering

### Features That Helped

| Feature | Description | Impact |
|---------|-------------|--------|
| `month_sin` | `sin(2*pi*month/12)` | Captures seasonal cycle; small but consistent improvement |
| `momentum_1_6` | `price_lag_1 - price_lag_6` | Short-term price trend signal |

### Features That Were Dropped (improved RMSE when removed)

| Feature | Why Dropped |
|---------|-------------|
| `price_roll6_mean` | Redundant with `price_roll3_mean` and lag features |
| `price_roll12_std` | Volatility signal adds noise on small validation set |
| `month_cos` | Redundant given `month_sin` + linear `month` already present |
| `momentum_1_12` | Long-term momentum too noisy; `momentum_1_6` suffices |

### Features That Didn't Help

| Feature | Reason |
|---------|--------|
| `price_lag_1 * pct_change_annual` | Interaction adds noise, not captured signal |
| `hpi * price_lag_1` | Highly collinear with constituent features |
| `price_lag_1 / price_roll3_mean` | Ratio features redundant with existing differences |
| `price_lag_3 - price_lag_12` | Acceleration metric; too noisy |

### Other Techniques That Didn't Help

| Technique | Result |
|-----------|--------|
| Sample weighting (linear) | 3,122 RMSE - distorts fit |
| Sample weighting (exponential) | 7,645 RMSE - severe distortion |

---

## Best Model Configuration

```
Pipeline:
  1. StandardScaler (zero-mean, unit-variance)
  2. Ridge(alpha=280.0)

Target transform: log1p(y) / expm1(pred)

Features used (19 of 23):
  - year, month
  - price_lag_1, price_lag_3, price_lag_6, price_lag_12
  - price_roll3_mean
  - pct_change_1m, pct_change_annual
  - hpi
  - average_price_detached, average_price_semi_detached,
    average_price_terraced, average_price_flat_maisonette
  - sales_volume (and other property-type columns if present)
  - month_sin (engineered)
  - momentum_1_6 (engineered: lag1 - lag6)

Features dropped:
  - price_roll6_mean, price_roll12_std, month_cos, momentum_1_12
```

---

## Experiment Log

| # | Change | val_rmse | val_mae | val_mape | Kept? |
|---|--------|----------|---------|----------|-------|
| 0 | Baseline: Ridge alpha=1.0 + log(y) | 10,255.90 | 9,006.40 | 3.56% | -- |
| 1 | HistGradientBoostingRegressor + log(y) | 31,948.37 | 28,506.41 | 11.29% | No |
| 1b | HistGradientBoostingRegressor, no log(y) | 31,882.69 | 28,440.73 | 11.26% | No |
| 2 | Ridge + cyclical month + interactions + momentum | 11,187.45 | 9,812.86 | 3.88% | No |
| 3 | Ridge alpha=270 (tuned), log(y) | 2,965.21 | 2,572.72 | 1.06% | Yes |
| 4 | + cyclical month_sin/cos + momentum, alpha=290 | 2,570.81 | 2,270.18 | 0.93% | Yes |
| 5 | Ridge ensemble (avg over 6 alphas) | 4,009.36 | 3,550.22 | 1.48% | No |
| 6 | + ratio and acceleration features | 2,732.01 | 2,353.98 | 0.97% | No |
| 7 | ElasticNet grid search | 2,568.12 | -- | -- | No |
| 8 | Box-Cox power transform on target | 136,359.96 | -- | -- | No |
| 9 | Polynomial features degree 2 + tuned alpha | 4,273.31 | -- | -- | No |
| 10 | Ridge + HGBR residual stacking | 3,785.59 | 2,949.37 | 1.15% | No |
| 11 | Sample weighting (linear) | 3,122.20 | 2,675.33 | 1.12% | No |
| 11b | Sample weighting (exponential) | 7,645.27 | 7,494.62 | 3.06% | No |
| 12 | Feature selection + alpha=280 | **2,322.28** | **2,089.08** | **0.86%** | Yes |

---

## Key Takeaways

1. **Regularisation dominates**: Tuning Ridge alpha from 1.0 to 280 cut RMSE by 71% (10,256 -> 2,965). On small tabular datasets, the default sklearn hyperparameters are almost never optimal.

2. **Linear models win for extrapolation**: Tree-based methods (HGBR) are the default choice for tabular ML, but they fundamentally cannot extrapolate. In time-series with trending targets, linear models have a structural advantage.

3. **Log-target transform is essential**: Prices span 50k-290k over 35 years. Fitting in log-space lets Ridge model proportional relationships (percentage growth) rather than absolute differences.

4. **Feature engineering has diminishing returns**: After alpha tuning, feature additions provided ~10% improvement. Removing noisy/redundant features was more effective than adding new ones.

5. **Simplicity wins on small data**: With ~300 training samples, every added parameter is a liability. The best model is a single Ridge regressor with 19 features -- no ensembles, no polynomials, no stacking.

---

## Final Performance

| Metric | Value |
|--------|-------|
| val_rmse | 2,322.28 |
| val_mae | 2,089.08 |
| val_mape | 0.86% |
| Training time | <0.1s |
| Improvement over baseline | **77.4%** |

Average UK house price in the validation period is approximately 250,000-290,000. A MAPE of 0.86% means the model's average prediction error is roughly 2,200 -- within typical monthly price fluctuation.

---

## Using the Model for Predictions on Unseen Data

### Overview

The trained model predicts next month's UK average house price given a row of historical features. To use it on new data you need to:

1. Prepare the raw CSV so it contains the same columns as the training data
2. Build the base features (lags, rolling stats, percentage changes)
3. Apply the same engineered features and feature selection used during training
4. Feed the processed row(s) through the fitted pipeline
5. Inverse-transform the prediction from log-space back to GBP

### Step-by-step Example

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from prepare import load_data, get_splits

# ── 1. Train the model on all available labelled data ────────────────────────
df = load_data()
X_train, y_train, X_val, y_val, feature_cols = get_splits(df)

def add_features(X):
    """Engineered features added during training."""
    month = X[:, 1]
    month_sin = np.sin(2 * np.pi * month / 12).reshape(-1, 1)
    month_cos = np.cos(2 * np.pi * month / 12).reshape(-1, 1)
    momentum_1_6 = (X[:, 2] - X[:, 4]).reshape(-1, 1)
    momentum_1_12 = (X[:, 2] - X[:, 5]).reshape(-1, 1)
    return np.hstack([X, month_sin, month_cos, momentum_1_6, momentum_1_12])

X_train = add_features(X_train)
X_val   = add_features(X_val)

# Combine train + val for final model (use all labelled data for production)
X_all = np.vstack([X_train, X_val])
y_all = np.concatenate([y_train, y_val])

# Feature selection: drop roll6 (7), roll12_std (8), month_cos (20), momentum_1_12 (22)
n_feat = X_all.shape[1]
keep_cols = [i for i in range(n_feat) if i not in {7, 8, 20, 22}]
X_all = X_all[:, keep_cols]

y_all_log = np.log1p(y_all)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg",    Ridge(alpha=280.0)),
])
model.fit(X_all, y_all_log)


# ── 2. Prepare a new data point for prediction ──────────────────────────────
# You need the same feature columns in the same order as get_splits() returns.
# The feature_cols list from get_splits() tells you exactly what is expected:
#
#   ['year', 'month',
#    'price_lag_1', 'price_lag_3', 'price_lag_6', 'price_lag_12',
#    'price_roll3_mean', 'price_roll6_mean', 'price_roll12_std',
#    'pct_change_1m', 'pct_change_annual', 'hpi',
#    'average_price_detached', 'average_price_semi_detached',
#    'average_price_terraced', 'average_price_flat_maisonette',
#    ... any other property-type columns present in the CSV]
#
# Example: predicting the price for the month following the latest data point.
# Suppose the most recent known average_price is from February 2026.

new_row = np.array([[
    2026,           # year
    2,              # month (of the observation, i.e. the month whose lags we know)
    285000,         # price_lag_1  (last month's price)
    282000,         # price_lag_3  (price 3 months ago)
    278000,         # price_lag_6  (price 6 months ago)
    270000,         # price_lag_12 (price 12 months ago)
    283000,         # price_roll3_mean
    280000,         # price_roll6_mean  (will be dropped, but must be present)
    4500,           # price_roll12_std  (will be dropped, but must be present)
    0.5,            # pct_change_1m
    3.2,            # pct_change_annual
    155.0,          # hpi
    420000,         # average_price_detached
    290000,         # average_price_semi_detached
    250000,         # average_price_terraced
    220000,         # average_price_flat_maisonette
]])

# Apply the same feature engineering
new_row = add_features(new_row)

# Apply the same feature selection
new_row = new_row[:, keep_cols]

# ── 3. Predict ───────────────────────────────────────────────────────────────
predicted_log = model.predict(new_row)
predicted_price = np.expm1(predicted_log)

print(f"Predicted next-month average price: £{predicted_price[0]:,.0f}")
```

### Building Features from a Raw CSV Update

If you receive an updated `uk_hpi_*.csv` file rather than constructing feature values manually, you can use `prepare.py` to handle feature construction automatically:

```python
import numpy as np
from prepare import load_data, build_base_features

# Load and build features from the updated CSV
df = load_data()
df = build_base_features(df)
df = df.dropna().reset_index(drop=True)

# The feature columns (same order as get_splits)
feature_cols = [
    "year", "month",
    "price_lag_1", "price_lag_3", "price_lag_6", "price_lag_12",
    "price_roll3_mean", "price_roll6_mean", "price_roll12_std",
    "pct_change_1m", "pct_change_annual", "hpi",
]
# Add property-type columns if present
extra_cols = [c for c in df.columns if c.startswith("average_price_")
              and c != "average_price" and df[c].notna().sum() > 100]
feature_cols += extra_cols

# Take the last row (most recent month) as the prediction input
X_new = df[feature_cols].iloc[[-1]].values

# Apply engineered features, selection, then predict
X_new = add_features(X_new)
X_new = X_new[:, keep_cols]

predicted_price = np.expm1(model.predict(X_new))
print(f"Predicted next-month average price: £{predicted_price[0]:,.0f}")
```

### Important Notes

- **Feature order matters**: The numpy arrays are positional. The columns must appear in the exact same order as `get_splits()` returns them. The `feature_cols` list from that function is the reference.
- **Dropped features must still be present before selection**: The `add_features()` function references columns by index (e.g., index 4 = `price_lag_6`). You must provide all 16+ base features so the indices align, even though 4 are dropped after engineering.
- **Lag features require history**: To predict the price for month T+1, you need the actual `average_price` for months T, T-2, T-5, and T-11 (for lags 1, 3, 6, 12). You also need at least 3 months of history for `price_roll3_mean`. If any of these are missing, the prediction will be unreliable.
- **Retraining**: As new monthly data becomes available, retrain the model on all available data (train + val + new months) to capture the latest trends. The alpha value (280) and feature set should remain stable, but can be re-validated periodically.
- **Model persistence**: For production use, save the fitted model with `joblib` to avoid retraining on every run:

```python
import joblib

# Save after training
joblib.dump(model, "model.joblib")
joblib.dump(keep_cols, "keep_cols.joblib")

# Load later
model = joblib.load("model.joblib")
keep_cols = joblib.load("keep_cols.joblib")
```
