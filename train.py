"""
train.py — Model, feature engineering, and training loop.
THIS FILE IS EDITED AND ITERATED ON BY THE AGENT.

Current experiment: Ridge alpha=290 + cyclical month + momentum features
Metric to minimise: val_rmse (£, lower is better)
Secondary metrics:  val_mae, val_mape (for context only)

Agent may change:
  - Model type and hyperparameters
  - Additional feature engineering (add columns to X_train/X_val)
  - Preprocessing (scaling, transforms, encoding)
  - Ensembling

Agent must NOT change:
  - The import of get_splits, rmse, mae, mape from prepare
  - The final print format: "val_rmse=XXXX.XX"
  - The train/val split logic (lives in prepare.py)
"""

import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Load data (do not modify) ─────────────────────────────────────────────────
from prepare import load_data, get_splits, rmse, mae, mape

t0 = time.time()
df = load_data()
X_train, y_train, X_val, y_val, feature_cols = get_splits(df)

# ── Additional feature engineering ────────────────────────────────────────────
def add_features(X):
    month = X[:, 1]
    # Cyclical month encoding
    month_sin = np.sin(2 * np.pi * month / 12).reshape(-1, 1)
    month_cos = np.cos(2 * np.pi * month / 12).reshape(-1, 1)
    # Price momentum
    momentum_1_6 = (X[:, 2] - X[:, 4]).reshape(-1, 1)   # lag1 - lag6
    momentum_1_12 = (X[:, 2] - X[:, 5]).reshape(-1, 1)  # lag1 - lag12
    return np.hstack([X, month_sin, month_cos, momentum_1_6, momentum_1_12])

X_train = add_features(X_train)
X_val = add_features(X_val)

# Log-transform target (reduces skew on 35 years of price growth)
y_train_fit = np.log1p(y_train)
y_val_fit   = np.log1p(y_val)

# ── Model definition ──────────────────────────────────────────────────────────
# Try dropping subsets of features to find optimal set
# Feature indices: 0=year, 1=month, 2=lag1, 3=lag3, 4=lag6, 5=lag12,
# 6=roll3, 7=roll6, 8=roll12_std, 9=pct_1m, 10=pct_annual, 11=hpi,
# 12+=property types, then engineered: sin, cos, mom16, mom112

# Try dropping correlated/redundant features
# Feature selection: drop roll6, roll12_std, month_cos, momentum_1_12
n_feat = X_train.shape[1]
keep_cols = [i for i in range(n_feat) if i not in {7, 8, 20, 22}]
X_train = X_train[:, keep_cols]
X_val = X_val[:, keep_cols]

model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg",    Ridge(alpha=280.0)),
])

# ── Training ──────────────────────────────────────────────────────────────────
model.fit(X_train, y_train_fit)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred_log = model.predict(X_val)
y_pred     = np.expm1(y_pred_log)

val_rmse = rmse(y_val, y_pred)
val_mae  = mae(y_val, y_pred)
val_mape = mape(y_val, y_pred)

elapsed = time.time() - t0

# ── Results (agent must preserve this print format exactly) ──────────────────
print(f"val_rmse={val_rmse:.2f}")
print(f"val_mae={val_mae:.2f}")
print(f"val_mape={val_mape:.4f}")
print(f"elapsed={elapsed:.1f}s")
print(f"model={model['reg'].__class__.__name__}")
