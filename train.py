"""
train.py — Model, feature engineering, and training loop.
THIS FILE IS EDITED AND ITERATED ON BY THE AGENT.

Current experiment: Baseline — Ridge Regression with standard scaling
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

# ── Optional: additional feature engineering ──────────────────────────────────
# Agent: you may add derived features here by transforming X_train and X_val.
# Example: log-transform the target, add polynomial features, etc.
# Keep feature engineering symmetric — apply same transforms to train AND val.

# Log-transform target (reduces skew on 35 years of price growth)
y_train_fit = np.log1p(y_train)
y_val_fit   = np.log1p(y_val)

# ── Model definition ──────────────────────────────────────────────────────────
# Agent: replace or modify this. Options include:
#   Ridge, Lasso, ElasticNet, GradientBoostingRegressor, RandomForestRegressor,
#   XGBRegressor (if installed), MLPRegressor, SVR, etc.
# Always wrap in a Pipeline with a scaler unless your model handles scale natively.

model = Pipeline([
    ("scaler", StandardScaler()),
    ("reg",    Ridge(alpha=1.0)),
])

# ── Training ──────────────────────────────────────────────────────────────────
model.fit(X_train, y_train_fit)

# ── Evaluation ────────────────────────────────────────────────────────────────
# Inverse-transform predictions back to £ space
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
