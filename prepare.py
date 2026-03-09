"""
prepare.py — Fixed constants, data loading, feature engineering, and evaluation.
The agent NEVER edits this file.

Dataset: UK House Price Index 1990–2026
Target:  average_price (next month's average UK house price)
Task:    Time-series regression — predict next month's price from history
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH   = Path("uk_hpi_1990_2026.csv")
TARGET_COL  = "average_price"
DATE_COL    = "ref_period_start"
VAL_SPLIT   = 0.15   # last 15% of timeline = held-out validation
TEST_SPLIT  = 0.10   # last 10% after val = test (agent never sees this)
RANDOM_SEED = 42

# ── Data loading & cleaning ───────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Parse date and sort chronologically
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL]).reset_index(drop=True)

    # Convert object % columns to float
    pct_cols = [c for c in df.columns if "percentage" in c]
    for col in pct_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop URL-style identifier columns
    df = df.drop(columns=["_about", "ref_region", "ref_month",
                           "ref_period_duration"], errors="ignore")

    return df


# ── Core feature set (prepare only, agent may ADD features in train.py) ───────
def build_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal, always-present features derived from raw columns.
    Agent can build additional features on top of these in train.py.
    """
    df = df.copy()

    # Date decomposition
    df["year"]  = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month

    # Lagged prices (1, 3, 6, 12 months) — core time-series features
    for lag in [1, 3, 6, 12]:
        df[f"price_lag_{lag}"] = df[TARGET_COL].shift(lag)

    # Rolling statistics
    df["price_roll3_mean"]  = df[TARGET_COL].shift(1).rolling(3).mean()
    df["price_roll6_mean"]  = df[TARGET_COL].shift(1).rolling(6).mean()
    df["price_roll12_std"]  = df[TARGET_COL].shift(1).rolling(12).std()

    # Month-over-month change (from raw data, already cleaned)
    df["pct_change_1m"] = pd.to_numeric(df["percentage_change"], errors="coerce")
    df["pct_change_annual"] = pd.to_numeric(df["percentage_annual_change"], errors="coerce")

    # HPI as a feature
    df["hpi"] = df["house_price_index"]

    return df


# ── Train / val split (chronological — no leakage) ───────────────────────────
def get_splits(df: pd.DataFrame):
    """
    Returns (X_train, y_train, X_val, y_val, feature_cols)
    Split is strictly chronological — no shuffling.
    """
    df = build_base_features(df)
    df = df.dropna().reset_index(drop=True)

    n = len(df)
    test_idx = int(n * (1 - TEST_SPLIT))
    val_idx  = int(n * (1 - TEST_SPLIT - VAL_SPLIT))

    feature_cols = [
        "year", "month",
        "price_lag_1", "price_lag_3", "price_lag_6", "price_lag_12",
        "price_roll3_mean", "price_roll6_mean", "price_roll12_std",
        "pct_change_1m", "pct_change_annual", "hpi",
    ]
    # Also include property-type prices if available
    extra_cols = [c for c in df.columns if c.startswith("average_price_")
                  and c != TARGET_COL and df[c].notna().sum() > 100]
    feature_cols += extra_cols

    # Keep only columns that exist after dropna
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values
    y = df[TARGET_COL].values

    X_train, y_train = X[:val_idx],   y[:val_idx]
    X_val,   y_val   = X[val_idx:test_idx], y[val_idx:test_idx]

    return X_train, y_train, X_val, y_val, feature_cols


# ── Evaluation metric ─────────────────────────────────────────────────────────
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))

def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    X_train, y_train, X_val, y_val, feat_cols = get_splits(df)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")
    print(f"Features ({len(feat_cols)}): {feat_cols}")
    print(f"Price range: £{y_train.min():,.0f} – £{y_train.max():,.0f}")
    print(f"Val price range: £{y_val.min():,.0f} – £{y_val.max():,.0f}")
    print("prepare.py OK ✓")
