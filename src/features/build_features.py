import pandas as pd
import numpy as np
from pathlib import Path

# Paths

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# Load data

vitals = pd.read_csv(RAW_DIR / "vitals_hourly.csv")
labs = pd.read_csv(RAW_DIR / "labs_hourly.csv")
cohort = pd.read_csv(RAW_DIR / "cohort.csv")

# Pivot vitals & labs to wide format

vitals_wide = vitals.pivot_table(
    index=["icustay_id", "hour"],
    columns="vital",
    values="value"
).reset_index()

labs_wide = labs.pivot_table(
    index=["icustay_id", "hour"],
    columns="lab",
    values="value"
).reset_index()

# Merge vitals + labs

df = pd.merge(vitals_wide, labs_wide, on=["icustay_id", "hour"], how="outer")

# Merge static cohort features

df = pd.merge(
    df,
    cohort[["icustay_id", "age", "gender"]],
    on="icustay_id",
    how="left"
)

# Sort properly

df = df.sort_values(["icustay_id", "hour"]).reset_index(drop=True)

# Feature engineering functions

def add_rolling_features(df, col, window=6):
    df[f"{col}_mean_{window}h"] = (
        df.groupby("icustay_id")[col]
        .rolling(window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df[f"{col}_std_{window}h"] = (
        df.groupby("icustay_id")[col]
        .rolling(window, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
    )

    df[f"{col}_min_{window}h"] = (
        df.groupby("icustay_id")[col]
        .rolling(window, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    df[f"{col}_max_{window}h"] = (
        df.groupby("icustay_id")[col]
        .rolling(window, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df[f"{col}_delta_{window}h"] = (
        df[col] - df.groupby("icustay_id")[col].shift(window)
    )

    df[f"{col}_missing_{window}h"] = (
        df.groupby("icustay_id")[col]
        .rolling(window, min_periods=1)
        .apply(lambda x: x.isna().all())
        .reset_index(level=0, drop=True)
    )

    return df


# Apply feature engineering

FEATURE_COLUMNS = [
    "heart_rate", "map", "resp_rate", "temperature", "spo2",
    "creatinine", "bilirubin", "platelets", "lactate"
]

for col in FEATURE_COLUMNS:
    if col in df.columns:
        df = add_rolling_features(df, col, window=6)


# Encode gender

df["gender"] = df["gender"].map({"M": 1, "F": 0})


# Save features

output_path = FEATURE_DIR / "features.csv"
df.to_csv(output_path, index=False)

print(f"Phase 4 features saved to {output_path}")