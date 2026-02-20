import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
MODEL_DIR = PROJECT_ROOT / "data" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load data - USE SOFA_HOURLY.CSV which includes SOFA component scores
print("Loading data with SOFA features...")
features = pd.read_csv(LABEL_DIR / "sofa_hourly.csv")  # Changed from features.csv
sepsis_onset = pd.read_csv(LABEL_DIR / "sepsis_onset.csv")

print(f"Features shape: {features.shape}")
print(f"Columns with 'sofa' in name: {[c for c in features.columns if 'sofa' in c.lower()]}")

# Merge onset time
df = features.merge(sepsis_onset, on="icustay_id", how="left")

# Prediction horizon (hours before sepsis)
PREDICTION_HORIZON = 6

# Initialize label
df["label"] = 0

# Assign positive labels
mask = (
    (df["hour"] >= df["sepsis_onset_hour"] - PREDICTION_HORIZON) &
    (df["hour"] < df["sepsis_onset_hour"])
)

df.loc[mask, "label"] = 1

# Remove rows after sepsis onset (no leakage)
df = df[
    (df["sepsis_onset_hour"].isna()) |
    (df["hour"] < df["sepsis_onset_hour"])
]

# Drop helper column
df = df.drop(columns=["sepsis_onset_hour"])

# Save
output_path = MODEL_DIR / "train_dataset.csv"
df.to_csv(output_path, index=False)

print(f"✓ train_dataset.csv created at {output_path}")
print(f"✓ Shape: {df.shape}")
print(f"✓ SOFA features included: {[c for c in df.columns if 'sofa' in c.lower()]}")
print(f"✓ Label distribution: {df['label'].value_counts().to_dict()}")
