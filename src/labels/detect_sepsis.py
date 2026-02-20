import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "data" / "labels"

df = pd.read_csv(LABEL_DIR / "sofa_hourly.csv")

# Compute baseline SOFA per ICU stay
baseline = (
    df[df["hour"] <= 6]
    .groupby("icustay_id")["sofa_total"]
    .median()
    .reset_index()
    .rename(columns={"sofa_total": "baseline_sofa"})
)

df = df.merge(baseline, on="icustay_id", how="left")

# Delta SOFA
df["delta_sofa"] = df["sofa_total"] - df["baseline_sofa"]

# Sepsis onset
sepsis_onset = (
    df[df["delta_sofa"] >= 2]
    .groupby("icustay_id")["hour"]
    .min()
    .reset_index()
    .rename(columns={"hour": "sepsis_onset_hour"})
)

sepsis_onset.to_csv(LABEL_DIR / "sepsis_onset.csv", index=False)

print("Sepsis onset detected")
