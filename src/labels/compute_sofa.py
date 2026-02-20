import pandas as pd
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "data" / "labels"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load features
df = pd.read_csv(FEATURE_DIR / "features.csv")

# --- SOFA sub-scores ---

def sofa_platelets(x):
    if pd.isna(x): return 0
    if x < 20: return 4
    if x < 50: return 3
    if x < 100: return 2
    if x < 150: return 1
    return 0

def sofa_bilirubin(x):
    if pd.isna(x): return 0
    if x >= 12.0: return 4
    if x >= 6.0: return 3
    if x >= 2.0: return 2
    if x >= 1.2: return 1
    return 0

def sofa_map(x):
    if pd.isna(x): return 0
    return 0 if x >= 70 else 1

def sofa_creatinine(x):
    if pd.isna(x): return 0
    if x >= 5.0: return 4
    if x >= 3.5: return 3
    if x >= 2.0: return 2
    if x >= 1.2: return 1
    return 0

# Apply scores
df["sofa_platelets"] = df["platelets"].apply(sofa_platelets)
df["sofa_bilirubin"] = df["bilirubin"].apply(sofa_bilirubin)
df["sofa_map"] = df["map"].apply(sofa_map)
df["sofa_creatinine"] = df["creatinine"].apply(sofa_creatinine)

# Total SOFA (partial)
df["sofa_total"] = (
    df["sofa_platelets"] +
    df["sofa_bilirubin"] +
    df["sofa_map"] +
    df["sofa_creatinine"]
)

# Save
output_path = OUTPUT_DIR / "sofa_hourly.csv"
df.to_csv(output_path, index=False)

print(f"SOFA scores saved to {output_path}")
