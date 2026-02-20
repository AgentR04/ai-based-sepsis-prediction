from google.cloud import bigquery
import pandas as pd
from pathlib import Path

PROJECT_ID = "ai-based-sepsis-prediction"
DATASET_ID = "sepsis"
TABLE_ID = "cohort"

client = bigquery.Client(project=PROJECT_ID)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

query = f"""
SELECT *
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
"""

df = client.query(query).to_dataframe()
df.to_csv(RAW_DIR / "cohort.csv", index=False)

print(f"Cohort exported to {RAW_DIR / 'cohort.csv'}")
