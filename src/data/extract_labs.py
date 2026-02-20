from google.cloud import bigquery
from pathlib import Path

# Configuration

PROJECT_ID = "ai-based-sepsis-prediction"
DATASET_ID = "sepsis"
COHORT_TABLE = "cohort"
LABS_TABLE = "labs_hourly"

client = bigquery.Client(project=PROJECT_ID)

# Resolve project root (optional but consistent)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Phase 3 SQL: Extract labs

labs_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{LABS_TABLE}` AS
SELECT
    c.icustay_id,
    FLOOR(TIMESTAMP_DIFF(le.charttime, c.intime, MINUTE) / 60) AS hour,
    CASE
        WHEN le.itemid = 50912 THEN 'creatinine'
        WHEN le.itemid = 50885 THEN 'bilirubin'
        WHEN le.itemid = 51265 THEN 'platelets'
        WHEN le.itemid = 50813 THEN 'lactate'
    END AS lab,
    AVG(le.valuenum) AS value
FROM `physionet-data.mimiciii_clinical.labevents` le
JOIN `{PROJECT_ID}.{DATASET_ID}.{COHORT_TABLE}` c
    ON le.hadm_id = c.hadm_id
WHERE
    le.itemid IN (50912, 50885, 51265, 50813)
    AND le.valuenum IS NOT NULL
    AND le.charttime BETWEEN c.intime
        AND TIMESTAMP_ADD(c.intime, INTERVAL 48 HOUR)
GROUP BY c.icustay_id, hour, lab
HAVING hour BETWEEN 0 AND 47
"""

# Run query

print("Running Phase 3: Labs extraction...")
client.query(labs_sql).result()
print("Labs table created successfully")

# Export labs to CSV

export_sql = f"""
SELECT *
FROM `{PROJECT_ID}.{DATASET_ID}.{LABS_TABLE}`
ORDER BY icustay_id, hour
"""

df = client.query(export_sql).to_dataframe()

csv_path = RAW_DIR / "labs_hourly.csv"
df.to_csv(csv_path, index=False)

print(f"Labs exported to {csv_path}")
