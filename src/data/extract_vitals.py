from google.cloud import bigquery
import pandas as pd
from pathlib import Path

PROJECT_ID = "ai-based-sepsis-prediction"
DATASET_ID = "sepsis"
COHORT_TABLE = "cohort"
VITALS_TABLE = "vitals_hourly"

client = bigquery.Client(project=PROJECT_ID)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

vitals_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{VITALS_TABLE}` AS
SELECT
    c.icustay_id,
    FLOOR(TIMESTAMP_DIFF(ce.charttime, c.intime, MINUTE) / 60) AS hour,
    CASE
        WHEN ce.itemid IN (211, 220045) THEN 'heart_rate'
        WHEN ce.itemid IN (456, 220052) THEN 'map'
        WHEN ce.itemid IN (618, 220210) THEN 'resp_rate'
        WHEN ce.itemid IN (223761, 678) THEN 'temperature'
        WHEN ce.itemid IN (646, 220277) THEN 'spo2'
    END AS vital,
    AVG(
        CASE
            -- Heart rate
            WHEN ce.itemid IN (211, 220045) THEN ce.valuenum

            -- MAP
            WHEN ce.itemid IN (456, 220052) THEN ce.valuenum

            -- Resp rate
            WHEN ce.itemid IN (618, 220210) THEN ce.valuenum

            -- Temperature (convert F → C)
            WHEN ce.itemid = 223761 THEN ce.valuenum
            WHEN ce.itemid = 678 THEN (ce.valuenum - 32) * 5.0 / 9.0

            -- SpO2
            WHEN ce.itemid IN (646, 220277) THEN ce.valuenum
        END
    ) AS value
FROM `physionet-data.mimiciii_clinical.chartevents` ce
JOIN `{PROJECT_ID}.{DATASET_ID}.{COHORT_TABLE}` c
    ON ce.icustay_id = c.icustay_id
WHERE
    ce.charttime BETWEEN c.intime AND TIMESTAMP_ADD(c.intime, INTERVAL 48 HOUR)
    AND (
        (ce.itemid IN (211, 220045) AND ce.valuenum BETWEEN 30 AND 220)
        OR (ce.itemid IN (456, 220052) AND ce.valuenum BETWEEN 40 AND 140)
        OR (ce.itemid IN (618, 220210) AND ce.valuenum BETWEEN 5 AND 60)
        OR (
            ce.itemid = 223761 AND ce.valuenum BETWEEN 34 AND 42
        )
        OR (
            ce.itemid = 678 AND ce.valuenum BETWEEN 93 AND 108
        )
        OR (ce.itemid IN (646, 220277) AND ce.valuenum BETWEEN 70 AND 100)
    )
GROUP BY c.icustay_id, hour, vital
HAVING hour BETWEEN 0 AND 47
"""



print("Running vitals aggregation query...")
client.query(vitals_sql).result()
print("Vitals table created")

export_sql = f"""
SELECT *
FROM `{PROJECT_ID}.{DATASET_ID}.{VITALS_TABLE}`
ORDER BY icustay_id, hour
"""

df = client.query(export_sql).to_dataframe()
df.to_csv(RAW_DIR / "vitals_hourly.csv", index=False)

print(f"Vitals exported to {RAW_DIR / 'vitals_hourly.csv'}")
