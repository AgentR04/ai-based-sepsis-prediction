from google.cloud import bigquery

PROJECT_ID = "ai-based-sepsis-prediction"   # ← replace this
DATASET_ID = "sepsis"
TABLE_ID = "cohort"

client = bigquery.Client(project=PROJECT_ID)
#only create dataset once
'''dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{DATASET_ID}")
dataset_ref.location = "US"

try:
    client.create_dataset(dataset_ref)
    print("Dataset created")
except Exception:
    print("Dataset already exists")
'''
cohort_sql = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}` AS
SELECT
    p.subject_id,
    a.hadm_id,
    i.icustay_id,
    p.gender,
    DATETIME_DIFF(a.admittime, p.dob, YEAR) AS age,
    a.admission_type,
    i.intime,
    i.outtime,
    DATETIME_DIFF(i.outtime, i.intime, HOUR) AS icu_los_hours
FROM `physionet-data.mimiciii_clinical.patients` p
JOIN `physionet-data.mimiciii_clinical.admissions` a
    ON p.subject_id = a.subject_id
JOIN `physionet-data.mimiciii_clinical.icustays` i
    ON a.hadm_id = i.hadm_id
WHERE
    DATETIME_DIFF(a.admittime, p.dob, YEAR) >= 18
    AND DATETIME_DIFF(i.outtime, i.intime, HOUR) >= 24
QUALIFY
    ROW_NUMBER() OVER (PARTITION BY p.subject_id ORDER BY i.intime) = 1
LIMIT 10000
"""

print("Running cohort query...")
job = client.query(cohort_sql)
job.result()
print("Cohort table created successfully")
verify_sql = f"""
SELECT COUNT(*) AS n
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
"""
df = client.query(verify_sql).to_dataframe()
print(df)
