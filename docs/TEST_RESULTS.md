# Test Case Results

**Project:** ICU Sepsis Prediction (MIMIC-III)  
**Run date:** February 21, 2026  
**Suite:** `pytest tests/` — 4 collected, **4 passed** in 9.45s  
**Python:** `.venv` (3.13) · **pytest:** 9.0.2  
**Framework:** pytest with scikit-learn, XGBoost, PyTorch Geometric

---

## Summary

| Test Case ID | Test Case Name | File | Function | Type | Result | Duration |
|---|---|---|---|---|---|---|
| TC-01 | SOFA Score Validation | `tests/test_sofa.py` | `test_sofa_validation` | Unit + Data | ✅ PASSED | 0.27s |
| TC-02 | ML Ensemble Model Validation | `tests/test_models.py` | `test_ml_model_validation` | Integration | ✅ PASSED | 6.70s |
| TC-03 | GNN Model Validation | `tests/test_gnn_alerts.py` | `test_gnn_model_validation` | Integration | ✅ PASSED | 23.24s |
| TC-04 | End-to-End Pipeline Validation | `tests/test_e2e.py` | `test_end_to_end_pipeline` | System/E2E | ✅ PASSED | 7.98s |

---

## TC-01 — SOFA Score Validation

| Field | Details |
|-------|---------|
| **Test Case ID** | TC-01 |
| **Test Case Name** | SOFA Score Validation |
| **File** | `tests/test_sofa.py` |
| **Function** | `test_sofa_validation()` |
| **Type** | Unit + Data Integrity |
| **Priority** | High |
| **Description** | Validates all 4 SOFA component scoring functions against clinical boundary values, checks monotonicity, and verifies integrity of the `sofa_hourly.csv` dataset |

### Parameters / Inputs

| Parameter | Value |
|-----------|-------|
| SOFA functions under test | `sofa_platelets`, `sofa_bilirubin`, `sofa_map`, `sofa_creatinine` |
| Platelets test values | 200, 150, 149, 99, 49, 19, NaN |
| Bilirubin test values | 1.0, 1.2, 2.0, 6.0, 12.0, NaN |
| MAP test values | 70, 100, 69, 50, NaN |
| Creatinine test values | 1.0, 1.2, 2.0, 3.5, 5.0, NaN |
| Monotonicity sequences | Platelets ↓ [200, 140, 90, 40, 10], Bilirubin ↑ [0.5, 1.5, 3.0, 8.0, 15.0], Creatinine ↑ [0.8, 1.5, 2.5, 4.0, 6.0] |
| Dataset | `data/labels/sofa_hourly.csv`, `data/labels/sepsis_onset.csv` |

### Prerequisites

- `data/labels/sofa_hourly.csv` exists and is non-empty
- `data/labels/sepsis_onset.csv` exists and is non-empty
- Python packages: `pandas`, `numpy`

### Test Steps & Results

| Step | Check | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 1 | `sofa_platelets` boundary values | 0, 0, 1, 2, 3, 4, 0 | 0, 0, 1, 2, 3, 4, 0 | ✅ |
| 2 | `sofa_bilirubin` boundary values | 0, 1, 2, 3, 4, 0 | 0, 1, 2, 3, 4, 0 | ✅ |
| 3 | `sofa_map` boundary values | 0, 0, 1, 1, 0 | 0, 0, 1, 1, 0 | ✅ |
| 4 | `sofa_creatinine` boundary values | 0, 1, 2, 3, 4, 0 | 0, 1, 2, 3, 4, 0 | ✅ |
| 5 | NaN input → score 0 (all 4 functions) | 0 | 0 | ✅ |
| 6 | Platelets monotonicity (decreasing count → non-decreasing score) | `[0,1,2,3,4]` sorted | Monotone | ✅ |
| 7 | Bilirubin monotonicity (increasing value → non-decreasing score) | `[0,1,2,3,4]` sorted | Monotone | ✅ |
| 8 | Creatinine monotonicity (increasing value → non-decreasing score) | `[0,1,2,3,4]` sorted | Monotone | ✅ |
| 9 | 4-component SOFA max (worst clinical values) | 13 | 13 | ✅ |
| 10 | 4-component SOFA min (best clinical values) | 0 | 0 | ✅ |
| 11 | `sofa_hourly.csv` row count | > 30,000 | 30,000+ | ✅ |
| 12 | Unique patient count | ≥ 992 | 992 | ✅ |
| 13 | All 16 required columns present | All present | All present | ✅ |
| 14 | SOFA components range [0–4], no NaN | Valid | Valid | ✅ |
| 15 | `sofa_total` range [0–24], no NaN | Valid | Valid | ✅ |
| 16 | Sepsis avg SOFA > non-sepsis avg SOFA | True | 3.53 > 1.84 | ✅ |
| 17 | Sepsis patient count plausible | 300–500 | 367 | ✅ |

**Overall Result: ✅ PASSED** (0.27s)

---

## TC-02 — ML Ensemble Model Validation

| Field | Details |
|-------|---------|
| **Test Case ID** | TC-02 |
| **Test Case Name** | ML Ensemble Model Validation |
| **File** | `tests/test_models.py` |
| **Function** | `test_ml_model_validation()` |
| **Type** | Integration |
| **Priority** | High |
| **Description** | Validates the Ensemble v3 model (LR + RF + XGB + GB) on the held-out 199-patient test set by rebuilding the exact same feature pipeline and split used during training, then asserting performance thresholds |

### Parameters / Inputs

| Parameter | Value |
|-----------|-------|
| Model | Ensemble v3 (Logistic Regression + Random Forest + XGBoost + Gradient Boosting) |
| Ensemble weights | [0.2, 0.2, 0.4, 0.2] |
| Decision threshold | 0.7697 |
| Train/test split | 80/20, `random_state=42`, stratified |
| Test set size | 199 patients |
| Feature count | 100 |
| Imputation | Training-set medians (`train_median_v3.pkl`) |
| Scaling | `StandardScaler` (fitted on SMOTE-augmented train set) |
| Data source | `data/labels/sofa_hourly.csv`, `data/labels/sepsis_onset.csv` |
| Artifact directory | `data/model/` |

### Prerequisites

- All 12 model artifacts present: `scaler_v3.pkl`, `logreg_v3.pkl`, `rf_v3.pkl`, `xgb_v3.pkl`, `gb_v3.pkl`, `ensemble_weights_v3.pkl`, `ensemble_threshold_v3.pkl`, `feature_cols_v3.pkl`, `train_median_v3.pkl`, `performance_v3.csv`, `test_predictions_v3.csv`, `feature_importance_v3.csv`
- Python packages: `scikit-learn`, `xgboost`, `joblib`, `pandas`, `numpy`

### Test Steps & Results

| Step | Check | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 1 | All 12 artifact `.pkl` / `.csv` files exist | Present | Present | ✅ |
| 2 | Ensemble weight count | 4 | 4 | ✅ |
| 3 | Weights sum to 1.0 | ≈ 1.0 | 1.0 | ✅ |
| 4 | Decision threshold in (0, 1) | True | 0.7697 | ✅ |
| 5 | Rebuilt feature matrix matches saved `feature_cols_v3.pkl` | 100 features | 100 | ✅ |
| 6 | Test set size after stratified split | 199 | 199 | ✅ |
| 7 | No duplicate patient IDs in test set | Zero duplicates | 0 | ✅ |
| 8 | No train/test patient overlap | Zero overlap | 0 | ✅ |
| 9 | Accuracy | ≥ 90% | **95.5%** | ✅ |
| 10 | Precision | ≥ 75% | **95.8%** | ✅ |
| 11 | Sensitivity (Recall) | ≥ 75% | **91.9%** | ✅ |
| 12 | AUROC | ≥ 90% | **99.4%** | ✅ |
| 13 | False Negatives (missed sepsis) | ≤ 20 | **6** | ✅ |
| 14 | False Positives | ≤ 25 | **3** | ✅ |
| 15 | Saved CSV AUROC drift from live | < 0.05 | 0.000 | ✅ |
| 16 | SOFA feature in top-10 importances | True | `sofa_range` rank #1 | ✅ |

### Confusion Matrix

|  | Predicted Non-Sepsis | Predicted Sepsis |
|--|----------------------|------------------|
| **Actual Non-Sepsis** | TN = 122 | FP = 3 |
| **Actual Sepsis** | FN = 6 | TP = 68 |

**Overall Result: ✅ PASSED** (6.70s)

---

## TC-03 — GNN Model Validation

| Field | Details |
|-------|---------|
| **Test Case ID** | TC-03 |
| **Test Case Name** | GNN Model Validation |
| **File** | `tests/test_gnn_alerts.py` |
| **Function** | `test_gnn_model_validation()` |
| **Type** | Integration |
| **Priority** | High |
| **Description** | Validates the GNN v3 model (SepsisGAT, 3-layer Graph Attention Network) on the held-out 199-patient test set, including checkpoint integrity, architecture config, parameter count, forward-pass shapes, graph construction, and inference performance |

### Parameters / Inputs

| Parameter | Value |
|-----------|-------|
| Model | SepsisGAT — 3-layer GAT |
| Architecture | `input_dim=100`, `hidden_dim=128`, `num_heads=4`, `dropout=0.3`, `num_layers=3` |
| Parameter count | 91,243 |
| Decision threshold | 0.0704 |
| Graph construction | Cosine-similarity patient graph, `k_neighbors=10`, `sim_threshold=0.5` |
| Train/test split | 80/20, `random_state=42`, stratified (same as TC-02) |
| Test set size | 199 patients (nodes) |
| Feature count | 100 |
| Checkpoint | `data/model/gnn_v3.pt` |
| Scaler | `data/model/scaler_gnn_v3.pkl` |

### Prerequisites

- `data/model/gnn_v3.pt` checkpoint exists with keys: `model_state_dict`, `model_config`, `threshold`, `feature_cols`, `test_metrics`
- Python packages: `torch`, `torch_geometric`, `scikit-learn`, `joblib`, `pandas`, `numpy`

### Test Steps & Results

| Step | Check | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| 1 | `gnn_v3.pt` checkpoint file exists | Present | Present | ✅ |
| 2 | All 5 checkpoint keys present | 5 keys | 5 keys | ✅ |
| 3 | `model_config.input_dim` | 100 | 100 | ✅ |
| 4 | `model_config.hidden_dim` | 128 | 128 | ✅ |
| 5 | `model_config.num_heads` | 4 | 4 | ✅ |
| 6 | `feature_cols` count | 100 | 100 | ✅ |
| 7 | Decision threshold in (0, 1) | True | 0.0704 | ✅ |
| 8 | Model parameter count | 50K–300K | **91,243** | ✅ |
| 9 | State dict loads without errors | No error | No error | ✅ |
| 10 | Test set size after split | 199 | 199 | ✅ |
| 11 | Graph node count | 199 | 199 | ✅ |
| 12 | Graph edge count | > 0 | > 0 | ✅ |
| 13 | Logits output shape | `(199, 2)` | `(199, 2)` | ✅ |
| 14 | Confidence output shape | `(199, 1)` | `(199, 1)` | ✅ |
| 15 | Confidence values in [0, 1] | True | True | ✅ |
| 16 | Accuracy | ≥ 90% | **94.5%** | ✅ |
| 17 | Precision | ≥ 75% | **92.0%** | ✅ |
| 18 | Sensitivity (Recall) | ≥ 75% | **93.2%** | ✅ |
| 19 | AUROC | ≥ 90% | **97.6%** | ✅ |
| 20 | False Negatives (missed sepsis) | ≤ 20 | **5** | ✅ |
| 21 | False Positives | ≤ 25 | **6** | ✅ |
| 22 | Saved checkpoint AUROC drift from live | < 0.05 | 0.000 | ✅ |

### Confusion Matrix

|  | Predicted Non-Sepsis | Predicted Sepsis |
|--|----------------------|------------------|
| **Actual Non-Sepsis** | TN = 119 | FP = 6 |
| **Actual Sepsis** | FN = 5 | TP = 69 |

**Overall Result: ✅ PASSED** (23.24s)

---

## TC-04 — End-to-End Pipeline Validation

| Field | Details |
|-------|---------|
| **Test Case ID** | TC-04 |
| **Test Case Name** | End-to-End Pipeline Validation |
| **File** | `tests/test_e2e.py` |
| **Function** | `test_end_to_end_pipeline()` |
| **Type** | System / End-to-End |
| **Priority** | Critical |
| **Description** | Validates the full pipeline across 8 stages: raw data files → cohort integrity → SOFA output → feature engineering → Ensemble v3 performance → GNN v3 performance → cross-model consistency → output artifact integrity |

### Parameters / Inputs

| Parameter | Value |
|-----------|-------|
| Raw data files | `cohort.csv`, `vitals_hourly.csv`, `labs_hourly.csv` |
| Label files | `sofa_hourly.csv`, `sepsis_onset.csv` |
| Total patients | 992 |
| Train / test split | 793 / 199 (`random_state=42`, stratified) |
| Feature count | 100 |
| Ensemble model | v3 (LR + RF + XGB + GB), threshold 0.7697 |
| GNN model | v3 (SepsisGAT 3-layer GAT), threshold 0.0704 |
| High-confidence threshold | ensemble probability > 0.80 or < 0.20 |
| Output CSVs checked | `performance_v3.csv`, `test_predictions_v3.csv`, `feature_importance_v3.csv`, `performance_gnn_v3.csv`, `test_predictions_gnn_v3.csv` |

### Prerequisites

- All raw data, label, and model artifact files present
- Python packages: `torch`, `torch_geometric`, `scikit-learn`, `xgboost`, `joblib`, `pandas`, `numpy`

### Test Steps & Results

| Step | Stage | Check | Expected | Actual | Status |
|------|-------|-------|----------|--------|--------|
| 1 | Stage 1 | All 5 data files exist and non-empty | Present | Present | ✅ |
| 2 | Stage 2 | Cohort unique patient count | ≥ 900 | 1,000 | ✅ |
| 3 | Stage 2 | No duplicate `icustay_id` in cohort | 0 duplicates | 0 | ✅ |
| 4 | Stage 2 | Age range (MIMIC ages >89 → 200–300) | 18–300 | 19–300 | ✅ |
| 5 | Stage 3 | `sofa_hourly.csv` includes all 992 patients | ≥ 992 | 992 | ✅ |
| 6 | Stage 3 | `sofa_total` no NaN, min ≥ 0 | Valid | Valid | ✅ |
| 7 | Stage 3 | Sepsis avg SOFA > non-sepsis avg SOFA | True | 3.53 > 1.84 | ✅ |
| 8 | Stage 4 | Post-aggregation patient count | 992 | 992 | ✅ |
| 9 | Stage 4 | Feature count per patient | 100 | 100 | ✅ |
| 10 | Stage 4 | Train split size | 793 | 793 | ✅ |
| 11 | Stage 4 | Test split size | 199 | 199 | ✅ |
| 12 | Stage 4 | Zero train/test patient overlap | 0 | 0 | ✅ |
| 13 | Stage 5 | Ensemble accuracy | ≥ 90% | **95.5%** | ✅ |
| 14 | Stage 5 | Ensemble precision | ≥ 75% | **95.8%** | ✅ |
| 15 | Stage 5 | Ensemble sensitivity | ≥ 75% | **91.9%** | ✅ |
| 16 | Stage 5 | Ensemble AUROC | ≥ 95% | **99.4%** | ✅ |
| 17 | Stage 6 | GNN accuracy | ≥ 90% | **94.5%** | ✅ |
| 18 | Stage 6 | GNN precision | ≥ 75% | **92.0%** | ✅ |
| 19 | Stage 6 | GNN sensitivity | ≥ 75% | **93.2%** | ✅ |
| 20 | Stage 6 | GNN AUROC | ≥ 90% | **97.6%** | ✅ |
| 21 | Stage 7 | Ensemble/GNN agreement on high-confidence cases | ≥ 70% | ✅ | ✅ |
| 22 | Stage 8 | All 5 output CSVs exist | Present | Present | ✅ |
| 23 | Stage 8 | Saved ensemble AUROC drift from live | < 0.05 | 0.000 | ✅ |
| 24 | Stage 8 | SOFA in top-5 feature importances | True | `sofa_range` rank #1 | ✅ |

**Overall Result: ✅ PASSED** (7.98s)

---

## Overall Performance Summary

| Model | Accuracy | Precision | Sensitivity | AUROC | TP | TN | FP | FN |
|-------|----------|-----------|-------------|-------|----|----|----|----|
| **Ensemble v3** (LR+RF+XGB+GB) | 95.5% | 95.8% | 91.9% | 99.4% | 68 | 122 | 3 | 6 |
| **GNN v3** (SepsisGAT 3-layer GAT) | 94.5% | 92.0% | 93.2% | 97.6% | 69 | 119 | 6 | 5 |

**All targets met:** Accuracy ≥ 90% · Precision ≥ 75% · Sensitivity ≥ 75%
