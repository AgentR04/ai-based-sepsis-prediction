# Sepsis Prediction System

A clinical machine learning system for **early-onset sepsis detection** in ICU patients, built on the **MIMIC-III** database. The system combines a high-performance **Ensemble ML model** (LR + RF + XGBoost + Gradient Boosting) with a **Graph Attention Network (SepsisGAT)** that leverages patient similarity graphs for contextual prediction.

---

## Results at a Glance

| Metric | Ensemble v3 | GNN v3 |
|---|---|---|
| Accuracy | 95.5% | 94.5% |
| Precision | 95.8% | 91.1% |
| Sensitivity (Recall) | 91.9% | 97.3% |
| F1-Score | 93.8% | 94.1% |
| AUROC | **0.9939** | 0.9778 |
| AUPRC | 0.9891 | 0.9383 |
| Brier Score | 0.0269 | 0.0603 |

> All results on unseen held-out test set (80/20 stratified split, `random_state=42`)

The **Ensemble** leads on precision and AUROC. The **GNN** leads on sensitivity — it catches more true sepsis cases, which is clinically prioritised to avoid missed diagnoses.

---

## Clinical Significance

Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. Early detection is critical — each hour of delayed treatment increases mortality by approximately 7%. This system:

- Computes **hourly SOFA scores** from raw vitals and lab values
- Aggregates 100 patient-level clinical features from ICU time-series
- Uses **SMOTE** to handle class imbalance (sepsis is relatively rare)
- Achieves **97.3% sensitivity** (GNN) — misses only 2 out of 74 sepsis cases in test set
- Top predictive feature: `sofa_range` (41% importance) — reflecting organ dysfunction trajectory

---

## Architecture

### Ensemble v3

Four base models trained independently, combined with learned weights:

```
Logistic Regression  (weight = 0.2)
Random Forest        (weight = 0.2)  ──► Weighted soft vote ──► Threshold 0.7697 ──► Prediction
XGBoost              (weight = 0.4)
Gradient Boosting    (weight = 0.2)
```

Weights and threshold optimised via grid search on validation AUROC + sensitivity.

### GNN v3 — SepsisGAT

A 3-layer Graph Attention Network connecting patients in a cosine-similarity graph:

```
Patient features (100-dim)
        │
  BatchNorm + Linear projection (→ 128-dim)
        │
  GAT Layer 1  (128-dim, 4 heads, dropout=0.3)
        │
  GAT Layer 2  (128-dim, 4 heads, dropout=0.3)
        │
  GAT Layer 3  (128-dim, 1 head)   ←── skip connection
        │
  Classifier MLP (128 → 64 → 32 → 2)
  Confidence head (128 → 1, Sigmoid)
```

**Graph construction**: Cosine similarity between patient feature vectors. Patients with similarity ≥ 0.5 are connected. Average ~8.7 edges per training patient, ~6.1 in test set.

**Parameters**: 91,243 total

---

## Feature Engineering

100 patient-level features are aggregated from hourly time-series:

### Vital Signs (aggregated: mean, max, min, std)
`heart_rate`, `map`, `resp_rate`, `spo2`, `temperature`

### Lab Values (aggregated: max, mean, min)
`lactate`, `creatinine`, `bilirubin`, `platelets`

### SOFA Components
`sofa_total`, `sofa_map`, `sofa_creatinine`, `sofa_platelets`, `sofa_bilirubin`

### Engineered Features
| Feature | Description |
|---|---|
| `sofa_range` | max − min SOFA (top feature, 41% importance) |
| `sofa_total_delta_max` | largest single-hour SOFA jump |
| `risk_composite` | SOFA×2 + tachycardia + hypotension + lactate flags |
| `sirs_score` | SIRS criterion count (HR, RR, temp) |
| `critical_hours_ratio` | fraction of hours with HR > 110 |
| `map_critical_ratio` | fraction of hours with MAP < 60 |
| `sofa_x_hr`, `sofa_x_map`, `sofa_x_lactate` | SOFA interaction terms |
| `vital_instability` | sum of HR/MAP/RR std devs |
| `sofa_peak_hour_frac` | normalised timing of peak SOFA |

### Top 10 Feature Importances (XGBoost)

| Rank | Feature | Contribution |
|---|---|---|
| 1 | sofa_range | 41.0% |
| 2 | sofa_total_max | 11.9% |
| 3 | sofa_total_delta_max | 3.1% |
| 4 | map_critical_sum | 1.4% |
| 5 | lactate_max | 1.4% |
| 6 | sofa_x_hr_mean | 1.2% |
| 7 | sofa_map_mean | 1.2% |
| 8 | map_hr_ratio_mean | 1.1% |
| 9 | sofa_total_delta_mean | 1.1% |
| 10 | sofa_x_map_max | 1.1% |

---

## SOFA Scoring

SOFA is computed hourly from four organ systems:

| Score | MAP (mmHg) | Creatinine | Bilirubin | Platelets ×10³ |
|---|---|---|---|---|
| 0 | ≥ 70 | < 1.2 | < 1.2 | ≥ 150 |
| 1 | < 70 | 1.2–1.9 | 1.2–1.9 | 100–149 |
| 2 | — | 2.0–3.4 | 2.0–5.9 | 50–99 |
| 3 | — | 3.5–4.9 | 6.0–11.9 | 20–49 |
| 4 | — | ≥ 5.0 | ≥ 12.0 | < 20 |

Sepsis is defined as a SOFA increase ≥ 2 from baseline (Sepsis-3 criteria).

Sepsis patients average **2.79× higher SOFA** than non-sepsis patients in this dataset.

---

## Cross-Validation Results

### Ensemble v3 — 5-Fold Stratified CV

| Fold | Accuracy | Precision | Sensitivity | F1 | AUROC |
|---|---|---|---|---|---|
| 1 | 95.0% | 90.0% | 97.3% | 93.5% | 0.9859 |
| 2 | 97.5% | 96.0% | 97.3% | 96.6% | 0.9880 |
| 3 | 96.5% | 92.3% | 98.6% | 95.4% | 0.9935 |
| 4 | 96.0% | 91.1% | 98.6% | 94.7% | 0.9883 |
| 5 | 97.0% | 93.5% | 98.6% | 96.0% | 0.9935 |
| **Mean** | **96.4%** | **92.6%** | **98.1%** | **95.3%** | **0.9899** |
| Std | 1.0% | 2.3% | 0.7% | 1.2% | 0.0035 |

### Bootstrap 95% Confidence Intervals (n = 1000)

| Metric | Estimate | 95% CI |
|---|---|---|
| Accuracy | 0.9547 | [0.9246 – 0.9799] |
| Precision | 0.9590 | [0.9077 – 1.0000] |
| Sensitivity | 0.9173 | [0.8513 – 0.9737] |
| AUROC | 0.9941 | [0.9859 – 0.9993] |
| AUPRC | 0.9894 | [0.9746 – 0.9989] |

---

## Project Structure

```
ipd_sepsis/
├── dashboard.py                  # Streamlit dashboard (5 tabs)
├── requirements.txt
├── pytest.ini
│
├── src/
│   ├── data/
│   │   ├── create_cohort.py      # MIMIC-III cohort extraction
│   │   ├── export_cohort.py
│   │   ├── extract_labs.py       # Lab value extraction
│   │   └── extract_vitals.py     # Vital sign extraction
│   ├── features/
│   │   └── build_features.py     # 100-feature engineering pipeline
│   ├── labels/
│   │   ├── compute_sofa.py       # Hourly SOFA computation
│   │   ├── detect_sepsis.py      # Sepsis-3 onset detection
│   │   └── train_dataset.py      # Train/test dataset builder
│   ├── models/
│   │   ├── train_models_maximize_v3.py   # Ensemble v3 training (main)
│   │   └── train_gnn_v3.py              # GNN v3 training (main)
│   └── alerts/
│       ├── gnn_model.py          # SepsisGAT architecture
│       ├── patient_graph.py      # Graph construction utilities
│       ├── alert_generator.py    # Risk alert generation
│       └── train_gnn.py
│
├── tests/
│   ├── test_sofa.py              # SOFA boundary + data integrity tests
│   ├── test_features.py          # Feature engineering tests
│   ├── test_labels.py            # Label/sepsis detection tests
│   ├── test_models.py            # Ensemble validation (12 checks)
│   ├── test_gnn_alerts.py        # GNN validation (12 checks)
│   └── test_e2e.py               # End-to-end pipeline test
│
├── data/
│   ├── raw/                      # MIMIC-III source data
│   │   ├── cohort.csv
│   │   ├── vitals_hourly.csv
│   │   └── labs_hourly.csv
│   ├── labels/
│   │   ├── sofa_hourly.csv       # Hourly SOFA scores (39,331 rows)
│   │   └── sepsis_onset.csv      # Sepsis onset labels
│   ├── features/
│   │   └── features.csv          # 100 engineered features
│   └── model/                    # Saved artifacts
│       ├── logreg_v3.pkl
│       ├── rf_v3.pkl
│       ├── xgb_v3.pkl
│       ├── gb_v3.pkl
│       ├── scaler_v3.pkl
│       ├── ensemble_weights_v3.pkl
│       ├── ensemble_threshold_v3.pkl
│       ├── feature_cols_v3.pkl
│       ├── feature_importance_v3.csv
│       ├── performance_v3.csv
│       ├── cv_results_v3.csv
│       ├── test_predictions_v3.csv
│       ├── gnn_v3.pt
│       ├── scaler_gnn_v3.pkl
│       ├── performance_gnn_v3.csv
│       ├── cv_results_gnn_v3.csv
│       └── test_predictions_gnn_v3.csv
│
└── docs/
    ├── PROJECT_STATUS.md
    ├── CLINICAL_MODEL_REPORT.md
    ├── GNN_ALERTS_GUIDE.md
    └── TESTING_GUIDE.md
```

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ipd_sepsis

# Create virtual environment
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**

| Package | Purpose |
|---|---|
| pandas, numpy | Data processing |
| scikit-learn | ML models, preprocessing, metrics |
| xgboost | XGBoost classifier |
| imbalanced-learn | SMOTE oversampling |
| torch ≥ 2.1 | GNN training |
| torch-geometric ≥ 2.4 | GAT layers, graph data structures |
| streamlit | Dashboard |
| plotly | Interactive charts |
| joblib | Model serialisation |

---

## Running the Models

### Train Ensemble v3

```bash
python src/models/train_models_maximize_v3.py
```

Output includes:
- Per-model metrics (LR, RF, XGB, GB)
- Ensemble weight and threshold optimisation
- Held-out test performance
- 5-fold cross-validation
- Top 20 feature importances with clinical labels
- Final model comparison summary

### Train GNN v3

```bash
python src/models/train_gnn_v3.py
```

Output includes:
- Training progress (every 50 epochs)
- Early stopping with best-model restoration
- Held-out test performance
- GNN vs Ensemble v3 comparison table
- Per-patient predictions for top risk cases

---

## Running the Dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501` with 5 tabs:

| Tab | Contents |
|---|---|
| **Overview** | Key metric cards, model comparison bar chart, confusion matrices, pipeline diagram |
| **SOFA Analysis** | Score distributions, component comparisons, time trends, boundary test results |
| **Ensemble Model** | Feature importances, 5-fold CV, probability distributions, bootstrap CIs |
| **GNN Model** | Architecture details, patient similarity graph visualisation, confidence distributions |
| **Full Test Report** | All validation checks (30/30 PASS), radar chart, per-patient prediction table |

---

## Running Tests

```bash
# All tests
pytest tests/ -v -s

# Individual test files
pytest tests/test_sofa.py -v -s        # 6/6 checks — SOFA scoring
pytest tests/test_models.py -v -s      # 12/12 checks — Ensemble validation
pytest tests/test_gnn_alerts.py -v -s  # 12/12 checks — GNN validation
pytest tests/test_e2e.py -v -s         # End-to-end pipeline
```

### Test Coverage

| Test File | Checks | Runtime |
|---|---|---|
| `test_sofa.py` | 6/6 | ~0.3s |
| `test_models.py` | 12/12 | ~7s |
| `test_gnn_alerts.py` | 12/12 | ~11s |
| `test_e2e.py` | Pass | ~15s |
| **Total** | **30/30** | **~14s** |

Each test file validates:
- `test_sofa.py` — 23 SOFA boundary checks, NaN integrity, distribution, separability
- `test_models.py` — individual models, ensemble, bootstrap CI, 5-fold CV, calibration, feature importance
- `test_gnn_alerts.py` — checkpoint validity, architecture config, graph structure, inference, metrics
- `test_e2e.py` — full data → features → model → prediction pipeline

---

## Data Pipeline

```
MIMIC-III raw tables
        │
        ▼
create_cohort.py      ──► cohort.csv         (ICU stays, demographics)
extract_vitals.py     ──► vitals_hourly.csv  (HR, MAP, RR, SpO2, Temp)
extract_labs.py       ──► labs_hourly.csv    (Lactate, Creatinine, Bili, Plt)
        │
        ▼
compute_sofa.py       ──► sofa_hourly.csv    (per-hour SOFA components)
detect_sepsis.py      ──► sepsis_onset.csv   (Sepsis-3 onset timestamps)
        │
        ▼
build_features.py     ──► features.csv       (100 patient-level features)
train_dataset.py      ──► train_dataset.csv  (labelled ML-ready dataset)
        │
        ├──► train_models_maximize_v3.py  ──► Ensemble v3 artifacts
        └──► train_gnn_v3.py             ──► GNN v3 artifacts
```

---

## Clinical Notes

> **This system is a research prototype.** Before any clinical deployment:
>
> - Validate on your institution's own patient population
> - Conduct prospective clinical evaluation
> - Calibrate alert thresholds to your clinical context and bed capacity
> - Obtain appropriate regulatory clearance
> - Integrate monitoring and feedback loops
> - Ensure compliance with local data governance requirements

**Why SOFA range is the top feature**: A rising SOFA score indicates worsening multi-organ dysfunction. The range (max − min) captures the trajectory — even patients who arrive critically ill but stabilise are distinguished from those who progressively deteriorate.

**Why GNN sensitivity > Ensemble sensitivity**: The graph structure allows the model to incorporate signals from clinically similar patients who developed sepsis, even when a patient's own features are borderline. This "neighbour evidence" boosts recall at the cost of some precision.

---

## References

- **Sepsis-3 Definition**: Singer M, et al. (2016). "The Third International Consensus Definitions for Sepsis and Septic Shock." *JAMA* 315(8):801–810.
- **MIMIC-III**: Johnson AEW, et al. (2016). "MIMIC-III, a freely accessible critical care database." *Scientific Data* 3:160035.
- **Graph Attention Networks**: Veličković P, et al. (2018). "Graph Attention Networks." *ICLR 2018*.
- **XGBoost**: Chen T, Guestrin C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD 2016*.
- **SMOTE**: Chawla NV, et al. (2002). "SMOTE: Synthetic Minority Over-sampling Technique." *JAIR* 16:321–357.

---

**Version**: 3.0  
**Last Updated**: February 2026  
**Python**: 3.13  
**Framework**: PyTorch 2.x + PyTorch Geometric 2.x


## Features

### Core Sepsis Prediction
- **Feature Engineering**: Automated extraction and aggregation of clinical vitals and lab values
- **SOFA Score Computation**: Calculate Sequential Organ Failure Assessment scores
- **Sepsis Detection**: Identify sepsis onset using Sepsis-3 criteria
- **Multiple ML Models**: 
  - Logistic Regression
  - Random Forest
  - Gradient Boosting (XGBoost)
  - Neural Networks
- **Comprehensive Testing**: Full test suite with >90% code coverage

### 🆕 GNN-Based Alert System (NEW!)
- **Patient Similarity Graphs**: Connect patients based on clinical features and missing data patterns
- **Graph Neural Networks**: Learn from similar historical cases to predict sepsis risk
- **Early Warning Alerts**: Generate tiered alerts (CRITICAL/HIGH/MEDIUM/LOW) for patients with incomplete data
- **Similar Case Matching**: Identify historical patients with similar presentations
- **Intelligent Missingness Handling**: Leverage patterns in missing data for better predictions

## Project Structure

```
ipd_sepsis/
├── src/
│   ├── data/           # Data extraction and cohort creation
│   ├── features/       # Feature engineering pipeline
│   ├── labels/         # SOFA scores and sepsis detection
│   ├── models/         # Traditional ML model training
│   └── alerts/         # 🆕 GNN-based alert system
│       ├── patient_graph.py      # Graph construction
│       ├── gnn_model.py          # GNN architecture
│       ├── alert_generator.py   # Alert generation
│       └── train_gnn.py         # Training script
├── tests/              # Comprehensive test suite
├── data/               # Data directory
├── GNN_ALERTS_GUIDE.md # 🆕 Detailed GNN documentation
├── demo_gnn_alerts.py  # 🆕 Interactive demo
├── TESTING_GUIDE.md    # Testing documentation
└── requirements.txt    # Dependencies
```

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ipd_sepsis

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies**:
- Python 3.8+
- pandas, numpy, scikit-learn
- xgboost, imbalanced-learn
- PyTorch 2.1+ (for GNN alerts)
- PyTorch Geometric 2.4+ (for GNN alerts)

## Quick Start

### 1. Traditional Sepsis Prediction

```bash
# Train traditional ML models
python -m src.models.train_models_optimized

# Run tests
pytest tests/ -v
```

### 2. GNN-Based Alert System

```bash
# Run interactive demo
python demo_gnn_alerts.py

# Train GNN on your data
python -m src.alerts.train_gnn \
    --features data/features/features.csv \
    --labels data/labels/sepsis_onset.csv \
    --train-dataset data/model/train_dataset.csv \
    --epochs 100

# Run GNN tests
pytest tests/test_gnn_alerts.py -v
```

## Usage Examples

### Traditional Model Prediction

```python
import pandas as pd
import pickle

# Load trained model
with open('data/model/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load patient features
features = pd.read_csv('patient_features.csv')

# Predict sepsis risk
predictions = model.predict_proba(features)[:, 1]
high_risk = predictions > 0.7
```

### GNN Alert Generation

```python
from src.alerts.gnn_model import PatientSimilarityGNN
from src.alerts.alert_generator import SepsisAlertGenerator
import torch
import pickle

# Load trained GNN
checkpoint = torch.load('data/model/gnn_sepsis_model.pt')
model = PatientSimilarityGNN(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Load graph builder
with open('data/model/graph_builder.pkl', 'rb') as f:
    graph_builder = pickle.load(f)

# Create alert generator
alert_gen = SepsisAlertGenerator(model, graph_builder)

# Generate alerts for new patients
alerts = alert_gen.generate_alerts(current_patients_df, patient_ids)

# Show high-priority alerts
for alert in alerts:
    if alert.alert_level in ['CRITICAL', 'HIGH']:
        print(f"{alert.patient_id}: {alert.risk_score:.1%} risk")
        print(f"  Missing data: {alert.missingness_ratio:.1%}")
        print(f"  Recommendation: {alert._get_recommendation()}")
```

## GNN Alert System

The GNN-based alert system addresses a critical clinical challenge: **predicting sepsis risk for patients with incomplete clinical data**.

### Key Innovation

Traditional ML models struggle with missing data. The GNN system:
1. **Builds patient similarity graphs** connecting patients with similar clinical profiles
2. **Learns from complete AND incomplete data** by propagating information through the graph
3. **Identifies historical similar cases** who developed sepsis
4. **Generates early alerts** for at-risk patients, enabling earlier intervention

### When to Use GNN Alerts

- ✅ Patient has significant missing data (>15% of features)
- ✅ Need to make predictions before all labs return
- ✅ Want clinical context from similar historical cases
- ✅ Interested in uncertainty quantification (confidence scores)

### Alert Levels

| Level | Risk Threshold | Action |
|-------|---------------|--------|
| **CRITICAL** | ≥75% or High+High Missing | Immediate assessment, consider empiric treatment |
| **HIGH** | ≥60% | Close monitoring, complete workup, assess sepsis criteria |
| **MEDIUM** | ≥45% | Increased monitoring, prioritize data collection |
| **LOW** | <45% | Routine monitoring |

## Documentation

- **[GNN_ALERTS_GUIDE.md](GNN_ALERTS_GUIDE.md)**: Comprehensive GNN documentation
  - Architecture details
  - Training guide
  - Clinical interpretation
  - API reference
  
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)**: Testing documentation
  - Test structure
  - Running tests
  - Adding new tests

- **[CLINICAL_MODEL_REPORT.md](CLINICAL_MODEL_REPORT.md)**: Model performance report
  - Evaluation metrics
  - Model comparison
  - Clinical recommendations

## Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test modules
pytest tests/test_features.py -v
pytest tests/test_models.py -v
pytest tests/test_gnn_alerts.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

## Performance

### Traditional Models
- **Precision**: 0.78-0.85
- **Recall**: 0.75-0.82
- **F1 Score**: 0.76-0.83
- **AUROC**: 0.85-0.90

### GNN Alert System
- **Precision**: 0.70-0.85 (with missing data)
- **Recall**: 0.75-0.90
- **F1 Score**: 0.72-0.87
- **AUROC**: 0.85-0.92
- **Handles**: 15-50% missing data effectively

## Clinical Validation

⚠️ **Important**: This is a research/demonstration system. Before clinical deployment:

1. Validate on your institution's data
2. Conduct prospective clinical trials
3. Establish alert thresholds appropriate for your setting
4. Integrate with existing clinical workflows
5. Ensure regulatory compliance (FDA, CE marking, etc.)
6. Implement clinical oversight and quality monitoring

## Contributing

Contributions are welcome! Areas of interest:

- Temporal GNN models for time-series data
- Explainability/interpretability features
- Scalability improvements
- Integration with EHR systems
- Clinical validation studies

## License

[Specify your license]

## Citation

If you use this system in your research, please cite:

```
[Add citation information]
```

## References

### Sepsis Definitions
- Singer M, et al. (2016). "The Third International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3)." JAMA.

### Graph Neural Networks
- Kipf TN, Welling M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR.
- Veličković P, et al. (2018). "Graph Attention Networks." ICLR.

### Missing Data in Healthcare
- Little RJA, Rubin DB. (2019). "Statistical Analysis with Missing Data." Wiley.

## Contact

[Your contact information]

---

**Last Updated**: February 2026
**Version**: 2.0.0 (with GNN Alert System)
