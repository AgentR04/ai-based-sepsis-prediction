"""
Optimized Model Training with Advanced Techniques
Goal: Achieve 90-95% AUROC through:
1. Hyperparameter optimization
2. Advanced feature engineering
3. Ensemble methods
4. Cross-validation
5. Two-stage class balancing (Undersampling + SMOTE)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix
)
from sklearn.feature_selection import SelectFromModel, RFE

# ------------------------
# Paths
# ------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "model" / "train_dataset_balanced.csv"  # Using pre-balanced dataset

# ------------------------
# Configuration Parameters
# ------------------------
# Clinical Optimization: Prioritize SENSITIVITY (recall) over precision
# For sepsis detection, missing a case (false negative) is worse than false alarm (false positive)
OPTIMIZE_FOR_SENSITIVITY = True

# ------------------------
# Load dataset
# ------------------------
df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("OPTIMIZED SEPSIS PREDICTION MODEL TRAINING")
print("=" * 70)
print(f"\nDataset shape: {df.shape}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"Positive rate: {df['label'].sum() / len(df) * 100:.2f}%")

# ------------------------
# Feature Engineering: Add Interaction Features
# ------------------------
print("\n" + "=" * 70)
print("PHASE 1: ADVANCED FEATURE ENGINEERING")
print("=" * 70)

FEATURE_COLS = [c for c in df.columns if c not in ["icustay_id", "hour", "label"]]
SOFA_FEATURES = [c for c in FEATURE_COLS if 'sofa' in c.lower()]
print(f"Original features: {len(FEATURE_COLS)}")
print(f"✓ SOFA features: {SOFA_FEATURES}")

# Create interaction features for key clinical indicators
if 'sofa_total' in df.columns and 'heart_rate' in df.columns:
    df['sofa_hr_interaction'] = df['sofa_total'] * df['heart_rate']
    
if 'map' in df.columns and 'heart_rate' in df.columns:
    df['shock_index'] = df['heart_rate'] / (df['map'] + 1e-6)  # Avoid division by zero
    
if 'platelets' in df.columns and 'creatinine' in df.columns:
    df['plt_creat_ratio'] = df['platelets'] / (df['creatinine'] + 1e-6)

# Add trend features if rolling features exist
rolling_features = [c for c in df.columns if '_6h' in c and '_mean_' in c]
for col in rolling_features[:5]:  # Top 5 rolling features
    base_col = col.replace('_mean_6h', '')
    if base_col in df.columns:
        df[f'{base_col}_trend'] = df[base_col] - df[col]

# Update feature columns
FEATURE_COLS = [c for c in df.columns if c not in ["icustay_id", "hour", "label"]]
print(f"Enhanced features: {len(FEATURE_COLS)}")

# ------------------------
# Split by ICU stay (NO data leakage)
# ------------------------
icu_ids = df["icustay_id"].unique()
train_ids, val_ids = train_test_split(icu_ids, test_size=0.2, random_state=42, stratify=None)

train_df = df[df["icustay_id"].isin(train_ids)]
val_df = df[df["icustay_id"].isin(val_ids)]

X_train = train_df[FEATURE_COLS]
y_train = train_df["label"]

X_val = val_df[FEATURE_COLS]
y_val = val_df["label"]

print(f"Train: {len(X_train)} samples, {y_train.sum()} positive ({y_train.sum()/len(y_train)*100:.2f}%)")
print(f"Val: {len(X_val)} samples, {y_val.sum()} positive ({y_val.sum()/len(y_val)*100:.2f}%)")

# ------------------------
# Impute missing values with median
# ------------------------
X_train = X_train.fillna(X_train.median())
X_val = X_val.fillna(X_train.median())

# Remove features with zero variance
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.01)
X_train = pd.DataFrame(
    selector.fit_transform(X_train),
    columns=X_train.columns[selector.get_support()],
    index=X_train.index
)
X_val = pd.DataFrame(
    selector.transform(X_val),
    columns=X_train.columns,
    index=X_val.index
)

print(f"Features after variance filter: {X_train.shape[1]}")

# ------------------------
# Standardize features
# ------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Dataset is already balanced - no need for SMOTE
X_train_res = X_train_scaled
y_train_res = y_train

print("\n" + "=" * 70)
print("PHASE 2: USING PRE-BALANCED DATASET")
print("=" * 70)
print(f"Training samples: {len(X_train_res)}")
print(f"Positive: {y_train_res.sum()} ({y_train_res.sum()/len(y_train_res)*100:.2f}%)")
print(f"Negative: {(y_train_res==0).sum()} ({(y_train_res==0).sum()/len(y_train_res)*100:.2f}%)")
print(f"Balance ratio: {(y_train_res==0).sum() / y_train_res.sum():.2f}:1")

# ------------------------
# OPTIMIZED MODELS WITH HYPERPARAMETER TUNING
# ------------------------
print("\n" + "=" * 70)
print("PHASE 3: MODEL TRAINING WITH HYPERPARAMETER OPTIMIZATION")
print("=" * 70)

results = {}

# Model 1: Optimized Random Forest
print("\n[1/4] Training Optimized Random Forest...")
rf_params = {
    'n_estimators': [300, 500],
    'max_depth': [15, None],
    'min_samples_leaf': [1, 2],  # Lower values to capture positive cases better
    'class_weight': ['balanced', 'balanced_subsample']
}

# Use recall as scoring metric if optimizing for sensitivity
scoring_metric = 'recall' if OPTIMIZE_FOR_SENSITIVITY else 'roc_auc'

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params,
    cv=2,  # Reduced CV folds for speed
    scoring=scoring_metric,
    n_jobs=2,  # Limit parallelization to avoid timeout
    verbose=1
)
rf_grid.fit(X_train_res, y_train_res)
rf_best = rf_grid.best_estimator_

rf_probs = rf_best.predict_proba(X_val_scaled)[:, 1]
rf_preds = rf_best.predict(X_val_scaled)

results['Random Forest (Optimized)'] = {
    'probs': rf_probs,
    'preds': rf_preds,
    'auroc': roc_auc_score(y_val, rf_probs),
    'auprc': average_precision_score(y_val, rf_probs),
    'accuracy': accuracy_score(y_val, rf_preds),
    'precision': precision_score(y_val, rf_preds),
    'recall': recall_score(y_val, rf_preds),
    'f1': f1_score(y_val, rf_preds),
    'best_params': rf_grid.best_params_
}

print(f"AUROC: {results['Random Forest (Optimized)']['auroc']:.4f}")
print(f"Accuracy: {results['Random Forest (Optimized)']['accuracy']:.4f}")
print(f"Best params: {rf_grid.best_params_}")

# Model 2: Optimized XGBoost
print("\n[2/4] Training Optimized XGBoost...")
xgb_params = {
    'n_estimators': [300, 500],
    'max_depth': [6, 10],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8],
    'scale_pos_weight': [len(y_train_res[y_train_res==0]) / len(y_train_res[y_train_res==1])]
}

xgb_grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
    xgb_params,
    cv=2,  # Reduced CV folds for speed
    scoring=scoring_metric,
    n_jobs=2,
    verbose=1
)
xgb_grid.fit(X_train_res, y_train_res)
xgb_best = xgb_grid.best_estimator_

xgb_probs = xgb_best.predict_proba(X_val_scaled)[:, 1]
xgb_preds = xgb_best.predict(X_val_scaled)

results['XGBoost (Optimized)'] = {
    'probs': xgb_probs,
    'preds': xgb_preds,
    'auroc': roc_auc_score(y_val, xgb_probs),
    'auprc': average_precision_score(y_val, xgb_probs),
    'accuracy': accuracy_score(y_val, xgb_preds),
    'precision': precision_score(y_val, xgb_preds),
    'recall': recall_score(y_val, xgb_preds),
    'f1': f1_score(y_val, xgb_preds),
    'best_params': xgb_grid.best_params_
}

print(f"AUROC: {results['XGBoost (Optimized)']['auroc']:.4f}")
print(f"Accuracy: {results['XGBoost (Optimized)']['accuracy']:.4f}")
print(f"Best params: {xgb_grid.best_params_}")

# Model 3: Gradient Boosting
print("\n[3/4] Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train_res, y_train_res)

gb_probs = gb.predict_proba(X_val_scaled)[:, 1]
gb_preds = gb.predict(X_val_scaled)

results['Gradient Boosting'] = {
    'probs': gb_probs,
    'preds': gb_preds,
    'auroc': roc_auc_score(y_val, gb_probs),
    'auprc': average_precision_score(y_val, gb_probs),
    'accuracy': accuracy_score(y_val, gb_preds),
    'precision': precision_score(y_val, gb_preds),
    'recall': recall_score(y_val, gb_preds),
    'f1': f1_score(y_val, gb_preds)
}

print(f"AUROC: {results['Gradient Boosting']['auroc']:.4f}")
print(f"Accuracy: {results['Gradient Boosting']['accuracy']:.4f}")

# Model 4: Ensemble Voting Classifier
print("\n[4/4] Training Ensemble (Soft Voting)...")
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_best),
        ('xgb', xgb_best),
        ('gb', gb)
    ],
    voting='soft',
    weights=[2, 2, 1]  # Weight best models more
)
ensemble.fit(X_train_res, y_train_res)

ensemble_probs = ensemble.predict_proba(X_val_scaled)[:, 1]
ensemble_preds = ensemble.predict(X_val_scaled)

results['Ensemble (Voting)'] = {
    'probs': ensemble_probs,
    'preds': ensemble_preds,
    'auroc': roc_auc_score(y_val, ensemble_probs),
    'auprc': average_precision_score(y_val, ensemble_probs),
    'accuracy': accuracy_score(y_val, ensemble_preds),
    'precision': precision_score(y_val, ensemble_preds),
    'recall': recall_score(y_val, ensemble_preds),
    'f1': f1_score(y_val, ensemble_preds)
}

print(f"AUROC: {results['Ensemble (Voting)']['auroc']:.4f}")
print(f"Accuracy: {results['Ensemble (Voting)']['accuracy']:.4f}")

# ------------------------
# FINAL RESULTS SUMMARY
# ------------------------
print("\n" + "=" * 70)
print("FINAL PERFORMANCE COMPARISON")
print("=" * 70)

results_df = pd.DataFrame([
    {
        'Model': name,
        'AUROC': data['auroc'],
        'AUPRC': data['auprc'],
        'Accuracy': data['accuracy'],
        'Precision': data['precision'],
        'Recall': data['recall'],
        'F1-Score': data['f1']
    }
    for name, data in results.items()
]).sort_values('AUROC', ascending=False)

print("\n", results_df.to_string(index=False))

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_auroc = results_df.iloc[0]['AUROC']

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name}")
print(f"   AUROC: {best_auroc:.4f} ({best_auroc*100:.2f}%)")
print(f"   Accuracy: {results_df.iloc[0]['Accuracy']:.4f} ({results_df.iloc[0]['Accuracy']*100:.2f}%)")
print(f"{'='*70}")

# Detailed classification report for best model
best_preds = results[best_model_name]['preds']
print(f"\nDetailed Classification Report ({best_model_name}):")
print(classification_report(y_val, best_preds, target_names=['No Sepsis', 'Sepsis']))

# Confusion matrix
cm = confusion_matrix(y_val, best_preds)
print(f"\nConfusion Matrix:")
print(f"                Predicted")
print(f"              No Sepsis  Sepsis")
print(f"Actual No     {cm[0,0]:8d}  {cm[0,1]:6d}")
print(f"       Sepsis {cm[1,0]:8d}  {cm[1,1]:6d}")

# Calculate specificity and sensitivity
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print(f"\nClinical Metrics:")
print(f"  Sensitivity (Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"  Specificity:          {specificity:.4f} ({specificity*100:.2f}%)")

# Feature importance (for best tree-based model)
if 'Random Forest' in best_model_name or 'XGBoost' in best_model_name:
    print(f"\n{'='*70}")
    print(f"TOP 20 MOST IMPORTANT FEATURES ({best_model_name})")
    print(f"{'='*70}")
    
    if 'Random Forest' in best_model_name:
        importances = rf_best.feature_importances_
    elif 'XGBoost' in best_model_name:
        importances = xgb_best.feature_importances_
    
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False).head(20)
    
    print(feature_importance.to_string(index=False))

print(f"\n{'='*70}")
print("OPTIMIZATION COMPLETE")
print(f"{'='*70}\n")

# Save results
results_df.to_csv(PROJECT_ROOT / "data" / "model" / "optimized_results.csv", index=False)
print(f"Results saved to: data/model/optimized_results.csv")
