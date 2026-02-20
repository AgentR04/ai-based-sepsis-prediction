import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix)

preds = pd.read_csv('data/model/test_predictions_v3.csv')
test_ids = preds['icustay_id'].values

df = pd.read_csv('data/model/train_dataset.csv')
import joblib
feature_cols = joblib.load('data/model/feature_cols_v3.pkl')
scaler = joblib.load('data/model/scaler_v3.pkl')
train_median = joblib.load('data/model/train_median_v3.pkl')

test_df = df[df['icustay_id'].isin(test_ids)].copy()
test_df = test_df.fillna(train_median)
X_test = test_df[feature_cols].values
X_test_scaled = scaler.transform(X_test)
y_test = test_df['label_max'].values

models = {}
for name, path in [('Logistic Regression', 'data/model/logreg_v3.pkl'),
                   ('Random Forest', 'data/model/rf_v3.pkl'),
                   ('XGBoost', 'data/model/xgb_v3.pkl'),
                   ('Gradient Boosting', 'data/model/gb_v3.pkl')]:
    models[name] = joblib.load(path)

print(f"Test: {len(y_test)} patients | Sepsis: {y_test.sum()} | Non-sepsis: {(y_test==0).sum()}")
for name, model in models.items():
    prob = model.predict_proba(X_test_scaled)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"{name}: Acc={accuracy_score(y_test,pred):.4f} Prec={precision_score(y_test,pred):.4f} Sens={recall_score(y_test,pred):.4f} F1={f1_score(y_test,pred):.4f} AUROC={roc_auc_score(y_test,prob):.4f} AUPRC={average_precision_score(y_test,prob):.4f} TP={tp} FP={fp} FN={fn} TN={tn}")

ens_prob = preds.set_index('icustay_id')['predicted_prob'].loc[test_df['icustay_id']].values
ens_threshold = joblib.load('data/model/ensemble_threshold_v3.pkl')
ens_pred = (ens_prob >= ens_threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_test, ens_pred).ravel()
print(f"Ensemble v3: Acc={accuracy_score(y_test,ens_pred):.4f} Prec={precision_score(y_test,ens_pred):.4f} Sens={recall_score(y_test,ens_pred):.4f} F1={f1_score(y_test,ens_pred):.4f} AUROC={roc_auc_score(y_test,ens_prob):.4f} AUPRC={average_precision_score(y_test,ens_prob):.4f} TP={tp} FP={fp} FN={fn} TN={tn}")
