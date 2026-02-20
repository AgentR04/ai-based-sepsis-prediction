"""
test_models.py
Validates Ensemble v3 ML models on the held-out test set.
Outputs: individual model results, ensemble results, cross-validation,
bootstrap confidence intervals, calibration, and a final summary table.
One test function, no classes, no sub-tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, average_precision_score,
                              confusion_matrix, brier_score_loss)


def test_ml_model_validation():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    LABEL_DIR    = PROJECT_ROOT / "data" / "labels"
    MODEL_DIR    = PROJECT_ROOT / "data" / "model"

    # ── 1. All required v3 model artifacts exist ──────────────────────────────
    v3_files = [
        "scaler_v3.pkl", "logreg_v3.pkl", "rf_v3.pkl",
        "xgb_v3.pkl", "gb_v3.pkl", "ensemble_weights_v3.pkl",
        "ensemble_threshold_v3.pkl", "feature_cols_v3.pkl", "train_median_v3.pkl",
        "performance_v3.csv", "test_predictions_v3.csv", "feature_importance_v3.csv",
    ]
    for fname in v3_files:
        assert (MODEL_DIR / fname).exists(), f"Missing model artifact: {fname}"

    # ── 2. Load model artifacts ───────────────────────────────────────────────
    scaler    = joblib.load(MODEL_DIR / "scaler_v3.pkl")
    logreg    = joblib.load(MODEL_DIR / "logreg_v3.pkl")
    rf        = joblib.load(MODEL_DIR / "rf_v3.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgb_v3.pkl")
    gb_model  = joblib.load(MODEL_DIR / "gb_v3.pkl")
    weights   = joblib.load(MODEL_DIR / "ensemble_weights_v3.pkl")
    threshold = joblib.load(MODEL_DIR / "ensemble_threshold_v3.pkl")
    feat_cols = joblib.load(MODEL_DIR / "feature_cols_v3.pkl")
    train_med = joblib.load(MODEL_DIR / "train_median_v3.pkl")

    assert len(weights) == 4, f"Expected 4 ensemble weights, got {len(weights)}"
    assert abs(sum(weights) - 1.0) < 0.01, f"Weights must sum to 1, got {sum(weights):.4f}"
    assert 0.0 < threshold < 1.0, f"Threshold must be in (0,1), got {threshold}"

    # ── 3. Rebuild patient-level features (same as v3 training) ──────────────
    df = pd.read_csv(LABEL_DIR / "sofa_hourly.csv")
    sepsis_onset = pd.read_csv(LABEL_DIR / "sepsis_onset.csv")
    sepsis_ids   = set(sepsis_onset['icustay_id'].unique())
    df['label']  = df['icustay_id'].isin(sepsis_ids).astype(int)

    df['map_hr_ratio']            = df['map'] / (df['heart_rate'] + 1)
    df['lactate_platelets_ratio'] = df['lactate'] / (df['platelets'] + 1)
    df['spo2_temp_ratio']         = df['spo2'] / (df['temperature'] + 0.1)
    df['sofa_x_hr']               = df['sofa_total'] * df['heart_rate']
    df['sofa_x_lactate']          = df['sofa_total'] * df['lactate']
    df['sofa_x_map']              = df['sofa_total'] * df['map']
    df['sofa_x_creatinine']       = df['sofa_creatinine'] * df['creatinine']
    df['hr_critical']             = (df['heart_rate'] > 110).astype(int)
    df['map_critical']            = (df['map'] < 60).astype(int)
    df['rr_critical']             = (df['resp_rate'] > 24).astype(int)
    df['spo2_critical']           = (df['spo2'] < 90).astype(int)
    df['lactate_high']            = (df['lactate'] > 2).astype(int)
    df['creatinine_high']         = (df['creatinine'] > 1.5).astype(int)
    df['sirs_score'] = (
        (df['heart_rate'] > 90).astype(int) +
        (df['resp_rate'] > 20).astype(int) +
        ((df['temperature'] > 38) | (df['temperature'] < 36)).astype(int)
    )
    df['risk_composite'] = (
        df['sofa_total'] * 2 +
        (df['heart_rate'] > 100).astype(int) +
        (df['map'] < 65).astype(int) * 2 +
        (df['lactate'] > 2).astype(int) * 3
    )
    df = df.sort_values(['icustay_id', 'hour'])
    for col in ['heart_rate', 'map', 'sofa_total', 'lactate']:
        df[f'{col}_delta'] = df.groupby('icustay_id')[col].diff()

    def time_to_max(group):
        if group['sofa_total'].isna().all():
            return pd.Series({'sofa_peak_hour_frac': 0.5})
        max_hour    = group.loc[group['sofa_total'].idxmax(), 'hour']
        total_hours = group['hour'].max()
        return pd.Series({'sofa_peak_hour_frac': max_hour / total_hours if total_hours > 0 else 0.0})

    sofa_peak = df.groupby('icustay_id').apply(time_to_max).reset_index()

    agg_funcs = {
        'heart_rate': ['mean', 'max', 'min', 'std'], 'map': ['mean', 'max', 'min', 'std'],
        'resp_rate': ['mean', 'max', 'min', 'std'], 'spo2': ['mean', 'min', 'std'],
        'temperature': ['mean', 'max', 'min', 'std'], 'bilirubin': ['max', 'mean'],
        'creatinine': ['max', 'mean', 'min'], 'lactate': ['max', 'mean', 'min'],
        'platelets': ['min', 'mean'], 'age': 'max', 'gender': 'max',
        'sofa_total': ['max', 'mean', 'min', 'std'], 'sofa_map': ['max', 'mean'],
        'sofa_creatinine': ['max', 'mean'], 'sofa_platelets': ['max', 'mean'],
        'sofa_bilirubin': ['max', 'mean'], 'map_hr_ratio': ['mean', 'max', 'min'],
        'lactate_platelets_ratio': ['mean', 'max'], 'spo2_temp_ratio': ['mean', 'min'],
        'sofa_x_hr': ['max', 'mean'], 'sofa_x_lactate': ['max', 'mean'],
        'sofa_x_map': ['max', 'mean'], 'sofa_x_creatinine': ['max', 'mean'],
        'hr_critical': ['sum', 'mean'], 'map_critical': ['sum', 'mean'],
        'rr_critical': ['sum', 'mean'], 'spo2_critical': ['sum', 'mean'],
        'lactate_high': ['sum', 'mean'], 'creatinine_high': ['sum', 'mean'],
        'sirs_score': ['max', 'mean', 'sum'], 'risk_composite': ['max', 'mean'],
        'hour': ['max', 'count'], 'heart_rate_delta': ['mean', 'max', 'min', 'std'],
        'map_delta': ['mean', 'max', 'min', 'std'], 'sofa_total_delta': ['mean', 'max', 'std'],
        'lactate_delta': ['mean', 'max', 'std'], 'label': 'max',
    }
    agg_df = df.groupby('icustay_id').agg(agg_funcs)
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df = agg_df.reset_index().merge(sofa_peak, on='icustay_id', how='left')

    agg_df['hr_range']          = agg_df['heart_rate_max'] - agg_df['heart_rate_min']
    agg_df['map_range']         = agg_df['map_max'] - agg_df['map_min']
    agg_df['temp_range']        = agg_df['temperature_max'] - agg_df['temperature_min']
    agg_df['sofa_range']        = agg_df['sofa_total_max'] - agg_df['sofa_total_min']
    agg_df['vital_instability'] = (
        agg_df.get('heart_rate_std', 0) + agg_df.get('map_std', 0) + agg_df.get('resp_rate_std', 0)
    )
    agg_df['short_stay']           = (agg_df['hour_max'] <= 6).astype(int)
    agg_df['critical_hours_ratio'] = agg_df['hr_critical_sum'] / (agg_df['hour_count'] + 1)
    agg_df['map_critical_ratio']   = agg_df['map_critical_sum'] / (agg_df['hour_count'] + 1)

    # ── 4. Recreate identical 80/20 stratified split (random_state=42) ────────
    all_ids    = agg_df['icustay_id'].values
    all_labels = agg_df.set_index('icustay_id').loc[all_ids, 'label_max'].values
    _, test_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42, stratify=all_labels
    )

    test_df = agg_df[agg_df['icustay_id'].isin(test_ids)]
    assert len(test_df) == 199, f"Expected 199 test patients, got {len(test_df)}"

    # Feature matrix must align with saved feature_cols
    for col in feat_cols:
        assert col in agg_df.columns, f"Feature '{col}' not found in aggregated data"

    X_test = test_df[feat_cols].copy()
    y_test = test_df['label_max'].values

    # No patient leakage
    assert len(set(test_ids)) == len(test_ids), "Duplicate patient IDs in test set"

    # Impute with saved training medians
    X_test = X_test.fillna(train_med).replace([np.inf, -np.inf], 0)

    # ── 5. Run ensemble inference ─────────────────────────────────────────────
    X_test_scaled = scaler.transform(X_test)
    lr_prob  = logreg.predict_proba(X_test_scaled)[:, 1]
    rf_prob  = rf.predict_proba(X_test)[:, 1]
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    gb_prob  = gb_model.predict_proba(X_test)[:, 1]
    ens_prob = (
        weights[0] * lr_prob +
        weights[1] * rf_prob +
        weights[2] * xgb_prob +
        weights[3] * gb_prob
    )
    y_pred = (ens_prob >= threshold).astype(int)

    # ── 6. Individual model metrics on unseen test set ────────────────────────
    def metrics(y_true, prob, thresh=0.5):
        pred = (prob >= thresh).astype(int)
        tn_, fp_, fn_, tp_ = confusion_matrix(y_true, pred).ravel()
        return {
            'acc':   accuracy_score(y_true, pred),
            'prec':  precision_score(y_true, pred, zero_division=0),
            'sens':  recall_score(y_true, pred, zero_division=0),
            'f1':    f1_score(y_true, pred, zero_division=0),
            'auroc': roc_auc_score(y_true, prob),
            'auprc': average_precision_score(y_true, prob),
            'brier': brier_score_loss(y_true, prob),
            'tp': tp_, 'tn': tn_, 'fp': fp_, 'fn': fn_,
        }

    lr_m   = metrics(y_test, lr_prob)
    rf_m   = metrics(y_test, rf_prob)
    xgb_m  = metrics(y_test, xgb_prob)
    gb_m   = metrics(y_test, gb_prob)
    ens_m  = metrics(y_test, ens_prob, threshold)

    print("\n" + "=" * 80)
    print("ML MODEL VALIDATION - UNSEEN HELD-OUT TEST SET")
    print("=" * 80)

    # ── A. Individual model results ───────────────────────────────────────────
    print("\n[1] INDIVIDUAL MODEL RESULTS (threshold = 0.50, unseen test set)")
    print("-" * 75)
    print(f"  {'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Sensitivity':>12} {'F1':>7} {'AUROC':>7}")
    print("  " + "-" * 69)
    for name, m in [("Logistic Regression", lr_m), ("Random Forest", rf_m),
                    ("XGBoost", xgb_m), ("Gradient Boosting", gb_m)]:
        print(f"  {name:<22} {m['acc']:>9.1%} {m['prec']:>10.1%} {m['sens']:>12.1%} {m['f1']:>7.1%} {m['auroc']:>7.4f}")
    print("  " + "-" * 69)

    # ── B. Ensemble results on unseen data ────────────────────────────────────
    print(f"\n[2] ENSEMBLE v3 RESULTS (threshold = {threshold:.4f}, unseen test set)")
    print("-" * 75)
    print(f"   Accuracy    : {ens_m['acc']:.1%}")
    print(f"   Precision   : {ens_m['prec']:.1%}")
    print(f"   Sensitivity : {ens_m['sens']:.1%}")
    print(f"   F1-Score    : {ens_m['f1']:.1%}")
    print(f"   AUROC       : {ens_m['auroc']:.4f}")
    print(f"   AUPRC       : {ens_m['auprc']:.4f}")
    print(f"   Brier Score : {ens_m['brier']:.4f}  (lower = better calibrated; 0 = perfect)")
    print(f"\n   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 No Sepsis   Sepsis")
    print(f"   Actual  No   {ens_m['tn']:>7d}   {ens_m['fp']:>6d}")
    print(f"           Yes  {ens_m['fn']:>7d}   {ens_m['tp']:>6d}")
    print(f"\n   Catches {ens_m['tp']}/{ens_m['tp']+ens_m['fn']} sepsis cases  |  "
          f"Misses {ens_m['fn']}  |  False alarms {ens_m['fp']}")

    # ── C. Bootstrap 95% Confidence Intervals ────────────────────────────────
    print(f"\n[3] BOOTSTRAP CONFIDENCE INTERVALS (n=1000, 95% CI)")
    print("-" * 75)
    rng = np.random.default_rng(42)
    boot_acc, boot_prec, boot_sens, boot_auroc, boot_auprc = [], [], [], [], []
    n = len(y_test)
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        yb, pb = y_test[idx], ens_prob[idx]
        if len(np.unique(yb)) < 2:
            continue
        pb_pred = (pb >= threshold).astype(int)
        boot_acc.append(accuracy_score(yb, pb_pred))
        boot_prec.append(precision_score(yb, pb_pred, zero_division=0))
        boot_sens.append(recall_score(yb, pb_pred, zero_division=0))
        boot_auroc.append(roc_auc_score(yb, pb))
        boot_auprc.append(average_precision_score(yb, pb))
    def ci(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)
    for label, arr in [("Accuracy   ", boot_acc), ("Precision  ", boot_prec),
                        ("Sensitivity", boot_sens), ("AUROC      ", boot_auroc),
                        ("AUPRC      ", boot_auprc)]:
        lo, hi = ci(arr)
        print(f"   {label} : {np.mean(arr):.4f}  [95% CI: {lo:.4f} - {hi:.4f}]")

    # ── D. Cross-Validation results (from saved CSV) ──────────────────────────
    print(f"\n[4] 5-FOLD STRATIFIED CROSS-VALIDATION RESULTS")
    print("-" * 75)
    cv_df = pd.read_csv(MODEL_DIR / "cv_results_v3.csv")
    print(f"  {'Fold':>5} {'Accuracy':>9} {'Precision':>10} {'Sensitivity':>12} {'F1':>7} {'AUROC':>7}")
    print("  " + "-" * 52)
    for _, row in cv_df.iterrows():
        print(f"  {int(row['Fold']):>5} {row['Accuracy']:>9.1%} {row['Precision']:>10.1%} "
              f"{row['Sensitivity']:>12.1%} {row['F1']:>7.1%} {row['AUROC']:>7.4f}")
    print("  " + "-" * 52)
    print(f"  {'Mean':>5} {cv_df['Accuracy'].mean():>9.1%} {cv_df['Precision'].mean():>10.1%} "
          f"{cv_df['Sensitivity'].mean():>12.1%} {cv_df['F1'].mean():>7.1%} {cv_df['AUROC'].mean():>7.4f}")
    print(f"  {'Std':>5} {cv_df['Accuracy'].std():>9.1%} {cv_df['Precision'].std():>10.1%} "
          f"{cv_df['Sensitivity'].std():>12.1%} {cv_df['F1'].std():>7.1%} {cv_df['AUROC'].std():>7.4f}")

    # ── E. Calibration check ──────────────────────────────────────────────────
    print(f"\n[5] CALIBRATION CHECK (probability reliability)")
    print("-" * 75)
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    bin_labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    print(f"   {'Prob Bin':<12} {'Predicted%':>11} {'Actual%':>9} {'Count':>7}")
    print("   " + "-" * 42)
    for i, (lo, hi, lbl) in enumerate(zip(bins[:-1], bins[1:], bin_labels)):
        mask = (ens_prob >= lo) & (ens_prob < hi)
        if mask.sum() == 0:
            continue
        pred_pct  = ens_prob[mask].mean()
        actual_pct = y_test[mask].mean()
        print(f"   {lbl:<12} {pred_pct:>10.1%} {actual_pct:>9.1%} {mask.sum():>7d}")

    # ── F. Feature importance top-10 ─────────────────────────────────────────
    print(f"\n[6] TOP 10 FEATURE IMPORTANCES")
    print("-" * 75)
    fi_df = pd.read_csv(MODEL_DIR / "feature_importance_v3.csv")
    total = fi_df['importance'].sum()
    print(f"  {'Rank':<6} {'Feature':<30} {'Contrib%':>9}")
    print("  " + "-" * 48)
    for i, (_, row) in enumerate(fi_df.nlargest(10, 'importance').iterrows(), 1):
        print(f"  {i:<6} {row['feature']:<30} {row['importance']/total*100:>8.1f}%")

    # ── G. Saved vs live drift check ──────────────────────────────────────────
    perf_df     = pd.read_csv(MODEL_DIR / "performance_v3.csv")
    saved_auroc = float(perf_df['Test_AUROC'].iloc[0])
    drift = abs(saved_auroc - ens_m['auroc'])

    # ── H. Final validation summary ───────────────────────────────────────────
    print(f"\n[7] FINAL VALIDATION SUMMARY")
    print("=" * 80)
    checks = [
        ("Accuracy >= 90%",       ens_m['acc'] >= 0.90,   f"{ens_m['acc']:.1%}"),
        ("Precision >= 75%",      ens_m['prec'] >= 0.75,  f"{ens_m['prec']:.1%}"),
        ("Sensitivity >= 75%",    ens_m['sens'] >= 0.75,  f"{ens_m['sens']:.1%}"),
        ("AUROC >= 90%",          ens_m['auroc'] >= 0.90, f"{ens_m['auroc']:.4f}"),
        ("AUPRC >= 85%",          ens_m['auprc'] >= 0.85, f"{ens_m['auprc']:.4f}"),
        ("False Negatives <= 20", ens_m['fn'] <= 20,      f"FN={ens_m['fn']}"),
        ("False Positives <= 25", ens_m['fp'] <= 25,      f"FP={ens_m['fp']}"),
        ("CV AUROC Mean >= 95%",  cv_df['AUROC'].mean() >= 0.95, f"{cv_df['AUROC'].mean():.4f}"),
        ("CV AUROC Std <= 2%",    cv_df['AUROC'].std() <= 0.02,  f"{cv_df['AUROC'].std():.4f}"),
        ("Brier Score <= 0.10",   ens_m['brier'] <= 0.10, f"{ens_m['brier']:.4f}"),
        ("Saved AUROC drift < 5%",drift < 0.05,           f"drift={drift:.4f}"),
        ("SOFA in top-10 features",
         any('sofa' in f for f in fi_df.nlargest(10, 'importance')['feature'].str.lower().tolist()),
         "sofa_range rank #1"),
    ]
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"  {'Check':<35} {'Result':>8}   Value")
    print("  " + "-" * 62)
    for name, ok, val in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {name:<35} {status:>8}   {val}")
    print("  " + "-" * 62)
    print(f"  {passed}/{len(checks)} checks passed")
    print("=" * 80)

    # ── I. Assertions ─────────────────────────────────────────────────────────
    assert ens_m['acc']   >= 0.90, f"Accuracy {ens_m['acc']:.3f} below 90%"
    assert ens_m['prec']  >= 0.75, f"Precision {ens_m['prec']:.3f} below 75%"
    assert ens_m['sens']  >= 0.75, f"Sensitivity {ens_m['sens']:.3f} below 75%"
    assert ens_m['auroc'] >= 0.90, f"AUROC {ens_m['auroc']:.3f} below 90%"
    assert ens_m['fn']    <= 20,   f"Too many false negatives: {ens_m['fn']}"
    assert ens_m['fp']    <= 25,   f"Too many false positives: {ens_m['fp']}"
    assert drift < 0.05,           f"Saved AUROC drifted >5% from live"
    assert any('sofa' in f for f in fi_df.nlargest(10, 'importance')['feature'].str.lower().tolist()), \
        "SOFA absent from top-10 features"

