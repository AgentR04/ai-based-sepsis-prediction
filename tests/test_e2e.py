"""
test_e2e.py
End-to-end pipeline validation: data integrity, feature engineering,
Ensemble v3 and GNN v3 performance, cross-model consistency.
One test function, no classes, no sub-tests.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              roc_auc_score, confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


# ── SepsisGAT (copied from train_gnn_v3.py) ─────────────────────────────────

class _SepsisGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.3, num_layers=3):
        super().__init__()
        self.dropout   = dropout
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
        self.bn1   = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
        self.bn2   = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout, concat=False)
        self.bn3   = nn.BatchNorm1d(hidden_dim)
        self.skip  = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)
        )
        self.confidence = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr=None):
        x      = self.input_norm(x)
        x      = self.input_proj(x)
        x_skip = self.skip(F.relu(x))
        x      = F.relu(x)
        x      = F.dropout(x, p=self.dropout, training=self.training)
        x      = self.conv1(x, edge_index); x = self.bn1(x); x = F.relu(x)
        x      = F.dropout(x, p=self.dropout, training=self.training)
        x      = self.conv2(x, edge_index); x = self.bn2(x); x = F.relu(x)
        x      = F.dropout(x, p=self.dropout, training=self.training)
        x      = self.conv3(x, edge_index); x = self.bn3(x)
        x      = x + x_skip
        x      = F.relu(x)
        return self.classifier(x), self.confidence(x)


def _build_graph(X_scaled, y, k_neighbors=10, sim_threshold=0.5):
    n = X_scaled.shape[0]
    sim_matrix = cosine_similarity(X_scaled)
    np.fill_diagonal(sim_matrix, 0)
    edges, weights = [], []
    for i in range(n):
        sims  = sim_matrix[i].copy()
        top_k = np.argsort(sims)[-min(k_neighbors, n - 1):]
        added = 0
        for j in reversed(top_k):
            if sims[j] >= sim_threshold:
                edges.append([i, j]); weights.append(float(sims[j])); added += 1
        if added == 0:
            for j in top_k[-3:]:
                edges.append([i, j]); weights.append(max(float(sims[j]), 0.1))
    return Data(
        x=torch.FloatTensor(X_scaled),
        edge_index=torch.LongTensor(edges).t().contiguous(),
        edge_attr=torch.FloatTensor(weights).unsqueeze(1),
        y=torch.LongTensor(y)
    )


def _build_features(label_dir):
    """Build patient-level aggregated features (identical to v3 training)."""
    df           = pd.read_csv(label_dir / "sofa_hourly.csv")
    sepsis_onset = pd.read_csv(label_dir / "sepsis_onset.csv")
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

    return agg_df, sepsis_ids


def test_end_to_end_pipeline():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    LABEL_DIR    = PROJECT_ROOT / "data" / "labels"
    MODEL_DIR    = PROJECT_ROOT / "data" / "model"
    RAW_DIR      = PROJECT_ROOT / "data" / "raw"

    # ── Stage 1: All key data files exist and are non-empty ───────────────────
    data_files = {
        "sofa_hourly.csv":   LABEL_DIR / "sofa_hourly.csv",
        "sepsis_onset.csv":  LABEL_DIR / "sepsis_onset.csv",
        "cohort.csv":        RAW_DIR   / "cohort.csv",
        "vitals_hourly.csv": RAW_DIR   / "vitals_hourly.csv",
        "labs_hourly.csv":   RAW_DIR   / "labs_hourly.csv",
    }
    for name, path in data_files.items():
        assert path.exists(), f"Missing data file: {name}"
        assert path.stat().st_size > 0, f"Empty data file: {name}"

    # ── Stage 2: Cohort integrity ─────────────────────────────────────────────
    cohort = pd.read_csv(RAW_DIR / "cohort.csv")
    assert cohort['icustay_id'].nunique() >= 900,     "Cohort should have ≥900 patients"
    assert cohort['icustay_id'].duplicated().sum() == 0, "Cohort has duplicate icustay_id"
    if 'age' in cohort.columns:
        assert cohort['age'].min() >= 18,  "Cohort has patients younger than 18"
        # MIMIC de-identifies ages >89 by setting them to 200-300; allow up to 300
        assert cohort['age'].max() <= 300, "Cohort has implausible ages >300"

    # ── Stage 3: SOFA pipeline output ────────────────────────────────────────
    sofa_df     = pd.read_csv(LABEL_DIR / "sofa_hourly.csv")
    sepsis_df   = pd.read_csv(LABEL_DIR / "sepsis_onset.csv")
    sepsis_ids  = set(sepsis_df['icustay_id'].unique())
    n_patients  = sofa_df['icustay_id'].nunique()

    assert n_patients >= 992, f"Expected ≥992 patients in sofa_hourly.csv, got {n_patients}"
    assert sofa_df['sofa_total'].isna().sum() == 0,   "sofa_total has NaN"
    assert sofa_df['sofa_total'].min() >= 0,           "sofa_total has negative values"

    sofa_df['is_sepsis'] = sofa_df['icustay_id'].isin(sepsis_ids)
    avg_s  = sofa_df[sofa_df['is_sepsis']]['sofa_total'].mean()
    avg_ns = sofa_df[~sofa_df['is_sepsis']]['sofa_total'].mean()
    assert avg_s > avg_ns, f"Sepsis SOFA avg ({avg_s:.2f}) ≤ non-sepsis ({avg_ns:.2f})"

    # Early-onset patients included (onset hour ≤ 2 should be in sofa_hourly)
    early_onset = sepsis_df[sepsis_df.get('onset_hour', sepsis_df.iloc[:, 1]) <= 2]['icustay_id'] \
        if 'onset_hour' in sepsis_df.columns else pd.Series(dtype=int)
    if len(early_onset) > 0:
        missing_early = set(early_onset) - set(sofa_df['icustay_id'].unique())
        assert len(missing_early) == 0, f"{len(missing_early)} early-onset patients missing from sofa_hourly"

    # ── Stage 4: Feature engineering ─────────────────────────────────────────
    agg_df, _ = _build_features(LABEL_DIR)
    feat_cols  = [c for c in agg_df.columns if c not in ['icustay_id', 'label_max']]

    assert agg_df['icustay_id'].nunique() == 992, \
        f"Expected 992 patients post-aggregation, got {agg_df['icustay_id'].nunique()}"
    assert len(feat_cols) == 100, f"Expected 100 features, got {len(feat_cols)}"

    all_ids    = agg_df['icustay_id'].values
    all_labels = agg_df.set_index('icustay_id').loc[all_ids, 'label_max'].values
    train_ids, test_ids = train_test_split(
        all_ids, test_size=0.2, random_state=42, stratify=all_labels
    )
    assert len(train_ids) == 793, f"Train split: expected 793, got {len(train_ids)}"
    assert len(test_ids)  == 199, f"Test split: expected 199, got {len(test_ids)}"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test patient overlap detected"

    # ── Stage 5: Ensemble v3 performance ─────────────────────────────────────
    feat_cols_ens = joblib.load(MODEL_DIR / "feature_cols_v3.pkl")
    train_med     = joblib.load(MODEL_DIR / "train_median_v3.pkl")
    scaler_ens    = joblib.load(MODEL_DIR / "scaler_v3.pkl")
    logreg        = joblib.load(MODEL_DIR / "logreg_v3.pkl")
    rf            = joblib.load(MODEL_DIR / "rf_v3.pkl")
    xgb_model     = joblib.load(MODEL_DIR / "xgb_v3.pkl")
    gb_model      = joblib.load(MODEL_DIR / "gb_v3.pkl")
    weights       = joblib.load(MODEL_DIR / "ensemble_weights_v3.pkl")
    threshold_ens = joblib.load(MODEL_DIR / "ensemble_threshold_v3.pkl")

    test_df  = agg_df[agg_df['icustay_id'].isin(test_ids)]
    X_test_e = test_df[feat_cols_ens].fillna(train_med).replace([np.inf, -np.inf], 0)
    y_test   = test_df['label_max'].values

    X_test_scaled = scaler_ens.transform(X_test_e)
    ens_prob = (
        weights[0] * logreg.predict_proba(X_test_scaled)[:, 1] +
        weights[1] * rf.predict_proba(X_test_e)[:, 1] +
        weights[2] * xgb_model.predict_proba(X_test_e)[:, 1] +
        weights[3] * gb_model.predict_proba(X_test_e)[:, 1]
    )
    ens_pred  = (ens_prob >= threshold_ens).astype(int)
    ens_acc   = accuracy_score(y_test, ens_pred)
    ens_prec  = precision_score(y_test, ens_pred, zero_division=0)
    ens_sens  = recall_score(y_test, ens_pred, zero_division=0)
    ens_auroc = roc_auc_score(y_test, ens_prob)

    assert ens_acc   >= 0.90, f"Ensemble acc {ens_acc:.3f} < 90%"
    assert ens_prec  >= 0.75, f"Ensemble prec {ens_prec:.3f} < 75%"
    assert ens_sens  >= 0.75, f"Ensemble sens {ens_sens:.3f} < 75%"
    assert ens_auroc >= 0.95, f"Ensemble AUROC {ens_auroc:.3f} < 95%"

    # ── Stage 6: GNN v3 performance ───────────────────────────────────────────
    ckpt       = torch.load(MODEL_DIR / "gnn_v3.pt", map_location='cpu', weights_only=False)
    feat_cols_gnn = ckpt['feature_cols']
    gnn_thresh    = ckpt['threshold']
    cfg           = ckpt['model_config']

    gnn_model = _SepsisGAT(
        input_dim=cfg['input_dim'], hidden_dim=cfg['hidden_dim'],
        num_heads=cfg['num_heads'], dropout=cfg.get('dropout', 0.3)
    )
    gnn_model.load_state_dict(ckpt['model_state_dict'])
    gnn_model.eval()

    X_test_g = test_df[feat_cols_gnn].copy()
    scaler_gnn_path = MODEL_DIR / "scaler_gnn_v3.pkl"
    if scaler_gnn_path.exists():
        scaler_gnn   = joblib.load(scaler_gnn_path)
        X_test_g_sc  = scaler_gnn.transform(X_test_g.fillna(X_test_g.median()).replace([np.inf, -np.inf], 0))
    else:
        sc           = StandardScaler()
        X_test_g_sc  = sc.fit_transform(X_test_g.fillna(X_test_g.median()).replace([np.inf, -np.inf], 0))

    test_graph = _build_graph(X_test_g_sc, y_test)
    with torch.no_grad():
        gnn_logits, _ = gnn_model(test_graph.x, test_graph.edge_index)
    gnn_prob  = F.softmax(gnn_logits, dim=1)[:, 1].numpy()
    gnn_pred  = (gnn_prob >= gnn_thresh).astype(int)
    gnn_acc   = accuracy_score(y_test, gnn_pred)
    gnn_prec  = precision_score(y_test, gnn_pred, zero_division=0)
    gnn_sens  = recall_score(y_test, gnn_pred, zero_division=0)
    gnn_auroc = roc_auc_score(y_test, gnn_prob)

    assert gnn_acc   >= 0.90, f"GNN acc {gnn_acc:.3f} < 90%"
    assert gnn_prec  >= 0.75, f"GNN prec {gnn_prec:.3f} < 75%"
    assert gnn_sens  >= 0.75, f"GNN sens {gnn_sens:.3f} < 75%"
    assert gnn_auroc >= 0.90, f"GNN AUROC {gnn_auroc:.3f} < 90%"

    # ── Stage 7: Cross-model consistency on high-confidence cases ─────────────
    high_conf_mask = (ens_prob > 0.80) | (ens_prob < 0.20)
    if high_conf_mask.sum() >= 10:
        ens_hc  = ens_pred[high_conf_mask]
        gnn_hc  = gnn_pred[high_conf_mask]
        agreement_rate = (ens_hc == gnn_hc).mean()
        assert agreement_rate >= 0.70, (
            f"Ensemble/GNN agreement on high-confidence cases: {agreement_rate:.2f} < 70%"
        )

    # ── Stage 8: All output CSVs exist and saved metrics match live ───────────
    output_csvs = [
        "performance_v3.csv", "test_predictions_v3.csv", "feature_importance_v3.csv",
        "performance_gnn_v3.csv", "test_predictions_gnn_v3.csv",
    ]
    for fname in output_csvs:
        assert (MODEL_DIR / fname).exists(), f"Missing output CSV: {fname}"

    perf_ens = pd.read_csv(MODEL_DIR / "performance_v3.csv")
    saved_ens_auroc = float(perf_ens['Test_AUROC'].iloc[0])
    assert abs(saved_ens_auroc - ens_auroc) < 0.05, (
        f"Saved ensemble AUROC ({saved_ens_auroc:.4f}) drifted >5% from live ({ens_auroc:.4f})"
    )

    fi_df  = pd.read_csv(MODEL_DIR / "feature_importance_v3.csv")
    top5   = fi_df.nlargest(5, 'importance')['feature'].str.lower().tolist()
    assert any('sofa' in f for f in top5), f"SOFA not in top-5 features: {top5}"

    print(f"[PASS] E2E pipeline — {n_patients} patients, 100 features, 793/199 split")
    print(f"       Ensemble: acc={ens_acc:.3f} prec={ens_prec:.3f} "
          f"sens={ens_sens:.3f} auroc={ens_auroc:.3f}")
    print(f"       GNN:      acc={gnn_acc:.3f} prec={gnn_prec:.3f} "
          f"sens={gnn_sens:.3f} auroc={gnn_auroc:.3f}")
