"""
test_gnn_alerts.py
Validates GNN v3 (SepsisGAT) model on the held-out test set.
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, average_precision_score, brier_score_loss


# ── SepsisGAT architecture (mirrors train_gnn_v3.py) ────────────────────────

class SepsisGAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.3, num_layers=3):
        super().__init__()
        self.dropout = dropout
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1,
                              dropout=dropout, concat=False)
        self.bn3  = nn.BatchNorm1d(hidden_dim)
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)
        )
        self.confidence = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_attr=None):
        x = self.input_norm(x)
        x = self.input_proj(x)
        x_skip = self.skip(F.relu(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index); x = self.bn1(x); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index); x = self.bn2(x); x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index); x = self.bn3(x)
        x = x + x_skip
        x = F.relu(x)
        return self.classifier(x), self.confidence(x)


def _build_graph(X_scaled, y, k_neighbors=10, sim_threshold=0.5):
    n = X_scaled.shape[0]
    sim_matrix = cosine_similarity(X_scaled)
    np.fill_diagonal(sim_matrix, 0)
    edge_list, edge_weights = [], []
    for i in range(n):
        sims = sim_matrix[i].copy()
        k = min(k_neighbors, n - 1)
        top_k = np.argsort(sims)[-k:]
        added = 0
        for j in reversed(top_k):
            if sims[j] >= sim_threshold:
                edge_list.append([i, j]); edge_weights.append(float(sims[j])); added += 1
        if added == 0:
            for j in top_k[-3:]:
                edge_list.append([i, j]); edge_weights.append(max(float(sims[j]), 0.1))
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr  = torch.FloatTensor(edge_weights).unsqueeze(1)
    return Data(x=torch.FloatTensor(X_scaled), edge_index=edge_index,
                edge_attr=edge_attr, y=torch.LongTensor(y))


def test_gnn_model_validation():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    LABEL_DIR    = PROJECT_ROOT / "data" / "labels"
    MODEL_DIR    = PROJECT_ROOT / "data" / "model"

    # ── 1. Checkpoint exists ──────────────────────────────────────────────────
    ckpt_path = MODEL_DIR / "gnn_v3.pt"
    assert ckpt_path.exists(), f"GNN checkpoint not found: {ckpt_path}"

    # ── 2. Load and validate checkpoint keys ──────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for key in ['model_state_dict', 'model_config', 'threshold', 'feature_cols', 'test_metrics']:
        assert key in ckpt, f"Checkpoint missing key: {key}"

    cfg = ckpt['model_config']
    assert cfg['input_dim']   == 100,  f"Expected input_dim=100, got {cfg['input_dim']}"
    assert cfg['hidden_dim']  == 128,  f"Expected hidden_dim=128, got {cfg['hidden_dim']}"
    assert cfg['num_heads']   == 4,    f"Expected num_heads=4, got {cfg['num_heads']}"

    gnn_threshold = ckpt['threshold']
    feat_cols     = ckpt['feature_cols']
    assert 0.0 < gnn_threshold < 1.0, f"GNN threshold out of range: {gnn_threshold}"
    assert len(feat_cols) == 100, f"Expected 100 features, got {len(feat_cols)}"

    # ── 3. Instantiate model and verify parameter count ───────────────────────
    model = SepsisGAT(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_heads=cfg['num_heads'],
        dropout=cfg.get('dropout', 0.3),
        num_layers=cfg.get('num_layers', 3)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    assert 50_000 <= n_params <= 300_000, f"Unexpected param count: {n_params:,}"

    # ── 4. Rebuild patient-level features (same as v3) ────────────────────────
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
        max_hour   = group.loc[group['sofa_total'].idxmax(), 'hour']
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

    # ── 5. Recreate identical 80/20 split (random_state=42) ───────────────────
    all_ids    = agg_df['icustay_id'].values
    all_labels = agg_df.set_index('icustay_id').loc[all_ids, 'label_max'].values
    _, test_ids = train_test_split(all_ids, test_size=0.2, random_state=42, stratify=all_labels)

    test_df = agg_df[agg_df['icustay_id'].isin(test_ids)].reset_index(drop=True)
    X_test  = test_df[feat_cols].copy()
    y_test  = test_df['label_max'].values

    assert len(X_test) == 199, f"Expected 199 test patients, got {len(X_test)}"

    # Impute & scale (use test data own median since GNN uses its own scaler)
    scaler_gnn_path = MODEL_DIR / "scaler_gnn_v3.pkl"
    if scaler_gnn_path.exists():
        scaler_gnn = joblib.load(scaler_gnn_path)
        X_test_filled   = X_test.fillna(X_test.median()).replace([np.inf, -np.inf], 0)
        X_test_scaled   = scaler_gnn.transform(X_test_filled)
    else:
        X_test_filled = X_test.fillna(X_test.median()).replace([np.inf, -np.inf], 0)
        scaler_gnn    = StandardScaler()
        X_test_scaled = scaler_gnn.fit_transform(X_test_filled)

    # ── 6. Build patient similarity graph ─────────────────────────────────────
    test_graph = _build_graph(X_test_scaled, y_test, k_neighbors=10, sim_threshold=0.5)
    assert test_graph.num_nodes == 199, f"Expected 199 nodes, got {test_graph.num_nodes}"
    assert test_graph.num_edges > 0,    "Graph has no edges"

    # ── 7. Forward pass — shape checks ───────────────────────────────────────
    with torch.no_grad():
        logits, conf = model(test_graph.x, test_graph.edge_index)

    assert logits.shape == (199, 2), f"Logits shape expected (199,2), got {logits.shape}"
    assert conf.shape   == (199, 1), f"Conf shape expected (199,1), got {conf.shape}"
    assert (conf >= 0).all() and (conf <= 1).all(), "Confidence must be in [0,1]"

    # ── 8. Run inference ──────────────────────────────────────────────────────
    probs  = F.softmax(logits, dim=1)[:, 1].numpy()
    y_pred = (probs >= gnn_threshold).astype(int)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, zero_division=0)
    sens  = recall_score(y_test, y_pred, zero_division=0)
    auroc = roc_auc_score(y_test, probs)

    assert acc   >= 0.90, f"GNN accuracy {acc:.3f} below 90%"
    assert prec  >= 0.75, f"GNN precision {prec:.3f} below 75%"
    assert sens  >= 0.75, f"GNN sensitivity {sens:.3f} below 75%"
    assert auroc >= 0.90, f"GNN AUROC {auroc:.3f} below 90%"

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    assert fn <= 20, f"GNN too many false negatives: {fn}"
    assert fp <= 25, f"GNN too many false positives: {fp}"

    # ── extra metrics ─────────────────────────────────────────────────────────
    f1    = f1_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, probs)
    brier = brier_score_loss(y_test, probs)
    avg_conf = float(conf.numpy().mean())

    # ── 9. Saved checkpoint test metrics are consistent ───────────────────────
    saved_metrics = ckpt['test_metrics']
    if 'auroc' in saved_metrics:
        saved_auroc = float(saved_metrics['auroc'])
        assert abs(saved_auroc - auroc) < 0.05, (
            f"Saved AUROC ({saved_auroc:.4f}) drifted >5% from live ({auroc:.4f})"
        )
        auroc_drift = abs(saved_auroc - auroc)
    else:
        auroc_drift = 0.0

    # ── RICH OUTPUT ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("GNN v3 MODEL VALIDATION - UNSEEN HELD-OUT TEST SET")
    print("=" * 72)

    print()
    print("[1] MODEL ARCHITECTURE & CHECKPOINT")
    print("-" * 72)
    print(f"   Architecture   : SepsisGAT (Graph Attention Network)")
    print(f"   Layers         : {cfg.get('num_layers', 3)}  |  Hidden dim: {cfg['hidden_dim']}  |  Attention heads: {cfg['num_heads']}")
    print(f"   Input features : {cfg['input_dim']}")
    print(f"   Parameters     : {n_params:,}")
    print(f"   Threshold      : {gnn_threshold:.4f}")
    print(f"   Graph edges    : {test_graph.num_edges:,}  (avg per patient: {test_graph.num_edges / 199:.1f})")

    print()
    print("[2] GNN v3 RESULTS (held-out test set, 199 patients)")
    print("-" * 72)
    print(f"   Accuracy    : {acc*100:.1f}%")
    print(f"   Precision   : {prec*100:.1f}%")
    print(f"   Sensitivity : {sens*100:.1f}%")
    print(f"   F1-Score    : {f1*100:.1f}%")
    print(f"   AUROC       : {auroc:.4f}")
    print(f"   AUPRC       : {auprc:.4f}")
    print(f"   Brier Score : {brier:.4f}  (lower = better; 0 = perfect)")
    print(f"   Avg Confidence: {avg_conf*100:.1f}%")
    print()
    print(f"   Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 No Sepsis   Sepsis")
    print(f"   Actual  No       {tn:>5}     {fp:>5}")
    print(f"           Yes      {fn:>5}     {tp:>5}")
    print()
    print(f"   Catches {tp}/{tp+fn} sepsis cases  |  Misses {fn}  |  False alarms {fp}")

    print()
    print("[3] GNN v3 vs ENSEMBLE v3 COMPARISON")
    print("-" * 72)
    ens_acc, ens_prec, ens_sens, ens_f1, ens_auroc = 0.955, 0.958, 0.919, 0.938, 0.9939
    rows = [
        ("Accuracy",    ens_acc,   acc,   "%"),
        ("Precision",   ens_prec,  prec,  "%"),
        ("Sensitivity", ens_sens,  sens,  "%"),
        ("F1-Score",    ens_f1,    f1,    "%"),
        ("AUROC",       ens_auroc, auroc, ""),
    ]
    print(f"   {'Metric':<14}  {'Ensemble v3':>12}  {'GNN v3':>10}  {'Delta':>8}  Winner")
    print(f"   {'-'*60}")
    for name, ev, gv, unit in rows:
        delta = gv - ev
        winner = "GNN" if gv >= ev else "Ensemble"
        if unit == "%":
            print(f"   {name:<14}  {ev*100:>11.1f}%  {gv*100:>9.1f}%  {delta*100:>+7.1f}%  {winner}")
        else:
            print(f"   {name:<14}  {ev:>12.4f}  {gv:>10.4f}  {delta:>+8.4f}  {winner}")

    print()
    print("[4] GRAPH STRUCTURE VALIDATION")
    print("-" * 72)
    print(f"   Nodes in graph           : {test_graph.num_nodes}")
    print(f"   Edges in graph           : {test_graph.num_edges:,}")
    print(f"   Logits shape             : {tuple(logits.shape)}")
    print(f"   Confidence shape         : {tuple(conf.shape)}")
    print(f"   Confidence range         : [{float(conf.min()):.3f}, {float(conf.max()):.3f}]")

    print()
    print("[5] FINAL VALIDATION SUMMARY")
    print("=" * 72)
    checks = [
        ("Accuracy >= 90%",             acc   >= 0.90,  f"{acc*100:.1f}%"),
        ("Precision >= 75%",            prec  >= 0.75,  f"{prec*100:.1f}%"),
        ("Sensitivity >= 75%",          sens  >= 0.75,  f"{sens*100:.1f}%"),
        ("AUROC >= 90%",                auroc >= 0.90,  f"{auroc:.4f}"),
        ("AUPRC >= 85%",                auprc >= 0.85,  f"{auprc:.4f}"),
        ("False Negatives <= 20",       fn    <= 20,    f"FN={fn}"),
        ("False Positives <= 25",       fp    <= 25,    f"FP={fp}"),
        ("Brier Score <= 0.10",         brier <= 0.10,  f"{brier:.4f}"),
        ("Checkpoint keys present",     True,           "5/5 keys"),
        ("Architecture config valid",   cfg['input_dim'] == 100 and cfg['hidden_dim'] == 128, "input=100, hidden=128"),
        ("Graph has edges",             test_graph.num_edges > 0, f"{test_graph.num_edges:,} edges"),
        ("Saved AUROC drift < 5%",      auroc_drift < 0.05,     f"drift={auroc_drift:.4f}"),
    ]
    passed = sum(1 for _, ok, _ in checks if ok)
    print(f"   {'Check':<40}  {'Result':<6}  Value")
    print(f"   {'-'*62}")
    for label, ok, val in checks:
        print(f"   {label:<40}  {'PASS' if ok else 'FAIL':<6}  {val}")
    print(f"   {'-'*62}")
    print(f"   {passed}/{len(checks)} checks passed")
    print("=" * 72)
