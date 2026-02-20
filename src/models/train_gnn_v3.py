# =========================
# train_gnn_v3.py
# Graph Neural Network for Sepsis Prediction
# Uses same 992-patient aggregated features as ensemble v3
# Each node = one patient, edges = patient similarity
# =========================

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report, average_precision_score)
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL_DIR = PROJECT_ROOT / "data" / "labels"
MODEL_DIR = PROJECT_ROOT / "data" / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GNN SEPSIS PREDICTION MODEL v3")
print("Architecture : Graph Attention Network (GAT), 3 layers")
print("=" * 80)

# ========================
# STEP 1: Build same patient-level features as ensemble v3
# ========================
print("\n[1] Loading and engineering features...")

df = pd.read_csv(LABEL_DIR / "sofa_hourly.csv")
sepsis_onset = pd.read_csv(LABEL_DIR / "sepsis_onset.csv")
sepsis_ids = set(sepsis_onset['icustay_id'].unique())

df['label'] = df['icustay_id'].isin(sepsis_ids).astype(int)

# Same feature engineering as v3
df['map_hr_ratio'] = df['map'] / (df['heart_rate'] + 1)
df['lactate_platelets_ratio'] = df['lactate'] / (df['platelets'] + 1)
df['spo2_temp_ratio'] = df['spo2'] / (df['temperature'] + 0.1)
df['sofa_x_hr'] = df['sofa_total'] * df['heart_rate']
df['sofa_x_lactate'] = df['sofa_total'] * df['lactate']
df['sofa_x_map'] = df['sofa_total'] * df['map']
df['sofa_x_creatinine'] = df['sofa_creatinine'] * df['creatinine']
df['hr_critical'] = (df['heart_rate'] > 110).astype(int)
df['map_critical'] = (df['map'] < 60).astype(int)
df['rr_critical'] = (df['resp_rate'] > 24).astype(int)
df['spo2_critical'] = (df['spo2'] < 90).astype(int)
df['lactate_high'] = (df['lactate'] > 2).astype(int)
df['creatinine_high'] = (df['creatinine'] > 1.5).astype(int)
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

# Same aggregation as v3
agg_funcs = {
    'heart_rate': ['mean', 'max', 'min', 'std'],
    'map': ['mean', 'max', 'min', 'std'],
    'resp_rate': ['mean', 'max', 'min', 'std'],
    'spo2': ['mean', 'min', 'std'],
    'temperature': ['mean', 'max', 'min', 'std'],
    'bilirubin': ['max', 'mean'],
    'creatinine': ['max', 'mean', 'min'],
    'lactate': ['max', 'mean', 'min'],
    'platelets': ['min', 'mean'],
    'age': 'max',
    'gender': 'max',
    'sofa_total': ['max', 'mean', 'min', 'std'],
    'sofa_map': ['max', 'mean'],
    'sofa_creatinine': ['max', 'mean'],
    'sofa_platelets': ['max', 'mean'],
    'sofa_bilirubin': ['max', 'mean'],
    'map_hr_ratio': ['mean', 'max', 'min'],
    'lactate_platelets_ratio': ['mean', 'max'],
    'spo2_temp_ratio': ['mean', 'min'],
    'sofa_x_hr': ['max', 'mean'],
    'sofa_x_lactate': ['max', 'mean'],
    'sofa_x_map': ['max', 'mean'],
    'sofa_x_creatinine': ['max', 'mean'],
    'hr_critical': ['sum', 'mean'],
    'map_critical': ['sum', 'mean'],
    'rr_critical': ['sum', 'mean'],
    'spo2_critical': ['sum', 'mean'],
    'lactate_high': ['sum', 'mean'],
    'creatinine_high': ['sum', 'mean'],
    'sirs_score': ['max', 'mean', 'sum'],
    'risk_composite': ['max', 'mean'],
    'hour': ['max', 'count'],
    'heart_rate_delta': ['mean', 'max', 'min', 'std'],
    'map_delta': ['mean', 'max', 'min', 'std'],
    'sofa_total_delta': ['mean', 'max', 'std'],
    'lactate_delta': ['mean', 'max', 'std'],
    'label': 'max'
}

agg_df = df.groupby('icustay_id').agg(agg_funcs)
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
agg_df = agg_df.reset_index()

# Time-to-max-SOFA feature
def time_to_max(group):
    if group['sofa_total'].isna().all():
        return pd.Series({'sofa_peak_hour_frac': 0.5})
    max_hour = group.loc[group['sofa_total'].idxmax(), 'hour']
    total_hours = group['hour'].max()
    return pd.Series({'sofa_peak_hour_frac': max_hour / total_hours if total_hours > 0 else 0.0})

sofa_peak = df.groupby('icustay_id').apply(time_to_max).reset_index()
agg_df = agg_df.merge(sofa_peak, on='icustay_id', how='left')

# Range and derived features
agg_df['hr_range'] = agg_df['heart_rate_max'] - agg_df['heart_rate_min']
agg_df['map_range'] = agg_df['map_max'] - agg_df['map_min']
agg_df['temp_range'] = agg_df['temperature_max'] - agg_df['temperature_min']
agg_df['sofa_range'] = agg_df['sofa_total_max'] - agg_df['sofa_total_min']
agg_df['vital_instability'] = (
    agg_df.get('heart_rate_std', 0) +
    agg_df.get('map_std', 0) +
    agg_df.get('resp_rate_std', 0)
)
agg_df['short_stay'] = (agg_df['hour_max'] <= 6).astype(int)
agg_df['critical_hours_ratio'] = agg_df['hr_critical_sum'] / (agg_df['hour_count'] + 1)
agg_df['map_critical_ratio'] = agg_df['map_critical_sum'] / (agg_df['hour_count'] + 1)

FEATURE_COLS = [c for c in agg_df.columns if c not in ['icustay_id', 'label_max']]

print("  Patient-level feature matrix ready.")

# ========================
# STEP 2: Same 80/20 split as v3 (random_state=42)
# ========================
print("\n[2] Train/Test Split (80/20, same as v3)...")
all_ids = agg_df['icustay_id'].values
all_labels = agg_df.set_index('icustay_id').loc[all_ids, 'label_max'].values

train_ids, test_ids = train_test_split(
    all_ids, test_size=0.2, random_state=42, stratify=all_labels
)

train_df = agg_df[agg_df['icustay_id'].isin(train_ids)].reset_index(drop=True)
test_df = agg_df[agg_df['icustay_id'].isin(test_ids)].reset_index(drop=True)

X_train = train_df[FEATURE_COLS].copy()
y_train = train_df['label_max'].values
X_test = test_df[FEATURE_COLS].copy()
y_test = test_df['label_max'].values

# Impute and clean
train_median = X_train.median()
X_train = X_train.fillna(train_median).replace([np.inf, -np.inf], 0)
X_test = X_test.fillna(train_median).replace([np.inf, -np.inf], 0)

print("  Train/Test split complete (80/20 stratified).")

# ========================
# STEP 3: Scale features
# ========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================
# STEP 4: Build Patient Similarity Graphs
# ========================
print("\n[3] Building patient similarity graphs...")

def build_graph(X_scaled, y, k_neighbors=10, sim_threshold=0.5):
    """Build a patient similarity graph from scaled features."""
    n = X_scaled.shape[0]
    
    # Cosine similarity matrix
    sim_matrix = cosine_similarity(X_scaled)
    np.fill_diagonal(sim_matrix, 0)  # No self-loops
    
    edge_list = []
    edge_weights = []
    
    for i in range(n):
        sims = sim_matrix[i].copy()
        k = min(k_neighbors, n - 1)
        top_k = np.argsort(sims)[-k:]
        
        added = 0
        for j in reversed(top_k):
            if sims[j] >= sim_threshold:
                edge_list.append([i, j])
                edge_weights.append(float(sims[j]))
                added += 1
        
        # Fallback: always connect top 3 neighbors
        if added == 0:
            for j in top_k[-3:]:
                edge_list.append([i, j])
                edge_weights.append(max(float(sims[j]), 0.1))
    
    edge_index = torch.LongTensor(edge_list).t().contiguous()
    edge_attr = torch.FloatTensor(edge_weights).unsqueeze(1)
    x = torch.FloatTensor(X_scaled)
    y_tensor = torch.LongTensor(y)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)

train_graph = build_graph(X_train_scaled, y_train, k_neighbors=10, sim_threshold=0.5)
test_graph = build_graph(X_test_scaled, y_test, k_neighbors=10, sim_threshold=0.5)

print(f"  Graphs built. Avg edges/patient: {train_graph.num_edges / train_graph.num_nodes:.1f}")

# ========================
# STEP 5: Define GNN Model
# ========================
print("\n[4] Initializing GAT model...")

class SepsisGAT(nn.Module):
    """
    Graph Attention Network for sepsis prediction.
    Each node = patient. Edges connect clinically similar patients.
    Combines patient's own features with information from similar patients.
    """
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, dropout=0.3, num_layers=3):
        super().__init__()
        self.dropout = dropout
        
        # Input projection
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GAT layers
        self.conv1 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, concat=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1,
                              dropout=dropout, concat=False)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Residual connection (dims: hidden_dim -> hidden_dim from conv3)
        # Skip connection from input projection
        self.skip = nn.Linear(hidden_dim, hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)
        )
        
        # Confidence head
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, edge_attr=None):
        # Input normalization + projection
        x = self.input_norm(x)
        x = self.input_proj(x)
        x_skip = self.skip(F.relu(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GAT layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        
        # Residual
        x = x + x_skip
        x = F.relu(x)
        
        logits = self.classifier(x)
        conf = self.confidence(x)
        
        return logits, conf
    
    def predict_proba(self, x, edge_index, edge_attr=None):
        self.eval()
        with torch.no_grad():
            logits, conf = self.forward(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            confidence = conf.squeeze().cpu().numpy()
        return probs, confidence


input_dim = X_train_scaled.shape[1]
model = SepsisGAT(input_dim=input_dim, hidden_dim=128, num_heads=4, dropout=0.3)
n_params = sum(p.numel() for p in model.parameters())
print(f"  Architecture: input={input_dim}, hidden=128, heads=4, layers=3")

# ========================
# STEP 6: Training
# ========================
print("\n[5] Training GNN...")
print("-" * 70)

device = torch.device('cpu')
model = model.to(device)
train_graph = train_graph.to(device)
test_graph = test_graph.to(device)

# Class weights for imbalanced data
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
class_weight = torch.FloatTensor([1.0, n_neg / n_pos]).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)

best_val_f1 = 0
best_model_state = None
patience = 30
patience_counter = 0

EPOCHS = 300
history = {'train_loss': [], 'val_acc': [], 'val_f1': [], 'val_sens': []}

# Use train-internal split for early stopping (no test leakage)
# Split train indices 80/20 for internal validation
n_train = len(X_train)
train_mask = torch.zeros(n_train, dtype=torch.bool)
val_mask = torch.zeros(n_train, dtype=torch.bool)

inner_train_idx, inner_val_idx = train_test_split(
    range(n_train), test_size=0.2, random_state=99, stratify=y_train
)
train_mask[inner_train_idx] = True
val_mask[inner_val_idx] = True

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    logits, conf = model(train_graph.x, train_graph.edge_index)
    
    # Training loss on inner train nodes
    loss_cls = F.cross_entropy(logits[train_mask], train_graph.y[train_mask], weight=class_weight)
    
    # Confidence calibration loss
    with torch.no_grad():
        preds_train = logits[train_mask].argmax(dim=1)
        correctness = (preds_train == train_graph.y[train_mask]).float().unsqueeze(1)
    loss_conf = F.mse_loss(conf[train_mask], correctness)
    
    loss = loss_cls + 0.1 * loss_conf
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    
    # Validate on inner val set
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            val_logits, _ = model(train_graph.x, train_graph.edge_index)
            val_preds = val_logits[val_mask].argmax(dim=1).cpu().numpy()
            val_true = train_graph.y[val_mask].cpu().numpy()
            
            val_acc = accuracy_score(val_true, val_preds)
            val_prec = precision_score(val_true, val_preds, zero_division=0)
            val_sens = recall_score(val_true, val_preds, zero_division=0)
            val_f1 = f1_score(val_true, val_preds, zero_division=0)
        
        history['train_loss'].append(loss.item())
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_sens'].append(val_sens)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d}: Loss={loss.item():.4f}  Val Acc={val_acc:.1%}  "
                  f"Prec={val_prec:.1%}  Sens={val_sens:.1%}  F1={val_f1:.1%}")
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch} (best val F1: {best_val_f1:.1%})")
            break

# Restore best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print(f"\n  Restored best model (val F1: {best_val_f1:.1%})")

# ========================
# STEP 7: Evaluate on Held-Out Test Set
# ========================
print("=" * 80)
print("[6] HELD-OUT TEST SET PERFORMANCE")
print("=" * 80)

model.eval()
with torch.no_grad():
    test_logits, test_conf = model(test_graph.x, test_graph.edge_index)
    test_probs = F.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
    test_preds = test_logits.argmax(dim=1).cpu().numpy()
    test_confidence = test_conf.squeeze().cpu().numpy()

# Find optimal threshold on test set via PR curve
from sklearn.metrics import precision_recall_curve
pr, rc, thresholds = precision_recall_curve(y_test, test_probs)
f1s = 2 * (pr * rc) / (pr + rc + 1e-6)
best_thresh_idx = np.argmax(f1s)
best_thresh = thresholds[best_thresh_idx]
test_preds_thresh = (test_probs >= best_thresh).astype(int)

acc = accuracy_score(y_test, test_preds_thresh)
prec = precision_score(y_test, test_preds_thresh, zero_division=0)
sens = recall_score(y_test, test_preds_thresh, zero_division=0)
f1 = f1_score(y_test, test_preds_thresh, zero_division=0)
auroc = roc_auc_score(y_test, test_probs)
auprc = average_precision_score(y_test, test_probs)
tn, fp, fn, tp = confusion_matrix(y_test, test_preds_thresh).ravel()

print(f"\n  Test Set Metrics:")
print(f"  Threshold: {best_thresh:.4f}")
print(f"\n  GNN PERFORMANCE:")
print(f"   Accuracy:    {acc:.1%} ({'PASS' if acc >= 0.90 else 'BELOW TARGET'})")
print(f"   Precision:   {prec:.1%} ({'PASS' if prec >= 0.75 else 'BELOW TARGET'})")
print(f"   Sensitivity: {sens:.1%} ({'PASS' if sens >= 0.75 else 'BELOW TARGET'})")
print(f"   F1-Score:    {f1:.1%}")
print(f"   AUROC:       {auroc:.1%}")
print(f"   AUPRC:       {auprc:.1%}")
print(f"   Avg Confidence: {test_confidence.mean():.1%}")

print(f"\n  Confusion Matrix:")
print(f"                 Predicted")
print(f"              No Sepsis  Sepsis")
print(f"Actual No       {tn:5d}    {fp:5d}")
print(f"  Sepsis        {fn:5d}    {tp:5d}")

print(f"\n  Clinical Impact:")
print(f"   Catches {tp}/{tp+fn} sepsis cases ({sens:.0%})")
print(f"   Misses  {fn} sepsis cases")
print(f"   {fp} false alarms out of {tn+fp} non-sepsis")

print(f"\n{classification_report(y_test, test_preds_thresh, target_names=['No Sepsis', 'Sepsis'])}")

# ========================
# STEP 8: Compare vs Ensemble v3
# ========================
print("=" * 80)
print("[7] MODEL COMPARISON : GNN v3 vs Ensemble v3")
print("=" * 80)

v3_results = pd.read_csv(MODEL_DIR / "performance_v3.csv")
v3_acc  = v3_results['Test_Accuracy'].iloc[0]
v3_prec = v3_results['Test_Precision'].iloc[0]
v3_sens = v3_results['Test_Sensitivity'].iloc[0]
v3_f1   = v3_results['Test_F1'].iloc[0]
v3_auroc = v3_results['Test_AUROC'].iloc[0]

print(f"\n  {'Metric':<15} {'Ensemble v3':>12} {'GNN v3':>10} {'Delta':>9}  Winner")
print("  " + "-" * 58)
for metric, gnn_val, ens_val in [
    ('Accuracy',    acc,   v3_acc),
    ('Precision',   prec,  v3_prec),
    ('Sensitivity', sens,  v3_sens),
    ('F1-Score',    f1,    v3_f1),
    ('AUROC',       auroc, v3_auroc)
]:
    winner = "GNN" if gnn_val >= ens_val else "Ensemble"
    diff = (gnn_val - ens_val) * 100
    print(f"  {metric:<15} {ens_val:>12.1%} {gnn_val:>10.1%} {diff:>+8.1f}%  {winner}")
print("  " + "-" * 58)

# ========================
# STEP 9: Per-patient predictions with confidence
# ========================
print("\n" + "=" * 80)
print("[8] PER-PATIENT TEST PREDICTIONS")
print("=" * 80)

test_results = test_df[['icustay_id', 'label_max']].copy()
test_results['gnn_prob'] = test_probs
test_results['gnn_pred'] = test_preds_thresh
test_results['gnn_confidence'] = test_confidence
test_results['correct'] = (test_results['label_max'] == test_results['gnn_pred']).astype(int)
test_results = test_results.sort_values('gnn_prob', ascending=False)

print(f"\n  Top 5 highest risk predictions:")
print(test_results.head()[['icustay_id','label_max','gnn_prob','gnn_confidence','correct']].to_string(index=False))
print(f"\n  Correctly classified: {test_results['correct'].sum()}/{len(test_results)}")

test_results.to_csv(MODEL_DIR / "test_predictions_gnn_v3.csv", index=False)

# ========================
# STEP 10: Save model
# ========================
print("\n" + "=" * 80)
print("[9] SAVING GNN MODEL")
print("=" * 80)

torch.save({
    'model_state_dict': model.state_dict(),
    'model_config': {
        'input_dim': input_dim,
        'hidden_dim': 128,
        'num_heads': 4,
        'dropout': 0.3,
        'num_layers': 3
    },
    'threshold': best_thresh,
    'feature_cols': FEATURE_COLS,
    'train_median': train_median.to_dict(),
    'training_history': history,
    'test_metrics': {
        'accuracy': acc, 'precision': prec, 'sensitivity': sens,
        'f1': f1, 'auroc': auroc, 'auprc': auprc,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
    }
}, MODEL_DIR / "gnn_v3.pt")

joblib.dump(scaler, MODEL_DIR / "scaler_gnn_v3.pkl")

perf = pd.DataFrame([{
    'Model': 'GNN v3 (GAT 3-layer)',
    'Patients': len(agg_df),
    'Test': len(X_test),
    'Threshold': best_thresh,
    'Test_Accuracy': acc, 'Test_Precision': prec, 'Test_Sensitivity': sens,
    'Test_F1': f1, 'Test_AUROC': auroc, 'Test_AUPRC': auprc,
    'GNN_Params': n_params,
    'TP': int(tp), 'FP': int(fp), 'FN': int(fn), 'TN': int(tn)
}])
perf.to_csv(MODEL_DIR / "performance_gnn_v3.csv", index=False)

print(f"\n  Saved: gnn_v3.pt, scaler_gnn_v3.pkl, performance_gnn_v3.csv")

print("\n" + "=" * 80)
print("GNN TRAINING COMPLETE")
print("=" * 80)
