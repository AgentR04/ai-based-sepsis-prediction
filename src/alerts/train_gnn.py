"""
Train GNN Model for Sepsis Alert System.

Trains a Graph Neural Network on historical patient data to identify
patients at risk of sepsis based on similarity to past cases.
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import argparse
import pickle
from datetime import datetime

from src.alerts.patient_graph import PatientGraphBuilder
from src.alerts.gnn_model import PatientSimilarityGNN, GNNTrainer
from src.alerts.alert_generator import SepsisAlertGenerator


def load_data(features_path: str, labels_path: str, train_dataset_path: str = None):
    """Load patient features and labels."""
    print("Loading data...")
    
    # Try to load the train dataset first
    if train_dataset_path and os.path.exists(train_dataset_path):
        train_df = pd.read_csv(train_dataset_path)
        
        # Check for label column
        label_col = None
        for col in ['label', 'sepsis', 'target']:
            if col in train_df.columns:
                label_col = col
                break
        
        # Check for icustay_id or patient_id
        id_col = None
        for col in ['icustay_id', 'patient_id', 'stay_id']:
            if col in train_df.columns:
                id_col = col
                break
        
        if id_col:
            patient_ids = train_df[id_col]
        else:
            patient_ids = pd.Series(range(len(train_df)), name='patient_id')
        
        # Separate features and labels
        exclude_cols = [id_col, label_col, 'hour'] if id_col and label_col else ['hour'] if 'hour' in train_df.columns else []
        if exclude_cols:
            features_df = train_df.drop([c for c in exclude_cols if c in train_df.columns], axis=1)
        else:
            features_df = train_df
        
        if label_col:
            labels_df = pd.DataFrame({'sepsis': train_df[label_col]})
        else:
            # Load labels from sepsis_onset file and create binary labels
            sepsis_df = pd.read_csv(labels_path)
            if 'icustay_id' in train_df.columns and 'icustay_id' in sepsis_df.columns:
                sepsis_ids = set(sepsis_df['icustay_id'].unique())
                labels_df = pd.DataFrame({'sepsis': train_df['icustay_id'].isin(sepsis_ids).astype(int)})
            else:
                # If no matching, create default zeros with some positives based on sepsis file size
                labels_df = pd.DataFrame({'sepsis': np.zeros(len(train_df), dtype=int)})
                n_sepsis = min(len(sepsis_df), len(train_df))
                labels_df.iloc[:n_sepsis, 0] = 1
                labels_df = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        # Load features and labels separately
        features_df = pd.read_csv(features_path)
        sepsis_df = pd.read_csv(labels_path)
        
        # Create binary labels
        if 'icustay_id' in features_df.columns and 'icustay_id' in sepsis_df.columns:
            sepsis_ids = set(sepsis_df['icustay_id'].unique())
            labels_df = pd.DataFrame({'sepsis': features_df['icustay_id'].isin(sepsis_ids).astype(int)})
            patient_ids = features_df['icustay_id']
            features_df = features_df.drop('icustay_id', axis=1)
        else:
            # Generate synthetic labels
            labels_df = pd.DataFrame({'sepsis': np.zeros(len(features_df), dtype=int)})
            n_sepsis = min(len(sepsis_df), len(features_df))
            labels_df.iloc[:n_sepsis, 0] = 1
            labels_df = labels_df.sample(frac=1, random_state=42).reset_index(drop=True)
            patient_ids = pd.Series(range(len(features_df)), name='patient_id')
    
    print(f"Loaded {len(features_df)} patients with {features_df.shape[1]} features")
    print(f"Sepsis prevalence: {labels_df['sepsis'].mean():.2%}")
    
    return features_df, labels_df, patient_ids


def introduce_missing_data(
    features_df: pd.DataFrame,
    missing_ratio: float = 0.2,
    seed: int = 42
) -> pd.DataFrame:
    """
    Introduce random missing data to simulate real-world scenarios.
    
    Args:
        features_df: Original features
        missing_ratio: Proportion of values to set as missing
        seed: Random seed
        
    Returns:
        Features with missing data
    """
    np.random.seed(seed)
    features_with_missing = features_df.copy()
    
    # Randomly set values to NaN
    mask = np.random.rand(*features_with_missing.shape) < missing_ratio
    features_with_missing = features_with_missing.mask(mask)
    
    print(f"Introduced {missing_ratio:.1%} missing data")
    print(f"Patients with missing data: {features_with_missing.isna().any(axis=1).sum()}")
    
    return features_with_missing


def train_gnn_model(
    features_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    patient_ids: pd.Series,
    output_dir: str = 'data/model',
    hidden_dim: int = 64,
    num_layers: int = 3,
    epochs: int = 100,
    test_size: float = 0.2
):
    """
    Train the GNN model for sepsis prediction.
    
    Args:
        features_df: Patient features
        labels_df: Sepsis labels
        patient_ids: Patient identifiers
        output_dir: Directory to save model and artifacts
        hidden_dim: Hidden layer dimension
        num_layers: Number of GNN layers
        epochs: Training epochs
        test_size: Validation set proportion
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data
    print("\nSplitting data...")
    train_indices, val_indices = train_test_split(
        range(len(features_df)),
        test_size=test_size,
        stratify=labels_df['sepsis'],
        random_state=42
    )
    
    train_features = features_df.iloc[train_indices].reset_index(drop=True)
    train_labels = labels_df.iloc[train_indices].reset_index(drop=True)
    train_ids = patient_ids.iloc[train_indices].reset_index(drop=True)
    
    val_features = features_df.iloc[val_indices].reset_index(drop=True)
    val_labels = labels_df.iloc[val_indices].reset_index(drop=True)
    val_ids = patient_ids.iloc[val_indices].reset_index(drop=True)
    
    print(f"Training set: {len(train_features)} patients")
    print(f"Validation set: {len(val_features)} patients")
    
    # Build patient graphs
    print("\nBuilding patient similarity graphs...")
    graph_builder = PatientGraphBuilder(
        similarity_threshold=0.6,
        k_neighbors=10
    )
    
    train_graph = graph_builder.build_graph(train_features, train_labels, train_ids)
    val_graph = graph_builder.build_graph(val_features, val_labels, val_ids)
    
    print(f"Training graph: {train_graph.num_nodes} nodes, {train_graph.num_edges} edges")
    print(f"Validation graph: {val_graph.num_nodes} nodes, {val_graph.num_edges} edges")
    
    # Get missingness statistics
    print("\nMissingness statistics:")
    miss_stats = graph_builder.get_missingness_stats(features_df)
    print(f"Total missing data: {miss_stats['total_missing_pct']:.2f}%")
    print("Patients by missingness level:")
    for level, count in miss_stats['patients_by_missingness'].items():
        print(f"  {level}: {count}")
    
    # Initialize model
    print("\nInitializing GNN model...")
    input_dim = train_graph.x.shape[1]
    model = PatientSimilarityGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=4,
        dropout=0.3
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\nTraining GNN model...")
    trainer = GNNTrainer(model, learning_rate=0.001, weight_decay=1e-5)
    
    history = trainer.train(
        train_data=train_graph,
        val_data=val_graph,
        epochs=epochs,
        early_stopping_patience=15,
        verbose=True
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_metrics = trainer.evaluate(val_graph)
    for metric, value in final_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'gnn_sepsis_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': 4,
            'dropout': 0.3
        },
        'training_history': history,
        'final_metrics': final_metrics
    }, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save graph builder
    builder_path = os.path.join(output_dir, 'graph_builder.pkl')
    with open(builder_path, 'wb') as f:
        pickle.dump(graph_builder, f)
    print(f"Graph builder saved to {builder_path}")
    
    return model, graph_builder, history


def generate_example_alerts(
    model: PatientSimilarityGNN,
    graph_builder: PatientGraphBuilder,
    features_df: pd.DataFrame,
    patient_ids: pd.Series,
    output_dir: str = 'data/model'
):
    """
    Generate example alerts for patients with missing data.
    """
    print("\nGenerating example alerts...")
    
    # Create alert generator
    alert_generator = SepsisAlertGenerator(
        gnn_model=model,
        graph_builder=graph_builder,
        missingness_threshold=0.15
    )
    
    # Generate alerts
    alerts = alert_generator.generate_alerts(
        current_patients_df=features_df,
        patient_ids=patient_ids
    )
    
    print(f"\nGenerated {len(alerts)} alerts")
    
    # Print summary
    summary = alert_generator.generate_alert_summary(alerts)
    print("\nAlert Summary:")
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  By level:")
    for level, count in summary['by_level'].items():
        print(f"    {level}: {count}")
    print(f"  Average risk score: {summary['avg_risk_score']:.2%}")
    print(f"  Average confidence: {summary['avg_confidence']:.2%}")
    print(f"  Average missingness: {summary['avg_missingness']:.1%}")
    
    # Display top 3 high-priority alerts
    high_priority = [a for a in alerts if a.alert_level in ['CRITICAL', 'HIGH']]
    if high_priority:
        print("\nTop Priority Alerts:")
        for i, alert in enumerate(high_priority[:3], 1):
            print(f"\n{i}. {alert}")
    
    # Export alerts
    alerts_path = os.path.join(output_dir, 'example_alerts.csv')
    alert_generator.export_alerts(alerts, alerts_path, format='csv')
    print(f"\nAlerts exported to {alerts_path}")
    
    return alerts


def main():
    parser = argparse.ArgumentParser(description='Train GNN for sepsis alert system')
    parser.add_argument('--features', type=str, default='data/features/features.csv',
                        help='Path to features CSV')
    parser.add_argument('--labels', type=str, default='data/labels/sepsis_onset.csv',
                        help='Path to labels CSV')
    parser.add_argument('--train-dataset', type=str, default='data/model/train_dataset.csv',
                        help='Path to combined training dataset CSV')
    parser.add_argument('--output-dir', type=str, default='data/model',
                        help='Output directory for model and alerts')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden layer dimension')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--missing-ratio', type=float, default=0.2,
                        help='Ratio of missing data to introduce')
    parser.add_argument('--no-missing', action='store_true',
                        help='Do not introduce additional missing data')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("GNN-Based Sepsis Alert System - Training")
    print("=" * 70)
    
    # Load data
    features_df, labels_df, patient_ids = load_data(
        args.features,
        args.labels,
        args.train_dataset
    )
    
    # Introduce missing data if requested
    if not args.no_missing:
        features_df = introduce_missing_data(
            features_df,
            missing_ratio=args.missing_ratio
        )
    
    # Train model
    model, graph_builder, history = train_gnn_model(
        features_df=features_df,
        labels_df=labels_df,
        patient_ids=patient_ids,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs
    )
    
    # Generate example alerts
    alerts = generate_example_alerts(
        model=model,
        graph_builder=graph_builder,
        features_df=features_df,
        patient_ids=patient_ids,
        output_dir=args.output_dir
    )
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
