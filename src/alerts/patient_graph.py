"""
Patient Graph Construction Module.

Builds a graph representation of patients where edges connect similar patients
based on available clinical features, handling missing data appropriately.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch_geometric.data import Data


class PatientGraphBuilder:
    """
    Builds a patient similarity graph for GNN-based alert generation.
    """
    
    def __init__(self, similarity_threshold: float = 0.7, k_neighbors: int = 10):
        """
        Initialize the graph builder.
        
        Args:
            similarity_threshold: Minimum similarity to create an edge
            k_neighbors: Number of nearest neighbors to connect per patient
        """
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def build_graph(
        self, 
        features_df: pd.DataFrame, 
        labels_df: pd.DataFrame,
        patient_ids: pd.Series = None
    ) -> Data:
        """
        Build a patient similarity graph.
        
        Args:
            features_df: Patient features (may contain missing values)
            labels_df: Sepsis labels
            patient_ids: Optional patient identifiers
            
        Returns:
            PyTorch Geometric Data object representing the patient graph
        """
        # Store feature columns
        self.feature_columns = features_df.columns.tolist()
        
        # Create missingness indicators (binary features showing which values are missing)
        missingness_indicators = features_df.isna().astype(float)
        missingness_indicators.columns = [f"{col}_missing" for col in missingness_indicators.columns]
        
        # Impute missing values with median (for similarity computation)
        # Use 0 as fallback when median is NaN (column is all missing)
        medians = features_df.median()
        medians = medians.fillna(0)
        features_imputed = features_df.fillna(medians)
        
        # Normalize features (replace any remaining NaN/inf with 0)
        features_normalized = self.scaler.fit_transform(features_imputed)
        features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Combine normalized features with missingness indicators
        combined_features = np.hstack([
            features_normalized,
            missingness_indicators.values
        ])
        
        # Compute pairwise similarity matrix
        similarity_matrix = self._compute_similarity(combined_features, missingness_indicators.values)
        
        # Build edge list based on k-nearest neighbors
        edge_index, edge_weights = self._build_edges(similarity_matrix)
        
        # Prepare node features (include missingness info)
        node_features = torch.FloatTensor(combined_features)
        
        # Prepare labels
        if 'sepsis' in labels_df.columns:
            labels = torch.LongTensor(labels_df['sepsis'].values)
        else:
            labels = torch.LongTensor(labels_df.values)
        
        # Calculate missingness ratio for each patient
        missingness_ratio = missingness_indicators.mean(axis=1).values
        
        # Create graph data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights,
            y=labels,
            missingness_ratio=torch.FloatTensor(missingness_ratio)
        )
        
        # Store patient IDs if provided
        if patient_ids is not None:
            graph_data.patient_ids = patient_ids.values
        
        return graph_data
    
    def _compute_similarity(
        self, 
        features: np.ndarray, 
        missingness: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise similarity between patients using vectorized operations.
        
        Combines feature similarity and missingness pattern similarity.
        Optimized with batch processing to avoid memory overflow.
        """
        n_patients = features.shape[0]
        n_features = features.shape[1] // 2  # Half are features, half are missingness indicators
        
        print(f"  Computing similarities for {n_patients} patients...")
        
        # Extract feature matrix (first half of columns)
        feat_matrix = features[:, :n_features]
        
        # Normalize features for cosine similarity (batch operation)
        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True) + 1e-8
        feat_normalized = feat_matrix / norms
        
        # Compute cosine similarity matrix (vectorized matrix multiplication)
        print("  - Computing feature similarity...")
        feat_similarity = np.dot(feat_normalized, feat_normalized.T)
        
        # Compute missingness pattern similarity (memory-efficient batched approach)
        print("  - Computing missingness similarity...")
        # Process in batches to avoid memory overflow
        batch_size = 1000
        missing_similarity = np.zeros((n_patients, n_patients), dtype=np.float32)
        
        for i in range(0, n_patients, batch_size):
            i_end = min(i + batch_size, n_patients)
            batch = missingness[i:i_end]
            
            # Compute pairwise difference for this batch
            for j in range(0, n_patients, batch_size):
                j_end = min(j + batch_size, n_patients)
                batch_j = missingness[j:j_end]
                
                # Compute mean absolute difference
                diff = np.abs(batch[:, np.newaxis, :] - batch_j[np.newaxis, :, :])
                missing_similarity[i:i_end, j:j_end] = 1 - diff.mean(axis=2)
        
        # Combined similarity (weighted average)
        print("  - Combining similarities...")
        similarity_matrix = 0.7 * feat_similarity + 0.3 * missing_similarity
        
        # Clip to valid range
        similarity_matrix = np.clip(similarity_matrix, 0, 1)
        
        print(f"  Similarity matrix computed: {similarity_matrix.shape}")
        
        return similarity_matrix
    
    def _build_edges(
        self, 
        similarity_matrix: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edge list and weights from similarity matrix.
        
        Connects each patient to their k most similar neighbors.
        Uses threshold as a soft filter - ensures at least some edges are created.
        """
        n_patients = similarity_matrix.shape[0]
        edge_list = []
        weight_list = []
        
        for i in range(n_patients):
            # Get k most similar patients (excluding self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self-loops
            
            # Find k nearest neighbors
            k = min(self.k_neighbors, n_patients - 1)
            top_k_indices = np.argsort(similarities)[-k:]
            
            # Connect to top k neighbors above threshold
            # If none are above threshold, connect to top neighbors anyway (with lower weight)
            edges_added = 0
            for j in reversed(top_k_indices):  # Start from most similar
                sim = similarities[j]
                if sim >= self.similarity_threshold:
                    edge_list.append([i, j])
                    weight_list.append(sim)
                    edges_added += 1
            
            # If no edges added due to threshold, add at least top 2 neighbors
            if edges_added == 0 and k >= 2:
                for j in top_k_indices[-2:]:
                    sim = max(similarities[j], 0.1)  # Ensure positive weight
                    edge_list.append([i, j])
                    weight_list.append(sim)
        
        # Convert to PyTorch tensors
        if len(edge_list) > 0:
            edge_index = torch.LongTensor(edge_list).t().contiguous()
            edge_weights = torch.FloatTensor(weight_list).unsqueeze(1)
        else:
            # Create empty edge structure (shouldn't happen with fallback logic)
            edge_index = torch.LongTensor([[], []])
            edge_weights = torch.FloatTensor([])
        
        return edge_index, edge_weights
    
    def get_missingness_stats(self, features_df: pd.DataFrame) -> Dict:
        """
        Calculate statistics about missing data in the dataset.
        """
        missingness = features_df.isna()
        
        stats = {
            'total_missing_pct': missingness.sum().sum() / (features_df.shape[0] * features_df.shape[1]) * 100,
            'patients_with_missing_pct': (missingness.any(axis=1).sum() / len(features_df)) * 100,
            'features_missing_by_column': (missingness.sum() / len(features_df) * 100).to_dict(),
            'patients_by_missingness': {
                'low (<10%)': (missingness.mean(axis=1) < 0.1).sum(),
                'moderate (10-30%)': ((missingness.mean(axis=1) >= 0.1) & (missingness.mean(axis=1) < 0.3)).sum(),
                'high (30-50%)': ((missingness.mean(axis=1) >= 0.3) & (missingness.mean(axis=1) < 0.5)).sum(),
                'severe (>50%)': (missingness.mean(axis=1) >= 0.5).sum()
            }
        }
        
        return stats
