"""
Sepsis Alert Generation System.

Generates early warning alerts for patients with missing data based on
similarity to historical patients who developed sepsis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import torch
from datetime import datetime

from .patient_graph import PatientGraphBuilder
from .gnn_model import PatientSimilarityGNN


class SepsisAlert:
    """
    Represents a sepsis early warning alert for a patient.
    """
    
    def __init__(
        self,
        patient_id: str,
        risk_score: float,
        confidence: float,
        missingness_ratio: float,
        similar_patients: List[Dict],
        alert_level: str,
        timestamp: datetime = None
    ):
        """
        Initialize a sepsis alert.
        
        Args:
            patient_id: Patient identifier
            risk_score: Predicted sepsis risk (0-1)
            confidence: Model confidence in prediction (0-1)
            missingness_ratio: Proportion of missing data (0-1)
            similar_patients: List of similar historical patients
            alert_level: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'
            timestamp: Alert generation time
        """
        self.patient_id = patient_id
        self.risk_score = risk_score
        self.confidence = confidence
        self.missingness_ratio = missingness_ratio
        self.similar_patients = similar_patients
        self.alert_level = alert_level
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary format."""
        return {
            'patient_id': self.patient_id,
            'risk_score': float(self.risk_score),
            'confidence': float(self.confidence),
            'missingness_ratio': float(self.missingness_ratio),
            'alert_level': self.alert_level,
            'similar_patients_count': len(self.similar_patients),
            'similar_patients': self.similar_patients,
            'timestamp': self.timestamp.isoformat(),
            'recommendation': self._get_recommendation()
        }
    
    def _get_recommendation(self) -> str:
        """Generate clinical recommendation based on alert level."""
        if self.alert_level == 'CRITICAL':
            return (
                "URGENT: High sepsis risk detected with significant missing data. "
                "Immediate clinical assessment recommended. Consider empiric treatment "
                "while completing diagnostic workup."
            )
        elif self.alert_level == 'HIGH':
            return (
                "High sepsis risk based on similarity to historical positive cases. "
                "Recommend close monitoring, complete missing labs/vitals, "
                "and assess for sepsis criteria."
            )
        elif self.alert_level == 'MEDIUM':
            return (
                "Moderate sepsis risk identified. Monitor patient closely and "
                "prioritize completion of missing clinical data for better assessment."
            )
        else:  # LOW
            return (
                "Low sepsis risk. Continue routine monitoring and data collection."
            )
    
    def __str__(self) -> str:
        """String representation of the alert."""
        return (
            f"SEPSIS ALERT [{self.alert_level}]\n"
            f"Patient: {self.patient_id}\n"
            f"Risk Score: {self.risk_score:.2%}\n"
            f"Confidence: {self.confidence:.2%}\n"
            f"Missing Data: {self.missingness_ratio:.1%}\n"
            f"Similar Cases: {len(self.similar_patients)}\n"
            f"Recommendation: {self._get_recommendation()}"
        )


class SepsisAlertGenerator:
    """
    Generates sepsis alerts for patients with missing data using GNN-based
    similarity matching with historical patients.
    """
    
    def __init__(
        self,
        gnn_model: PatientSimilarityGNN,
        graph_builder: PatientGraphBuilder,
        missingness_threshold: float = 0.15,
        risk_thresholds: Dict[str, float] = None
    ):
        """
        Initialize the alert generator.
        
        Args:
            gnn_model: Trained GNN model
            graph_builder: Patient graph builder
            missingness_threshold: Minimum missing data ratio to trigger alerts
            risk_thresholds: Risk score thresholds for alert levels
        """
        self.gnn_model = gnn_model
        self.graph_builder = graph_builder
        self.missingness_threshold = missingness_threshold
        
        # Default risk thresholds
        self.risk_thresholds = risk_thresholds or {
            'CRITICAL': 0.75,
            'HIGH': 0.60,
            'MEDIUM': 0.45,
            'LOW': 0.0
        }
        
        self.device = next(gnn_model.parameters()).device
    
    def generate_alerts(
        self,
        current_patients_df: pd.DataFrame,
        patient_ids: pd.Series = None,
        return_all: bool = False
    ) -> List[SepsisAlert]:
        """
        Generate alerts for patients with missing data.
        
        Args:
            current_patients_df: Current patient features (may have missing data)
            patient_ids: Patient identifiers
            return_all: If True, return alerts for all patients; if False, only
                       return alerts for patients above missingness threshold
                       
        Returns:
            List of SepsisAlert objects
        """
        # Calculate missingness for each patient
        missingness_ratios = current_patients_df.isna().mean(axis=1).values
        
        # Get patients with significant missing data
        if not return_all:
            missing_data_mask = missingness_ratios >= self.missingness_threshold
            if not missing_data_mask.any():
                return []
        else:
            missing_data_mask = np.ones(len(current_patients_df), dtype=bool)
        
        # Create dummy labels for prediction (we don't know the true labels yet)
        dummy_labels = pd.DataFrame({'sepsis': np.zeros(len(current_patients_df))})
        
        # Build patient graph
        graph_data = self.graph_builder.build_graph(
            current_patients_df,
            dummy_labels,
            patient_ids
        )
        
        # Move to device
        graph_data = graph_data.to(self.device)
        
        # Get predictions and embeddings
        self.gnn_model.eval()
        with torch.no_grad():
            logits, confidence, embeddings = self.gnn_model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_attr,
                return_embeddings=True
            )
            
            # Get sepsis probabilities
            probs = torch.softmax(logits, dim=1)
            risk_scores = probs[:, 1].cpu().numpy()
            confidence_scores = confidence.squeeze().cpu().numpy()
        
        # Generate alerts
        alerts = []
        
        for idx in np.where(missing_data_mask)[0]:
            patient_id = patient_ids.iloc[idx] if patient_ids is not None else f"Patient_{idx}"
            risk_score = risk_scores[idx]
            conf_score = confidence_scores[idx]
            miss_ratio = missingness_ratios[idx]
            
            # Find similar patients
            similar_patients = self._find_similar_patients(
                idx,
                embeddings,
                graph_data,
                top_k=5
            )
            
            # Determine alert level
            alert_level = self._determine_alert_level(risk_score, miss_ratio)
            
            # Create alert
            alert = SepsisAlert(
                patient_id=patient_id,
                risk_score=risk_score,
                confidence=conf_score,
                missingness_ratio=miss_ratio,
                similar_patients=similar_patients,
                alert_level=alert_level
            )
            
            alerts.append(alert)
        
        # Sort by risk score (highest first)
        alerts.sort(key=lambda a: a.risk_score, reverse=True)
        
        return alerts
    
    def _find_similar_patients(
        self,
        patient_idx: int,
        embeddings: torch.Tensor,
        graph_data,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find most similar historical patients.
        
        Args:
            patient_idx: Index of current patient
            embeddings: Patient embeddings from GNN
            graph_data: Graph data object
            top_k: Number of similar patients to return
            
        Returns:
            List of similar patient information
        """
        # Get patient embedding
        patient_emb = embeddings[patient_idx].unsqueeze(0)
        
        # Compute similarities to all other patients
        similarities = torch.cosine_similarity(
            patient_emb,
            embeddings,
            dim=1
        ).cpu().numpy()
        
        # Exclude self
        similarities[patient_idx] = -1
        
        # Get top-k most similar
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        similar_patients = []
        for sim_idx in top_k_indices:
            if similarities[sim_idx] > 0:
                patient_info = {
                    'similarity': float(similarities[sim_idx]),
                    'had_sepsis': bool(graph_data.y[sim_idx].item()),
                    'missingness': float(graph_data.missingness_ratio[sim_idx].item())
                }
                
                # Add patient ID if available
                if hasattr(graph_data, 'patient_ids'):
                    patient_info['patient_id'] = str(graph_data.patient_ids[sim_idx])
                
                similar_patients.append(patient_info)
        
        return similar_patients
    
    def _determine_alert_level(
        self,
        risk_score: float,
        missingness_ratio: float
    ) -> str:
        """
        Determine alert level based on risk score and missingness.
        
        Higher missingness with high risk score escalates the alert level.
        """
        # Base level from risk score
        if risk_score >= self.risk_thresholds['CRITICAL']:
            base_level = 'CRITICAL'
        elif risk_score >= self.risk_thresholds['HIGH']:
            base_level = 'HIGH'
        elif risk_score >= self.risk_thresholds['MEDIUM']:
            base_level = 'MEDIUM'
        else:
            base_level = 'LOW'
        
        # Escalate if high missingness with elevated risk
        if missingness_ratio > 0.4 and risk_score > 0.5:
            if base_level == 'HIGH':
                return 'CRITICAL'
            elif base_level == 'MEDIUM':
                return 'HIGH'
        
        return base_level
    
    def generate_alert_summary(self, alerts: List[SepsisAlert]) -> Dict:
        """
        Generate summary statistics for a list of alerts.
        
        Args:
            alerts: List of SepsisAlert objects
            
        Returns:
            Summary statistics dictionary
        """
        if not alerts:
            return {
                'total_alerts': 0,
                'by_level': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
                'avg_risk_score': 0.0,
                'avg_confidence': 0.0,
                'avg_missingness': 0.0
            }
        
        # Count by level
        level_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for alert in alerts:
            level_counts[alert.alert_level] += 1
        
        # Calculate averages
        avg_risk = np.mean([a.risk_score for a in alerts])
        avg_conf = np.mean([a.confidence for a in alerts])
        avg_miss = np.mean([a.missingness_ratio for a in alerts])
        
        summary = {
            'total_alerts': len(alerts),
            'by_level': level_counts,
            'avg_risk_score': float(avg_risk),
            'avg_confidence': float(avg_conf),
            'avg_missingness': float(avg_miss),
            'high_priority_count': level_counts['CRITICAL'] + level_counts['HIGH']
        }
        
        return summary
    
    def export_alerts(
        self,
        alerts: List[SepsisAlert],
        output_path: str,
        format: str = 'csv'
    ):
        """
        Export alerts to file.
        
        Args:
            alerts: List of alerts to export
            output_path: Path to save file
            format: 'csv' or 'json'
        """
        if format == 'csv':
            # Convert to DataFrame
            alert_dicts = [a.to_dict() for a in alerts]
            
            # Flatten similar patients info
            for alert_dict in alert_dicts:
                sepsis_count = sum(
                    1 for p in alert_dict['similar_patients'] if p['had_sepsis']
                )
                alert_dict['similar_sepsis_count'] = sepsis_count
                del alert_dict['similar_patients']  # Remove nested structure for CSV
            
            df = pd.DataFrame(alert_dicts)
            df.to_csv(output_path, index=False)
            
        elif format == 'json':
            import json
            alert_dicts = [a.to_dict() for a in alerts]
            with open(output_path, 'w') as f:
                json.dump(alert_dicts, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
