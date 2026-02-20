"""
Graph Neural Network for Patient Similarity and Sepsis Risk Prediction.

Uses message passing to propagate information from similar patients,
enabling risk prediction even for patients with missing data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from typing import Tuple, Dict
import numpy as np


class PatientSimilarityGNN(nn.Module):
    """
    Graph Attention Network for patient similarity-based sepsis prediction.
    
    Uses attention mechanisms to weigh information from similar patients,
    particularly useful when dealing with missing data.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Dimension of hidden layers
            num_layers: Number of graph convolution layers
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(PatientSimilarityGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Graph attention layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                in_channels = hidden_dim
            else:
                in_channels = hidden_dim * num_heads
            
            self.conv_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True if i < num_layers - 1 else False
                )
            )
            
            out_dim = hidden_dim * num_heads if i < num_layers - 1 else hidden_dim
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        # Prediction head
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 2)  # Binary classification
        
        # Confidence estimation head
        self.confidence_head = nn.Linear(hidden_dim, 1)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes/weights [num_edges, 1]
            return_embeddings: Whether to return node embeddings
            
        Returns:
            logits: Class logits [num_nodes, 2]
            confidence: Prediction confidence [num_nodes, 1]
            embeddings (optional): Node embeddings [num_nodes, hidden_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph convolution layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            
            if i < self.num_layers - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            x = x_new
        
        # Store embeddings
        embeddings = x
        
        # Classification head
        x_class = F.relu(self.fc1(x))
        x_class = F.dropout(x_class, p=self.dropout, training=self.training)
        logits = self.fc2(x_class)
        
        # Confidence estimation
        confidence = torch.sigmoid(self.confidence_head(embeddings))
        
        if return_embeddings:
            return logits, confidence, embeddings
        else:
            return logits, confidence
    
    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict sepsis probabilities and confidence scores.
        
        Returns:
            probabilities: Sepsis probabilities [num_nodes]
            confidence: Confidence scores [num_nodes]
        """
        self.eval()
        with torch.no_grad():
            logits, conf = self.forward(x, edge_index, edge_attr)
            probs = F.softmax(logits, dim=1)
            sepsis_probs = probs[:, 1].cpu().numpy()
            confidence = conf.squeeze().cpu().numpy()
        
        return sepsis_probs, confidence


class GNNTrainer:
    """
    Trainer for the Patient Similarity GNN.
    """
    
    def __init__(
        self,
        model: PatientSimilarityGNN,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = None,
        class_weights: torch.Tensor = None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GNN model to train
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on ('cuda' or 'cpu')
            class_weights: Optional class weights for imbalanced data
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        # Learning rate scheduler for better convergence
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, data: Data) -> float:
        """
        Train for one epoch.
        
        Args:
            data: Graph data object
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        data = data.to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        logits, confidence = self.model(data.x, data.edge_index, data.edge_attr)
        
        # Classification loss with optional class weighting
        if self.class_weights is not None:
            loss_cls = F.cross_entropy(logits, data.y, weight=self.class_weights)
        else:
            loss_cls = F.cross_entropy(logits, data.y)
        
        # Confidence calibration loss (MSE between confidence and correctness)
        predictions = logits.argmax(dim=1)
        correctness = (predictions == data.y).float().unsqueeze(1)
        loss_conf = F.mse_loss(confidence, correctness)
        
        # Total loss
        loss = loss_cls + 0.1 * loss_conf
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: Data) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            data: Graph data object
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            logits, confidence = self.model(data.x, data.edge_index, data.edge_attr)
            
            # Classification loss
            loss = F.cross_entropy(logits, data.y)
            
            # Accuracy
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == data.y).float().mean()
            
            # Sepsis-specific metrics
            true_labels = data.y.cpu().numpy()
            pred_labels = predictions.cpu().numpy()
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            # True positives, false positives, false negatives
            tp = ((pred_labels == 1) & (true_labels == 1)).sum()
            fp = ((pred_labels == 1) & (true_labels == 0)).sum()
            fn = ((pred_labels == 0) & (true_labels == 1)).sum()
            
            # Precision, Recall, F1
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_confidence': confidence.mean().item()
        }
        
        return metrics
    
    def train(
        self,
        train_data: Data,
        val_data: Data = None,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Train the model with optional early stopping.
        
        Args:
            train_data: Training graph data
            val_data: Validation graph data
            epochs: Number of training epochs
            early_stopping_patience: Epochs to wait before early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history
        """
        best_val_loss = float('inf')
        patience_counter = 0
        self.best_model_state = self.model.state_dict()  # Initialize
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_data)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_data is not None:
                val_metrics = self.evaluate(val_data)
                history['val_loss'].append(val_metrics['loss'])
                history['val_metrics'].append(val_metrics)
                
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val F1: {val_metrics['f1']:.4f}")
                
                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
                    
                # Update learning rate based on validation loss
                self.scheduler.step(val_metrics['loss'])
            else:
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
        
        return history
