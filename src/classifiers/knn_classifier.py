"""
k-NN Classifier Implementation

k-Nearest Neighbors classifier with rejection mechanism for uncertain predictions.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import yaml
from typing import Tuple, Optional
import os


class KNNClassifier:
    """
    k-NN classifier with distance-based rejection mechanism.
    
    Features:
    - Configurable number of neighbors
    - Weighted voting (uniform or distance-based)
    - Distance-based rejection mechanism
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize k-NN classifier.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        knn_config = self.config['knn']
        
        self.model = KNeighborsClassifier(
            n_neighbors=knn_config['n_neighbors'],
            weights=knn_config['weights'],
            algorithm=knn_config['algorithm'],
            metric=knn_config['metric'],
            n_jobs=self.config['training']['n_jobs']
        )
        
        self.scaler = StandardScaler()
        self.rejection_threshold = knn_config['rejection_threshold']
        self.classes = self.config['classes']
        self.is_trained = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the k-NN classifier.
        
        Args:
            X_train: Training feature vectors (n_samples, n_features)
            y_train: Training labels (n_samples,)
        """
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print("Training k-NN classifier...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        print("k-NN training completed.")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature vectors (n_samples, n_features)
            
        Returns:
            Predicted class labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_rejection(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with rejection mechanism based on neighbor distances.
        
        Args:
            X: Feature vectors (n_samples, n_features)
            
        Returns:
            predictions: Predicted class labels (uncertain predictions -> class 6)
            confidences: Confidence scores based on neighbor distances
            rejected: Boolean array indicating rejected predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction.")
        
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities from neighbors
        probabilities = self.model.predict_proba(X_scaled)
        max_probs = np.max(probabilities, axis=1)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        # Calculate average distance to k nearest neighbors as confidence measure
        distances, indices = self.model.kneighbors(X_scaled)
        avg_distances = np.mean(distances, axis=1)
        
        # Normalize distances to [0, 1] range (inverse relationship: closer = higher confidence)
        max_dist = np.max(avg_distances) if np.max(avg_distances) > 0 else 1.0
        distance_confidences = 1.0 - (avg_distances / max_dist)
        
        # Combine probability and distance confidence
        combined_confidences = (max_probs + distance_confidences) / 2.0
        
        # Apply rejection mechanism
        rejected = combined_confidences < self.rejection_threshold
        predictions = predicted_classes.copy()
        predictions[rejected] = 6  # Assign to Unknown class
        
        return predictions, combined_confidences, rejected
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_rejection: bool = True
    ) -> dict:
        """
        Evaluate classifier performance.
        
        Args:
            X: Feature vectors (n_samples, n_features)
            y: True labels (n_samples,)
            use_rejection: Whether to use rejection mechanism
            
        Returns:
            Dictionary with evaluation metrics
        """
        if use_rejection:
            predictions, confidences, rejected = self.predict_with_rejection(X)
        else:
            predictions = self.predict(X)
            confidences = None
            rejected = None
        
        accuracy = np.mean(predictions == y)
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions,
            'confidences': confidences,
            'rejected': rejected
        }
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model and scaler separately
        joblib.dump(self.model, f"{filepath}_model.pkl")
        joblib.dump(self.scaler, f"{filepath}_scaler.pkl")
        print(f"Model saved to {filepath}_model.pkl and {filepath}_scaler.pkl")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model and scaler.
        
        Args:
            filepath: Path to load the model from (without extension)
        """
        self.model = joblib.load(f"{filepath}_model.pkl")
        self.scaler = joblib.load(f"{filepath}_scaler.pkl")
        self.is_trained = True
        print(f"Model loaded from {filepath}_model.pkl and {filepath}_scaler.pkl")
    
    def get_info(self) -> dict:
        """
        Get information about the classifier configuration.
        
        Returns:
            Dictionary with classifier information
        """
        return {
            'type': 'k-NN',
            'n_neighbors': self.model.n_neighbors,
            'weights': self.model.weights,
            'algorithm': self.model.algorithm,
            'metric': self.model.metric,
            'rejection_threshold': self.rejection_threshold,
            'is_trained': self.is_trained
        }
