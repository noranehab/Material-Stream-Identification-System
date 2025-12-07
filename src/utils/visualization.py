"""
Visualization Utilities

Functions for visualizing results, confusion matrices, and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Dict, List
import yaml


class Visualization:
    """
    Visualization utilities for model evaluation and analysis.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize visualization utilities.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.classes = self.config['classes']
        self.class_names = [self.classes[i] for i in sorted(self.classes.keys())]
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot (optional)
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(
        self,
        labels: List[int],
        title: str = "Class Distribution",
        save_path: str = None
    ) -> None:
        """
        Plot class distribution.
        
        Args:
            labels: List of class labels
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        unique, counts = np.unique(labels, return_counts=True)
        class_counts = [counts[i] if i in unique else 0 for i in range(len(self.class_names))]
        
        plt.figure(figsize=(10, 6))
        plt.bar(self.class_names, class_counts)
        plt.title(title)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> None:
        """
        Print detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
        """
        report = classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            output_dict=False
        )
        print(report)
    
    def plot_training_comparison(
        self,
        svm_results: Dict,
        knn_results: Dict,
        save_path: str = None
    ) -> None:
        """
        Plot comparison between SVM and k-NN results.
        
        Args:
            svm_results: Dictionary with SVM evaluation results
            knn_results: Dictionary with k-NN evaluation results
            save_path: Path to save the plot (optional)
        """
        models = ['SVM', 'k-NN']
        accuracies = [svm_results['accuracy'], knn_results['accuracy']]
        
        plt.figure(figsize=(8, 6))
        plt.bar(models, accuracies)
        plt.title('Model Comparison')
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.grid(axis='y', alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

