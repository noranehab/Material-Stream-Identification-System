"""
Evaluation Module

Comprehensive evaluation of trained models with detailed metrics.
"""

import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.classifiers.svm_classifier import SVMClassifier
from src.classifiers.knn_classifier import KNNClassifier
from src.feature_extraction import FeatureExtractor
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_models(config_path: str = "config/config.yaml"):
    """
    Evaluate both SVM and k-NN models on validation set.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    
    # TODO: Implement evaluation pipeline
    # 1. Load validation data
    # 2. Extract features
    # 3. Load trained models
    # 4. Evaluate both models
    # 5. Generate confusion matrices
    # 6. Print classification reports
    # 7. Compare models and identify best performer
    
    print("\nEvaluation pipeline:")
    print("1. Loading validation data...")
    # data_loader = DataLoader(config_path)
    # X_val, y_val = load_validation_data()
    
    print("2. Extracting features...")
    # feature_extractor = FeatureExtractor(config_path)
    # X_val_features = extract_features(X_val)
    
    print("3. Loading trained models...")
    # svm = SVMClassifier(config_path)
    # svm.load("models/svm_model")
    # knn = KNNClassifier(config_path)
    # knn.load("models/knn_model")
    
    print("4. Evaluating SVM...")
    # svm_results = svm.evaluate(X_val_features, y_val, use_rejection=True)
    
    print("5. Evaluating k-NN...")
    # knn_results = knn.evaluate(X_val_features, y_val, use_rejection=True)
    
    print("6. Generating visualizations...")
    # viz = Visualization(config_path)
    # viz.plot_confusion_matrix(y_val, svm_results['predictions'], "results/svm_cm.png")
    # viz.plot_confusion_matrix(y_val, knn_results['predictions'], "results/knn_cm.png")
    # viz.plot_training_comparison(svm_results, knn_results, "results/comparison.png")
    
    print("7. Printing reports...")
    # viz.print_classification_report(y_val, svm_results['predictions'])
    # viz.print_classification_report(y_val, knn_results['predictions'])
    
    print("\nEvaluation completed!")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Dictionary with metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics


if __name__ == "__main__":
    evaluate_models()

