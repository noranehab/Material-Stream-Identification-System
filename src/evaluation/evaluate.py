"""
Evaluation Module

Comprehensive evaluation of trained models with detailed metrics.
"""

import numpy as np
import yaml
from pathlib import Path
import sys
import pickle
import joblib
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.classifiers.svm_classifier import SVMClassifier
from src.classifiers.knn_classifier import KNNClassifier
from src.feature_extraction import FeatureExtractor
from src.utils.data_loader import DataLoader
from src.utils.visualization import Visualization
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA


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
    
    # Create results directory
    results_dir = Path(config['evaluation']['results_path'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load validation data
    print("\n1. Loading validation data...")
    processed_dir = Path(config['dataset']['processed_data_path'])
    val_pkl_path = processed_dir / "val_data.pkl"
    
    if not val_pkl_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_pkl_path}. "
            "Please run scripts/prepare_data.py first."
        )
    
    with open(val_pkl_path, 'rb') as f:
        X_val_images, y_val = pickle.load(f)
    
    y_val = np.array(y_val)
    print(f"   Validation samples: {len(X_val_images)}")
    
    # 2. Extract features
    print("\n2. Extracting features from validation images...")
    feature_extractor = FeatureExtractor(config_path)
    print(f"   Feature extraction method: {config['feature_extraction']['method']}")
    
    X_val_features = []
    for i, img in enumerate(X_val_images):
        if (i + 1) % 25 == 0 or (i + 1) == len(X_val_images):
            progress_pct = (i + 1) / len(X_val_images) * 100
            print(f"      Progress: {i + 1}/{len(X_val_images)} ({progress_pct:.1f}%)")
        features = feature_extractor.extract_features(img)
        X_val_features.append(features)
    
    X_val_features = np.array(X_val_features)
    print(f"   Validation features shape: {X_val_features.shape}")
    
    # 3. Load trained models
    print("\n3. Loading trained models...")
    models_dir = Path(config['training']['model_save_path'])
    
    # Check for PCA transformers first (use regular models' PCA to ensure compatibility)
    print("\n3.5. Checking for PCA transformation...")
    pca_transformer = None
    # Check regular model PCA first (most reliable)
    svm_pca_path = models_dir / "svm_model_pca.pkl"
    if svm_pca_path.exists():
        print(f"   PCA transformer found at {svm_pca_path}. Applying PCA...")
        pca_transformer = joblib.load(str(svm_pca_path))
        X_val_features = pca_transformer.transform(X_val_features)
        print(f"   Features after PCA: {X_val_features.shape}")
    else:
        # Try ensemble PCA as fallback
        ensemble_pca_path = models_dir / "best_model_ensemble_pca.pkl"
        if ensemble_pca_path.exists():
            print(f"   PCA transformer found at {ensemble_pca_path}. Applying PCA...")
            pca_transformer = joblib.load(str(ensemble_pca_path))
            X_val_features = pca_transformer.transform(X_val_features)
            print(f"   Features after PCA: {X_val_features.shape}")
        else:
            print("   No PCA transformer found. Using original features.")
    
    # Use regular models to ensure PCA compatibility
    # (best_model versions may have been created with different PCA settings)
    svm_model_path = models_dir / "svm_model"
    knn_model_path = models_dir / "knn_model"
    
    # Try best models only if they exist AND we're not using PCA
    if pca_transformer is None:
        best_svm_path = models_dir / "best_model_svm"
        best_knn_path = models_dir / "best_model_knn"
        if (best_svm_path.parent / f"{best_svm_path.name}_model.pkl").exists():
            svm_model_path = best_svm_path
        if (best_knn_path.parent / f"{best_knn_path.name}_model.pkl").exists():
            knn_model_path = best_knn_path
    
    svm = SVMClassifier(config_path)
    if (svm_model_path.parent / f"{svm_model_path.name}_model.pkl").exists():
        svm.load(str(svm_model_path))
        print(f"   SVM model loaded from {svm_model_path}")
    else:
        print("   WARNING: SVM model not found. Skipping SVM evaluation.")
        svm = None
    
    knn = KNNClassifier(config_path)
    if (knn_model_path.parent / f"{knn_model_path.name}_model.pkl").exists():
        knn.load(str(knn_model_path))
        print(f"   k-NN model loaded from {knn_model_path}")
    else:
        print("   WARNING: k-NN model not found. Skipping k-NN evaluation.")
        knn = None
    
    # 4. Evaluate models
    print("\n4. Evaluating models...")
    svm_results = None
    knn_results = None
    svm_results_no_reject = None
    knn_results_no_reject = None
    
    if svm is not None:
        print("   Evaluating SVM...")
        svm_results = svm.evaluate(X_val_features, y_val, use_rejection=True)
        svm_results_no_reject = svm.evaluate(X_val_features, y_val, use_rejection=False)
    
    if knn is not None:
        print("   Evaluating k-NN...")
        knn_results = knn.evaluate(X_val_features, y_val, use_rejection=True)
        knn_results_no_reject = knn.evaluate(X_val_features, y_val, use_rejection=False)
    
    # 5. Calculate per-class metrics
    print("\n5. Calculating per-class metrics...")
    def calculate_per_class_metrics(y_true, y_pred, classes):
        """Calculate metrics per class."""
        per_class_metrics = {}
        for class_id in range(len(classes)):
            if class_id == 6:  # Skip Unknown class for primary metrics
                continue
            mask = y_true == class_id
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == y_true[mask])
                per_class_metrics[class_id] = {
                    'accuracy': class_acc,
                    'samples': mask.sum()
                }
        return per_class_metrics
    
    # Calculate metrics
    svm_per_class = None
    knn_per_class = None
    svm_avg_primary = 0.0
    knn_avg_primary = 0.0
    svm_rejection_rate = 0.0
    knn_rejection_rate = 0.0
    
    if svm_results is not None:
        svm_per_class = calculate_per_class_metrics(y_val, svm_results['predictions'], config['classes'])
        svm_avg_primary = np.mean([m['accuracy'] for m in svm_per_class.values()])
        if svm_results['rejected'] is not None:
            svm_rejection_rate = svm_results['rejected'].sum() / len(svm_results['rejected'])
    
    if knn_results is not None:
        knn_per_class = calculate_per_class_metrics(y_val, knn_results['predictions'], config['classes'])
        knn_avg_primary = np.mean([m['accuracy'] for m in knn_per_class.values()])
        if knn_results['rejected'] is not None:
            knn_rejection_rate = knn_results['rejected'].sum() / len(knn_results['rejected'])
    
    # Clear, formatted output
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    if svm_results is not None and knn_results is not None:
        print("\n" + "-"*70)
        print("OVERALL ACCURACY (All Classes)")
        print("-"*70)
        print(f"{'Metric':<30} {'SVM':<20} {'k-NN':<20}")
        print("-"*70)
        print(f"{'With Rejection':<30} {svm_results['accuracy']:.4f} ({svm_results['accuracy']*100:>6.2f}%)  {knn_results['accuracy']:.4f} ({knn_results['accuracy']*100:>6.2f}%)")
        print(f"{'Without Rejection':<30} {svm_results_no_reject['accuracy']:.4f} ({svm_results_no_reject['accuracy']*100:>6.2f}%)  {knn_results_no_reject['accuracy']:.4f} ({knn_results_no_reject['accuracy']*100:>6.2f}%)")
        print(f"{'Rejection Rate':<30} {svm_rejection_rate:>6.2%}              {knn_rejection_rate:>6.2%}")
        
        print("\n" + "-"*70)
        print("PRIMARY CLASSES ACCURACY (6 Classes - Excluding Unknown)")
        print("-"*70)
        print(f"{'Metric':<30} {'SVM':<20} {'k-NN':<20}")
        print("-"*70)
        print(f"{'With Rejection':<30} {svm_avg_primary:.4f} ({svm_avg_primary*100:>6.2f}%)  {knn_avg_primary:.4f} ({knn_avg_primary*100:>6.2f}%)")
        
        if svm_per_class and knn_per_class:
            print("\n" + "-"*70)
            print("PER-CLASS ACCURACY (Primary Classes)")
            print("-"*70)
            print(f"{'Class':<20} {'SVM':<25} {'k-NN':<25}")
            print("-"*70)
            for class_id in sorted(svm_per_class.keys()):
                class_name = config['classes'][class_id]
                svm_acc = svm_per_class[class_id]['accuracy']
                knn_acc = knn_per_class[class_id]['accuracy']
                svm_str = f"{svm_acc:.4f} ({svm_acc*100:>6.2f}%)"
                knn_str = f"{knn_acc:.4f} ({knn_acc*100:>6.2f}%)"
                print(f"{class_name:<20} {svm_str:<25} {knn_str:<25}")
        
        print("="*70)
    elif svm_results is not None:
        print(f"\nSVM Results:")
        print(f"  Overall Accuracy (with rejection): {svm_results['accuracy']:.4f} ({svm_results['accuracy']*100:.2f}%)")
        print(f"  Overall Accuracy (without rejection): {svm_results_no_reject['accuracy']:.4f} ({svm_results_no_reject['accuracy']*100:.2f}%)")
        print(f"  Primary Classes Accuracy: {svm_avg_primary:.4f} ({svm_avg_primary*100:.2f}%)")
    elif knn_results is not None:
        print(f"\nk-NN Results:")
        print(f"  Overall Accuracy (with rejection): {knn_results['accuracy']:.4f} ({knn_results['accuracy']*100:.2f}%)")
        print(f"  Overall Accuracy (without rejection): {knn_results_no_reject['accuracy']:.4f} ({knn_results_no_reject['accuracy']*100:.2f}%)")
        print(f"  Primary Classes Accuracy: {knn_avg_primary:.4f} ({knn_avg_primary*100:.2f}%)")
    
    # 6. Generate visualizations
    print("\n6. Generating visualizations...")
    viz = Visualization(config_path)
    
    if svm_results is not None:
        print("   Creating SVM confusion matrix...")
        viz.plot_confusion_matrix(
            y_val, 
            svm_results['predictions'], 
            str(results_dir / "svm_confusion_matrix.png")
        )
        plt.close()  # Close to free memory
    
    if knn_results is not None:
        print("   Creating k-NN confusion matrix...")
        viz.plot_confusion_matrix(
            y_val, 
            knn_results['predictions'], 
            str(results_dir / "knn_confusion_matrix.png")
        )
        plt.close()
    
    if svm_results is not None and knn_results is not None:
        print("   Creating model comparison chart...")
        viz.plot_training_comparison(
            svm_results, 
            knn_results, 
            str(results_dir / "model_comparison.png")
        )
        plt.close()
    
    # 7. Print classification reports
    print("\n7. Classification Reports:")
    print("\n" + "=" * 60)
    if svm_results is not None:
        print("\nSVM Classification Report (with rejection):")
        print("-" * 60)
        viz.print_classification_report(y_val, svm_results['predictions'])
    
    if knn_results is not None:
        print("\nk-NN Classification Report (with rejection):")
        print("-" * 60)
        viz.print_classification_report(y_val, knn_results['predictions'])
    
    # 8. Save detailed results
    print("\n8. Saving detailed results...")
    results_summary = {
        'svm': {
            'accuracy': float(svm_results['accuracy']) if svm_results is not None else None,
            'per_class_accuracy': {k: float(v['accuracy']) for k, v in svm_per_class.items()} if svm_results is not None else None,
            'rejection_rate': float(svm_results['rejected'].sum() / len(svm_results['rejected'])) if svm_results is not None and svm_results['rejected'] is not None else None
        } if svm_results is not None else None,
        'knn': {
            'accuracy': float(knn_results['accuracy']) if knn_results is not None else None,
            'per_class_accuracy': {k: float(v['accuracy']) for k, v in knn_per_class.items()} if knn_results is not None else None,
            'rejection_rate': float(knn_results['rejected'].sum() / len(knn_results['rejected'])) if knn_results is not None and knn_results['rejected'] is not None else None
        } if knn_results is not None else None
    }
    
    import json
    with open(results_dir / "evaluation_results.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"   Results saved to {results_dir}")
    print("\n" + "=" * 60)
    print("Evaluation completed!")
    print("=" * 60)


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

