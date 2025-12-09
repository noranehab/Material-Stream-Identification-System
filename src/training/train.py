"""
Training Module

Main training script for both SVM and k-NN classifiers.
"""

import numpy as np
import yaml
from pathlib import Path
import sys
import pickle
import shutil
import joblib
from sklearn.decomposition import PCA

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.classifiers.svm_classifier import SVMClassifier
from src.classifiers.knn_classifier import KNNClassifier
from src.feature_extraction import FeatureExtractor
from src.utils.data_loader import DataLoader

# Check if scikit-image is available
try:
    from skimage.feature import hog, local_binary_pattern
    from skimage import color
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def train_models(config_path: str = "config/config.yaml"):
    """
    Train both SVM and k-NN models.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("Material Classification - Model Training")
    print("=" * 60)
    
    # Create models directory if it doesn't exist
    models_dir = Path(config['training']['model_save_path'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load augmented training data and validation data
    print("\n1. Loading augmented training data and validation data...")
    processed_dir = Path(config['dataset']['processed_data_path'])
    
    train_pkl_path = processed_dir / "train_augmented.pkl"
    val_pkl_path = processed_dir / "val_data.pkl"
    
    if not train_pkl_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_pkl_path}. "
            "Please run scripts/prepare_data.py first."
        )
    if not val_pkl_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {val_pkl_path}. "
            "Please run scripts/prepare_data.py first."
        )
    
    with open(train_pkl_path, 'rb') as f:
        X_train_images, y_train = pickle.load(f)
    
    with open(val_pkl_path, 'rb') as f:
        X_val_images, y_val = pickle.load(f)
    
    print(f"   Training samples: {len(X_train_images)}")
    print(f"   Validation samples: {len(X_val_images)}")
    
    # Convert labels to numpy arrays
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    
    # 2. Extract features from all images
    print("\n2. Extracting features from all images...")
    
    # Check if scikit-image is available for combined features
    feature_method = config['feature_extraction']['method']
    if feature_method == "combined" and not HAS_SKIMAGE:
        print("   WARNING: scikit-image not available. Combined features will use OpenCV fallbacks.")
        print("   For best results, install scikit-image: pip install scikit-image")
    
    try:
        feature_extractor = FeatureExtractor(config_path)
        print(f"   Feature extraction method: {feature_method}")
        
        # Test feature extraction on a sample image
        if len(X_train_images) > 0:
            print("   Testing feature extraction on sample image...")
            test_features = feature_extractor.extract_features(X_train_images[0])
            print(f"   Sample feature vector shape: {test_features.shape}")
    except Exception as e:
        print(f"   ERROR: Failed to initialize feature extractor: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("   Processing training images...")
    print(f"   Total images to process: {len(X_train_images)}")
    X_train_features = []
    try:
        for i, img in enumerate(X_train_images):
            # Show progress more frequently
            if (i + 1) % 50 == 0 or (i + 1) == len(X_train_images):
                progress_pct = (i + 1) / len(X_train_images) * 100
                print(f"      Progress: {i + 1}/{len(X_train_images)} ({progress_pct:.1f}%)")
            try:
                features = feature_extractor.extract_features(img)
                X_train_features.append(features)
            except KeyboardInterrupt:
                print(f"\n      Interrupted at image {i + 1}")
                raise
            except Exception as e:
                print(f"      ERROR processing image {i + 1}: {e}")
                print(f"      Image shape: {img.shape if img is not None else 'None'}")
                import traceback
                traceback.print_exc()
                raise
    except KeyboardInterrupt:
        print("\n   Training interrupted by user.")
        raise
    except Exception as e:
        print(f"   ERROR during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    X_train_features = np.array(X_train_features)
    print(f"   Training features shape: {X_train_features.shape}")
    
    print("   Processing validation images...")
    print(f"   Total images to process: {len(X_val_images)}")
    X_val_features = []
    try:
        for i, img in enumerate(X_val_images):
            # Show progress more frequently
            if (i + 1) % 25 == 0 or (i + 1) == len(X_val_images):
                progress_pct = (i + 1) / len(X_val_images) * 100
                print(f"      Progress: {i + 1}/{len(X_val_images)} ({progress_pct:.1f}%)")
            try:
                features = feature_extractor.extract_features(img)
                X_val_features.append(features)
            except KeyboardInterrupt:
                print(f"\n      Interrupted at image {i + 1}")
                raise
            except Exception as e:
                print(f"      ERROR processing validation image {i + 1}: {e}")
                print(f"      Image shape: {img.shape if img is not None else 'None'}")
                import traceback
                traceback.print_exc()
                raise
    except KeyboardInterrupt:
        print("\n   Training interrupted by user.")
        raise
    except Exception as e:
        print(f"   ERROR during validation feature extraction: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    X_val_features = np.array(X_val_features)
    print(f"   Validation features shape: {X_val_features.shape}")
    
    # Apply PCA to reduce dimensionality for faster training
    print("\n2.5. Applying PCA for dimensionality reduction...")
    print(f"   Original feature dimension: {X_train_features.shape[1]}")
    
    # Use PCA to reduce dimensionality while retaining more information
    # Use variance-based selection (optimized to 99.5% from hyperparameter search)
    pca_variance = config['training'].get('pca_variance', 0.99)
    pca = PCA(n_components=pca_variance, random_state=config['training']['random_seed'])
    print(f"   Reducing to retain {pca_variance:.1%} variance...")
    X_train_features_reduced = pca.fit_transform(X_train_features)
    X_val_features_reduced = pca.transform(X_val_features)
    
    explained_variance = pca.explained_variance_ratio_.sum()
    n_components = X_train_features_reduced.shape[1]
    print(f"   Reduced feature dimension: {n_components}")
    print(f"   Explained variance: {explained_variance:.2%}")
    print(f"   This should significantly speed up training while retaining accuracy!")
    
    # Save PCA transformer for later use (if needed)
    pca_transformer = pca
    
    # Use reduced features for training
    X_train_features = X_train_features_reduced
    X_val_features = X_val_features_reduced
    
    # 3. Train SVM classifier
    print("\n3. Training SVM classifier...")
    svm_kernel = config['svm']['kernel']
    print(f"   Using {svm_kernel.upper()} kernel")
    if svm_kernel == 'rbf':
        print("   WARNING: RBF kernel can be very slow with large feature vectors.")
        print("   Consider using 'linear' kernel in config.yaml for faster training.")
    
    # Check for class weights
    use_class_weights = config['training'].get('use_class_weights', False)
    class_weights = None
    if use_class_weights:
        class_weights = config['training'].get('class_weights', None)
        if class_weights:
            print(f"   Using class-weighted training:")
            for class_id, weight in class_weights.items():
                class_name = config['classes'][class_id]
                print(f"     {class_name} (ID {class_id}): weight = {weight}")
            # Convert to format expected by sklearn
            class_weights = {int(k): float(v) for k, v in class_weights.items()}
    
    svm = SVMClassifier(config_path)
    # Pass class weights to training if available
    if class_weights:
        svm.model.set_params(class_weight=class_weights)
    svm.train(X_train_features, y_train)
    
    svm_model_path = models_dir / "svm_model"
    svm.save(str(svm_model_path))
    # Save PCA transformer for SVM model
    joblib.dump(pca_transformer, f"{svm_model_path}_pca.pkl")
    print(f"   SVM model saved to {svm_model_path}")
    print(f"   PCA transformer saved to {svm_model_path}_pca.pkl")
    
    # 4. Train k-NN classifier
    print("\n4. Training k-NN classifier...")
    knn = KNNClassifier(config_path)
    knn.train(X_train_features, y_train)
    
    knn_model_path = models_dir / "knn_model"
    knn.save(str(knn_model_path))
    # Save PCA transformer for k-NN model
    joblib.dump(pca_transformer, f"{knn_model_path}_pca.pkl")
    print(f"   k-NN model saved to {knn_model_path}")
    print(f"   PCA transformer saved to {knn_model_path}_pca.pkl")
    
    # 5. Evaluate both models on validation set
    print("\n5. Evaluating both models on validation set...")
    print("   Evaluating SVM (without rejection first)...")
    svm_results_no_reject = svm.evaluate(X_val_features, y_val, use_rejection=False)
    svm_accuracy_no_reject = svm_results_no_reject['accuracy']
    
    print("   Evaluating SVM (with rejection)...")
    svm_results = svm.evaluate(X_val_features, y_val, use_rejection=True)
    svm_accuracy = svm_results['accuracy']
    
    print("   Evaluating k-NN (without rejection first)...")
    knn_results_no_reject = knn.evaluate(X_val_features, y_val, use_rejection=False)
    knn_accuracy_no_reject = knn_results_no_reject['accuracy']
    
    print("   Evaluating k-NN (with rejection)...")
    knn_results = knn.evaluate(X_val_features, y_val, use_rejection=True)
    knn_accuracy = knn_results['accuracy']
    
    print(f"\n   Raw Accuracy (without rejection):")
    print(f"   SVM: {svm_accuracy_no_reject:.4f} ({svm_accuracy_no_reject*100:.2f}%)")
    print(f"   k-NN: {knn_accuracy_no_reject:.4f} ({knn_accuracy_no_reject*100:.2f}%)")
    
    # Calculate per-class accuracy
    def calculate_per_class_accuracy(y_true, y_pred, classes):
        """Calculate accuracy per class."""
        per_class_acc = {}
        for class_id in range(len(classes)):
            if class_id == 6:  # Skip Unknown class
                continue
            mask = y_true == class_id
            if mask.sum() > 0:
                class_acc = np.mean(y_pred[mask] == y_true[mask])
                per_class_acc[class_id] = class_acc
        return per_class_acc
    
    # Calculate per-class accuracy with rejection
    svm_per_class = calculate_per_class_accuracy(y_val, svm_results['predictions'], config['classes'])
    knn_per_class = calculate_per_class_accuracy(y_val, knn_results['predictions'], config['classes'])
    
    # Calculate per-class accuracy without rejection (for comparison)
    svm_per_class_no_reject = calculate_per_class_accuracy(y_val, svm_results_no_reject['predictions'], config['classes'])
    knn_per_class_no_reject = calculate_per_class_accuracy(y_val, knn_results_no_reject['predictions'], config['classes'])
    
    # Calculate average accuracy across 6 primary classes (excluding Unknown)
    svm_avg_primary = np.mean(list(svm_per_class.values()))
    knn_avg_primary = np.mean(list(knn_per_class.values()))
    svm_avg_primary_no_reject = np.mean(list(svm_per_class_no_reject.values()))
    knn_avg_primary_no_reject = np.mean(list(knn_per_class_no_reject.values()))
    
    # Show rejection rates
    svm_rejection_rate = 0.0
    knn_rejection_rate = 0.0
    if svm_results['rejected'] is not None:
        svm_rejection_rate = svm_results['rejected'].sum() / len(svm_results['rejected'])
    if knn_results['rejected'] is not None:
        knn_rejection_rate = knn_results['rejected'].sum() / len(knn_results['rejected'])
    
    # Clear, formatted output
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    
    print("\n" + "-"*70)
    print("OVERALL ACCURACY (All Classes)")
    print("-"*70)
    print(f"{'Metric':<30} {'SVM':<20} {'k-NN':<20}")
    print("-"*70)
    print(f"{'With Rejection':<30} {svm_accuracy:.4f} ({svm_accuracy*100:>6.2f}%)  {knn_accuracy:.4f} ({knn_accuracy*100:>6.2f}%)")
    print(f"{'Without Rejection':<30} {svm_accuracy_no_reject:.4f} ({svm_accuracy_no_reject*100:>6.2f}%)  {knn_accuracy_no_reject:.4f} ({knn_accuracy_no_reject*100:>6.2f}%)")
    print(f"{'Rejection Rate':<30} {svm_rejection_rate:>6.2%}              {knn_rejection_rate:>6.2%}")
    
    print("\n" + "-"*70)
    print("PRIMARY CLASSES ACCURACY (6 Classes - Excluding Unknown)")
    print("-"*70)
    print(f"{'Metric':<30} {'SVM':<20} {'k-NN':<20}")
    print("-"*70)
    print(f"{'With Rejection':<30} {svm_avg_primary:.4f} ({svm_avg_primary*100:>6.2f}%)  {knn_avg_primary:.4f} ({knn_avg_primary*100:>6.2f}%)")
    print(f"{'Without Rejection':<30} {svm_avg_primary_no_reject:.4f} ({svm_avg_primary_no_reject*100:>6.2f}%)  {knn_avg_primary_no_reject:.4f} ({knn_avg_primary_no_reject*100:>6.2f}%)")
    
    print("\n" + "-"*70)
    print("PER-CLASS ACCURACY (Primary Classes)")
    print("-"*70)
    print(f"{'Class':<20} {'SVM':<25} {'k-NN':<25}")
    print("-"*70)
    for class_id in sorted(svm_per_class.keys()):
        class_name = config['classes'][class_id]
        svm_acc = svm_per_class[class_id]
        knn_acc = knn_per_class[class_id]
        svm_str = f"{svm_acc:.4f} ({svm_acc*100:>6.2f}%)"
        knn_str = f"{knn_acc:.4f} ({knn_acc*100:>6.2f}%)"
        print(f"{class_name:<20} {svm_str:<25} {knn_str:<25}")
    
    print("="*70)
    
    # Use no-rejection accuracy for best model selection (more accurate representation)
    svm_avg_primary = svm_avg_primary_no_reject
    knn_avg_primary = knn_avg_primary_no_reject
    
    # 6. Save best model (using average accuracy across primary classes)
    print("\n6. Saving best model...")
    # Compare both models
    models_to_compare = [
        ('svm', svm_avg_primary, svm_accuracy, svm_model_path),
        ('knn', knn_avg_primary, knn_accuracy, knn_model_path)
    ]
    
    # Sort by primary accuracy
    models_to_compare.sort(key=lambda x: x[1], reverse=True)
    best_model_name, best_accuracy, best_overall_accuracy, best_model_path = models_to_compare[0]
    
    print(f"\n   Model Comparison:")
    print(f"   {'Model':<15} {'Primary Accuracy':<20} {'Overall Accuracy':<20}")
    print("   " + "-" * 55)
    for name, acc, overall, path in models_to_compare:
        print(f"   {name.upper():<15} {acc:.4f} ({acc*100:>6.2f}%)     {overall:.4f} ({overall*100:>6.2f}%)")
    
    # Copy best individual model
    best_model_dest = models_dir / f"best_model_{best_model_name}"
    if best_model_path is not None:
        shutil.copy2(f"{best_model_path}_model.pkl", f"{best_model_dest}_model.pkl")
        shutil.copy2(f"{best_model_path}_scaler.pkl", f"{best_model_dest}_scaler.pkl")
        # Copy PCA transformer
        if Path(f"{best_model_path}_pca.pkl").exists():
            shutil.copy2(f"{best_model_path}_pca.pkl", f"{best_model_dest}_pca.pkl")
        print(f"\n   Best model: {best_model_name.upper()} (Accuracy: {best_accuracy:.4f})")
        print(f"   Best model saved to {best_model_dest}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\n{'MODEL PERFORMANCE SUMMARY':^70}")
    print("-" * 70)
    print(f"{'Model':<15} {'Overall Accuracy':<20} {'Primary Classes Accuracy':<25}")
    print("-" * 70)
    print(f"{'SVM':<15} {svm_accuracy:.4f} ({svm_accuracy*100:>6.2f}%)     {svm_avg_primary:.4f} ({svm_avg_primary*100:>6.2f}%)")
    print(f"{'k-NN':<15} {knn_accuracy:.4f} ({knn_accuracy*100:>6.2f}%)     {knn_avg_primary:.4f} ({knn_avg_primary*100:>6.2f}%)")
    print("=" * 70)
    
    # Best model summary
    print(f"\n{'BEST MODEL SELECTED':^70}")
    print("-" * 70)
    print(f"{'Model':<20} {'Primary Accuracy':<25} {'Overall Accuracy':<25}")
    print("-" * 70)
    print(f"{best_model_name.upper():<20} {best_accuracy:.4f} ({best_accuracy*100:>6.2f}%)     {best_overall_accuracy:.4f} ({best_overall_accuracy*100:>6.2f}%)")
    
    # Check if target accuracy is met
    target_accuracy = config['evaluation']['target_accuracy']
    if best_accuracy >= target_accuracy:
        print(f"\n  [SUCCESS] Target accuracy ({target_accuracy:.2f}) achieved!")
    else:
        print(f"\n  [WARNING] Target accuracy ({target_accuracy:.2f}) not yet achieved.")
        print(f"     Current: {best_accuracy:.4f}, Gap: {target_accuracy - best_accuracy:.4f}")
        print(f"     Consider:")
        print(f"     - Trying different feature extraction methods")
        print(f"     - Adjusting hyperparameters further")
        print(f"     - Increasing data augmentation")
        print(f"     - Collecting more training data for low-performing classes")
        print(f"     - Checking if rejection threshold is too high")


if __name__ == "__main__":
    train_models()

