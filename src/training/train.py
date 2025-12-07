"""
Training Module

Main training script for both SVM and k-NN classifiers.
"""

import numpy as np
import yaml
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.classifiers.svm_classifier import SVMClassifier
from src.classifiers.knn_classifier import KNNClassifier
from src.feature_extraction import FeatureExtractor
from src.data_augmentation import DataAugmenter
from src.utils.data_loader import DataLoader


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
    
    # TODO: Implement training pipeline
    # 1. Load data using DataLoader
    # 2. Apply data augmentation using DataAugmenter
    # 3. Extract features using FeatureExtractor
    # 4. Train SVM classifier
    # 5. Train k-NN classifier
    # 6. Evaluate both models
    # 7. Save best model
    
    print("\nTraining pipeline:")
    print("1. Loading dataset...")
    # data_loader = DataLoader(config_path)
    # images, labels = data_loader.load_images_from_folders()
    
    print("2. Applying data augmentation...")
    # augmenter = DataAugmenter(config_path)
    # augmented_images, augmented_labels = augmenter.augment_dataset(images, labels)
    
    print("3. Extracting features...")
    # feature_extractor = FeatureExtractor(config_path)
    # X_train_features = [feature_extractor.extract_features(img) for img in X_train]
    # X_val_features = [feature_extractor.extract_features(img) for img in X_val]
    
    print("4. Training SVM classifier...")
    # svm = SVMClassifier(config_path)
    # svm.train(X_train_features, y_train)
    # svm.save("models/svm_model")
    
    print("5. Training k-NN classifier...")
    # knn = KNNClassifier(config_path)
    # knn.train(X_train_features, y_train)
    # knn.save("models/knn_model")
    
    print("6. Evaluating models...")
    # svm_results = svm.evaluate(X_val_features, y_val)
    # knn_results = knn.evaluate(X_val_features, y_val)
    
    print("7. Saving best model...")
    # Compare results and save best model as "best_model.pkl"
    
    print("\nTraining completed!")


if __name__ == "__main__":
    train_models()

