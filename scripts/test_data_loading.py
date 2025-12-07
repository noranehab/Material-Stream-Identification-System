"""
Test script to verify data loading works correctly.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader
import numpy as np


def main():
    """
    Test data loading functionality.
    """
    print("=" * 60)
    print("Testing Data Loading")
    print("=" * 60)
    
    # Initialize data loader
    data_loader = DataLoader("config/config.yaml")
    
    # Load images
    print("\nLoading images from dataset...")
    images, labels = data_loader.load_images_from_folders()
    
    if len(images) == 0:
        print("\n❌ ERROR: No images loaded! Check your data path.")
        return
    
    # Display statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    print(f"Total images: {len(images)}")
    print(f"Total labels: {len(labels)}")
    
    # Class distribution
    class_counts = data_loader.get_class_counts(labels)
    print("\nClass distribution:")
    for class_id in sorted(class_counts.keys()):
        class_name = data_loader.classes.get(class_id, f"Class_{class_id}")
        count = class_counts[class_id]
        print(f"  {class_name} (ID {class_id}): {count} images")
    
    # Image shape check
    if len(images) > 0:
        sample_shape = images[0].shape
        print(f"\nImage shape: {sample_shape}")
        print(f"Expected shape: {data_loader.image_size[::-1]} (H, W, C)")
        
        expected_h, expected_w = data_loader.image_size[::-1]  # (H, W) from (W, H)
        if sample_shape[:2] != (expected_h, expected_w):
            print("WARNING: Image shape doesn't match expected size!")
        else:
            print("SUCCESS: Image shapes are correct!")
    
    # Test train/val split
    print("\n" + "=" * 60)
    print("Testing Train/Validation Split")
    print("=" * 60)
    
    X_train, X_val, y_train, y_val = data_loader.split_dataset(images, labels)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    train_counts = data_loader.get_class_counts(y_train)
    val_counts = data_loader.get_class_counts(y_val)
    
    print("\nTraining set distribution:")
    for class_id in sorted(train_counts.keys()):
        class_name = data_loader.classes.get(class_id, f"Class_{class_id}")
        print(f"  {class_name}: {train_counts[class_id]} images")
    
    print("\nValidation set distribution:")
    for class_id in sorted(val_counts.keys()):
        class_name = data_loader.classes.get(class_id, f"Class_{class_id}")
        print(f"  {class_name}: {val_counts[class_id]} images")
    
    print("\nSUCCESS: Data loading test completed successfully!")
    print("\nNext steps:")
    print("1. Run: python scripts/prepare_data.py (to augment data)")
    print("2. Run: python scripts/train_models.py (to train models)")


if __name__ == "__main__":
    main()

