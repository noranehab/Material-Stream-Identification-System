"""
Data Preparation Script

Prepares and preprocesses the dataset:
1. Loads raw images
2. Applies data augmentation
3. Saves processed and augmented data
"""

import sys
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.data_loader import DataLoader
from src.data_augmentation import DataAugmenter
import yaml


def main():
    """
    Main data preparation pipeline.
    """
    config_path = "config/config.yaml"
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Data Preparation Pipeline")
    print("=" * 60)

    # Load configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Load raw images
    print("\n1. Loading raw images...")
    data_loader = DataLoader(config_path)
    images, labels = data_loader.load_images_from_folders()

    # 2. Split into train/val (before augmentation)
    print("\n2. Splitting dataset (train/val)...")
    X_train, X_val, y_train, y_val = data_loader.split_dataset(images, labels)
    print(f"   Train samples: {len(X_train)}")
    print(f"   Val samples  : {len(X_val)}")

    # 3. Analyze class distribution
    print("\n3. Analyzing class distribution (train)...")
    class_counts = data_loader.get_class_counts(y_train)
    print(f"   Train class distribution: {class_counts}")

    # 4. Apply augmentation to training set only
    print("\n4. Applying data augmentation (train only)...")
    augmenter = DataAugmenter(config_path)
    augmentation_plan = augmenter.balance_dataset(class_counts)

    augmented_images = []
    augmented_labels = []

    for class_id, count in class_counts.items():
        # Skip Unknown class (6) for augmentation
        if class_id == 6:
            continue
        class_images = [img for img, lbl in zip(X_train, y_train) if lbl == class_id]
        target_count = augmentation_plan.get(class_id, count)

        print(f"   Class {class_id}: {count} -> {target_count}")
        augmented_class_images = augmenter.augment_class(
            class_images, len(class_images), target_count
        )
        augmented_images.extend(augmented_class_images)
        augmented_labels.extend([class_id] * len(augmented_class_images))

    print(f"\n   Augmented train samples: {len(augmented_images)}")

    # 5. Save processed data (pickle for speed)
    print("\n5. Saving processed data to data/processed/ ...")
    with open(processed_dir / "train_augmented.pkl", "wb") as f:
        pickle.dump((augmented_images, augmented_labels), f)
    with open(processed_dir / "val_data.pkl", "wb") as f:
        pickle.dump((X_val, y_val), f)

    print("\nData preparation completed!")
    print("Files saved to data/processed/:")
    print(" - train_augmented.pkl")
    print(" - val_data.pkl")
    print("\nNext step: Run training script (scripts/train_models.py)")


if __name__ == "__main__":
    main()
