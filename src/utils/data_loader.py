"""
Data Loader Utility

Handles loading and preprocessing of images from the dataset.
Supports both folder-based and annotation file-based dataset structures.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
import yaml
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Loads and preprocesses images from the dataset.
    
    Supports:
    - Folder-based structure (each class in separate folder)
    - Annotation file-based structure (CSV/JSON with image paths and labels)
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize data loader.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = Path(self.config['dataset']['raw_data_path'])
        self.image_size = tuple(self.config['dataset']['image_size'])
        self.classes = self.config['classes']
    
    def load_images_from_folders(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load images from folder-based structure.
        
        Supports two structures:
        1. data/raw/dataset/glass/, data/raw/dataset/paper/, etc.
        2. data/raw/class_0/, data/raw/class_1/, etc.
        
        Returns:
            images: List of image arrays
            labels: List of corresponding class labels
        """
        images = []
        labels = []
        
        # Mapping from folder names to class IDs
        folder_to_class = {
            'glass': 0,
            'paper': 1,
            'cardboard': 2,
            'plastic': 3,
            'metal': 4,
            'trash': 5,
            'unknown': 6
        }
        
        # Check if dataset is in a subfolder (data/raw/dataset/)
        dataset_path = self.raw_data_path / "dataset"
        if not dataset_path.exists():
            dataset_path = self.raw_data_path
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Load images from each class folder
        for folder_name, class_id in folder_to_class.items():
            class_folder = dataset_path / folder_name
            
            if not class_folder.exists():
                print(f"Warning: Folder {class_folder} does not exist. Skipping class {class_id}.")
                continue
            
            # Get all image files in the folder
            image_files = [f for f in class_folder.iterdir() 
                          if f.suffix.lower() in image_extensions]
            
            print(f"Loading {len(image_files)} images from {folder_name} (class {class_id})...")
            
            skipped_count = 0
            skipped_reasons = {'empty': 0, 'corrupted': 0, 'other': 0}
            
            for img_file in image_files:
                try:
                    # Check if file is empty (0 bytes)
                    file_size = img_file.stat().st_size
                    if file_size == 0:
                        skipped_count += 1
                        skipped_reasons['empty'] += 1
                        continue
                    
                    # Load image using OpenCV
                    img = cv2.imread(str(img_file))
                    
                    if img is None:
                        skipped_count += 1
                        skipped_reasons['corrupted'] += 1
                        continue
                    
                    # Verify image has valid dimensions
                    if img.shape[0] == 0 or img.shape[1] == 0:
                        skipped_count += 1
                        skipped_reasons['corrupted'] += 1
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize to target size
                    img = cv2.resize(img, self.image_size)
                    
                    images.append(img)
                    labels.append(class_id)
                    
                except Exception as e:
                    skipped_count += 1
                    skipped_reasons['other'] += 1
                    continue
            
            # Print summary for this class
            loaded_count = len([l for l in labels if l == class_id])
            if skipped_count > 0:
                print(f"  -> Loaded: {loaded_count}, Skipped: {skipped_count} (empty: {skipped_reasons['empty']}, corrupted: {skipped_reasons['corrupted']}, other: {skipped_reasons['other']})")
        
        # Count total expected image files
        total_expected = 0
        for folder_name in folder_to_class.keys():
            class_folder = dataset_path / folder_name
            if class_folder.exists():
                total_expected += len([f for f in class_folder.iterdir() 
                                      if f.suffix.lower() in image_extensions])
        
        total_loaded = len(images)
        total_skipped = total_expected - total_loaded
        
        print(f"\n" + "=" * 60)
        print(f"Loading Summary")
        print(f"=" * 60)
        print(f"Total image files found: {total_expected}")
        print(f"Successfully loaded: {total_loaded}")
        if total_skipped > 0:
            print(f"Skipped (empty/corrupted): {total_skipped}")
        print(f"Class distribution: {self.get_class_counts(labels)}")
        
        return images, labels
    
    def load_images_from_annotations(self, annotation_file: str) -> Tuple[List[np.ndarray], List[int]]:
        """
        Load images from annotation file.
        
        Args:
            annotation_file: Path to CSV/JSON file with image paths and labels
        
        Returns:
            images: List of image arrays
            labels: List of corresponding class labels
        """
        images = []
        labels = []
        
        # TODO: Implement annotation-based loading
        # - Read annotation file (CSV or JSON)
        # - For each row/entry, load image from path
        # - Resize and preprocess image
        # - Append to images and labels lists
        
        return images, labels
    
    def split_dataset(
        self,
        images: List[np.ndarray],
        labels: List[int],
        test_size: float = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
        """
        Split dataset into train and validation sets.
        
        Args:
            images: List of images
            labels: List of labels
            test_size: Validation set size (from config if None)
        
        Returns:
            X_train, X_val, y_train, y_val
        """
        if test_size is None:
            test_size = self.config['dataset']['val_split']
        
        # Convert to numpy arrays for sklearn
        images_array = np.array(images)
        labels_array = np.array(labels)
        
        # Use stratified split to ensure balanced distribution
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                images_array, labels_array,
                test_size=test_size,
                stratify=labels_array,
                random_state=self.config['training']['random_seed']
            )
        except ValueError as e:
            # If stratification fails (e.g., some classes have too few samples),
            # fall back to regular split
            print(f"Warning: Stratified split failed ({e}). Using regular split.")
            X_train, X_val, y_train, y_val = train_test_split(
                images_array, labels_array,
                test_size=test_size,
                random_state=self.config['training']['random_seed']
            )
        
        # Convert back to lists for compatibility
        X_train = [X_train[i] for i in range(len(X_train))]
        X_val = [X_val[i] for i in range(len(X_val))]
        y_train = y_train.tolist()
        y_val = y_val.tolist()
        
        return X_train, X_val, y_train, y_val
    
    def get_class_counts(self, labels: List[int]) -> Dict[int, int]:
        """
        Count samples per class.
        
        Args:
            labels: List of class labels
        
        Returns:
            Dictionary mapping class_id to count
        """
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique, counts))

