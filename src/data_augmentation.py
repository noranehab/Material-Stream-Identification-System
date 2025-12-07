"""
Data Augmentation Module

Implements various augmentation techniques to increase dataset size by minimum 30%
and balance classes to target sample count (e.g., 500 per class).
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict
from pathlib import Path
import yaml
from PIL import Image

# Try to import albumentations, but make it optional
_HAS_ALBUMENTATIONS = None
_ALBUMENTATIONS_WARNING_SHOWN = False

def _check_albumentations():
    """Check if albumentations is available, show warning only once."""
    global _HAS_ALBUMENTATIONS, _ALBUMENTATIONS_WARNING_SHOWN
    
    if _HAS_ALBUMENTATIONS is None:
        try:
            import albumentations as A
            _HAS_ALBUMENTATIONS = True
        except ImportError:
            _HAS_ALBUMENTATIONS = False
            if not _ALBUMENTATIONS_WARNING_SHOWN:
                print("Note: Using OpenCV-based augmentation (albumentations not available).")
                _ALBUMENTATIONS_WARNING_SHOWN = True
    
    return _HAS_ALBUMENTATIONS

HAS_ALBUMENTATIONS = _check_albumentations()


class DataAugmenter:
    """
    Handles data augmentation for material classification dataset.
    
    Augmentation techniques:
    - Rotation
    - Flipping (horizontal/vertical)
    - Scaling
    - Color jittering
    - Noise addition
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the data augmenter with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.aug_config = self.config['augmentation']
        self.target_samples = self.config['dataset']['target_samples_per_class']
        
        # Build augmentation pipeline
        self.augmentation_pipeline = self._build_pipeline()
    
    def _build_pipeline(self):
        """
        Build augmentation pipeline based on configuration.
        
        Returns:
            Pipeline object (Albumentations Compose or OpenCV-based)
        """
        if HAS_ALBUMENTATIONS:
            return self._build_albumentations_pipeline()
        else:
            return self._build_opencv_pipeline()
    
    def _build_albumentations_pipeline(self):
        """Build pipeline using Albumentations."""
        transforms = []
        aug_tech = self.aug_config['techniques']
        
        if aug_tech['rotation']['enabled']:
            transforms.append(A.Rotate(limit=aug_tech['rotation']['max_angle'], p=aug_tech['rotation']['probability']))
        if aug_tech['horizontal_flip']['enabled']:
            transforms.append(A.HorizontalFlip(p=aug_tech['horizontal_flip']['probability']))
        if aug_tech['vertical_flip']['enabled']:
            transforms.append(A.VerticalFlip(p=aug_tech['vertical_flip']['probability']))
        if aug_tech['scaling']['enabled']:
            transforms.append(A.RandomScale(
                scale_limit=(aug_tech['scaling']['scale_range'][0] - 1.0, aug_tech['scaling']['scale_range'][1] - 1.0),
                p=aug_tech['scaling']['probability']
            ))
        if aug_tech['color_jitter']['enabled']:
            transforms.append(A.ColorJitter(
                brightness=aug_tech['color_jitter']['brightness'],
                contrast=aug_tech['color_jitter']['contrast'],
                saturation=aug_tech['color_jitter']['saturation'],
                hue=aug_tech['color_jitter']['hue'],
                p=aug_tech['color_jitter']['probability']
            ))
        if aug_tech['noise']['enabled']:
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=aug_tech['noise']['probability']))
        
        transforms.append(A.Resize(self.config['dataset']['image_size'][1], self.config['dataset']['image_size'][0]))
        return A.Compose(transforms)
    
    def _build_opencv_pipeline(self):
        """Build pipeline configuration for OpenCV-based augmentation."""
        return self.aug_config  # Return config for OpenCV-based augmentation
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to a single image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            Augmented image
        """
        if HAS_ALBUMENTATIONS:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image']
        else:
            return self._augment_image_opencv(image)
    
    def _augment_image_opencv(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentation using OpenCV."""
        aug_tech = self.aug_config['techniques']
        result = image.copy()
        h, w = result.shape[:2]
        
        # Rotation
        if aug_tech['rotation']['enabled'] and np.random.random() < aug_tech['rotation']['probability']:
            angle = np.random.uniform(-aug_tech['rotation']['max_angle'], aug_tech['rotation']['max_angle'])
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            result = cv2.warpAffine(result, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Horizontal Flip
        if aug_tech['horizontal_flip']['enabled'] and np.random.random() < aug_tech['horizontal_flip']['probability']:
            result = cv2.flip(result, 1)
        
        # Vertical Flip
        if aug_tech['vertical_flip']['enabled'] and np.random.random() < aug_tech['vertical_flip']['probability']:
            result = cv2.flip(result, 0)
        
        # Scaling
        if aug_tech['scaling']['enabled'] and np.random.random() < aug_tech['scaling']['probability']:
            scale = np.random.uniform(aug_tech['scaling']['scale_range'][0], aug_tech['scaling']['scale_range'][1])
            new_w, new_h = int(w * scale), int(h * scale)
            result = cv2.resize(result, (new_w, new_h))
            # Crop or pad to original size
            if scale > 1.0:
                start_x, start_y = (new_w - w) // 2, (new_h - h) // 2
                result = result[start_y:start_y+h, start_x:start_x+w]
            else:
                pad_x, pad_y = (w - new_w) // 2, (h - new_h) // 2
                result = cv2.copyMakeBorder(result, pad_y, h-new_h-pad_y, pad_x, w-new_w-pad_x, cv2.BORDER_REFLECT)
        
        # Color Jitter (convert to HSV, modify, convert back)
        if aug_tech['color_jitter']['enabled'] and np.random.random() < aug_tech['color_jitter']['probability']:
            hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV).astype(np.float32)
            if np.random.random() < 0.5:
                hsv[:,:,1] *= (1 + np.random.uniform(-aug_tech['color_jitter']['saturation'], aug_tech['color_jitter']['saturation']))
            if np.random.random() < 0.5:
                hsv[:,:,2] *= (1 + np.random.uniform(-aug_tech['color_jitter']['brightness'], aug_tech['color_jitter']['brightness']))
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Gaussian Noise
        if aug_tech['noise']['enabled'] and np.random.random() < aug_tech['noise']['probability']:
            noise = np.random.normal(0, np.random.uniform(5, 15), result.shape).astype(np.float32)
            result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        # Resize to target size
        target_size = tuple(self.config['dataset']['image_size'])
        result = cv2.resize(result, target_size)
        
        return result
    
    def augment_class(
        self,
        images: List[np.ndarray],
        current_count: int,
        target_count: int
    ) -> List[np.ndarray]:
        """
        Augment images for a specific class to reach target count.
        
        Args:
            images: List of images for the class
            current_count: Current number of images
            target_count: Target number of images
            
        Returns:
            List of augmented images (original + augmented)
        """
        augmented_images = images.copy()
        needed = max(0, target_count - current_count)
        
        # Calculate minimum increase percentage
        min_increase = int(current_count * (self.aug_config['min_increase_percentage'] / 100))
        needed = max(needed, min_increase)
        
        # Generate augmented images
        for i in range(needed):
            # Select random image from original set
            source_idx = np.random.randint(0, len(images))
            source_image = images[source_idx]
            
            # Apply augmentation
            augmented = self.augment_image(source_image)
            augmented_images.append(augmented)
        
        return augmented_images
    
    def balance_dataset(
        self,
        class_counts: Dict[int, int]
    ) -> Dict[int, int]:
        """
        Calculate augmentation requirements to balance dataset.
        
        Args:
            class_counts: Dictionary mapping class_id to current count
            
        Returns:
            Dictionary mapping class_id to target count after augmentation
        """
        augmentation_plan = {}
        
        for class_id, count in class_counts.items():
            if class_id == 6:  # Unknown class - may not need augmentation
                augmentation_plan[class_id] = count
            else:
                # Ensure minimum 30% increase
                min_target = int(count * (1 + self.aug_config['min_increase_percentage'] / 100))
                target = max(self.target_samples, min_target)
                augmentation_plan[class_id] = target
        
        return augmentation_plan


if __name__ == "__main__":
    # Example usage
    augmenter = DataAugmenter()
    
    # Test augmentation
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    augmented = augmenter.augment_image(test_image)
    
    print(f"Original shape: {test_image.shape}")
    print(f"Augmented shape: {augmented.shape}")
