"""
Feature Extraction Module

Converts raw images into fixed-size numerical feature vectors.
Implements multiple feature extraction methods:
- Histogram of Oriented Gradients (HOG)
- Local Binary Patterns (LBP)
- Color Histograms
- Combined features
"""

import numpy as np
import cv2
from typing import Union, Tuple, List
import yaml
from pathlib import Path

# Try to import scikit-image, but make it optional
_SKIMAGE_WARNING_SHOWN = False

try:
    from skimage.feature import hog, local_binary_pattern  # type: ignore
    from skimage import color  # type: ignore
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    # Warning will be shown only when needed (in methods that use it)


class FeatureExtractor:
    """
    Extracts fixed-size feature vectors from images.
    
    Supports multiple feature extraction methods:
    - HOG (Histogram of Oriented Gradients)
    - LBP (Local Binary Patterns)
    - Color Histograms
    - Combined features
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize feature extractor with configuration.
        
        Args:
            config_path: Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.feature_config = self.config['feature_extraction']
        self.method = self.feature_config['method']
        
    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: Input image as numpy array (H, W, C) in RGB format
            
        Returns:
            1D feature vector
        """
        if self.method == "histogram_of_oriented_gradients":
            return self._extract_hog(image)
        elif self.method == "local_binary_pattern":
            return self._extract_lbp(image)
        elif self.method == "color_histogram":
            return self._extract_color_histogram(image)
        elif self.method == "combined":
            return self._extract_combined(image)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.method}")
    
    def _extract_hog(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        global _SKIMAGE_WARNING_SHOWN
        if not HAS_SKIMAGE and not _SKIMAGE_WARNING_SHOWN:
            print("Note: Using OpenCV HOG implementation (scikit-image not available).")
            _SKIMAGE_WARNING_SHOWN = True
        
        if len(image.shape) == 3:
            if HAS_SKIMAGE:
                gray = color.rgb2gray(image)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        else:
            gray = image
        
        hog_config = self.feature_config['hog']
        
        if HAS_SKIMAGE:
            features = hog(
                gray,
                orientations=hog_config['orientations'],
                pixels_per_cell=tuple(hog_config['pixels_per_cell']),
                cells_per_block=tuple(hog_config['cells_per_block']),
                visualize=hog_config['visualize'],
                feature_vector=True
            )
        else:
            # OpenCV HOG implementation
            win_size = (image.shape[1], image.shape[0])
            block_size = (hog_config['cells_per_block'][0] * hog_config['pixels_per_cell'][0],
                          hog_config['cells_per_block'][1] * hog_config['pixels_per_cell'][1])
            block_stride = tuple(hog_config['pixels_per_cell'])
            cell_size = tuple(hog_config['pixels_per_cell'])
            nbins = hog_config['orientations']
            
            hog_descriptor = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            features = hog_descriptor.compute(gray.astype(np.uint8)).flatten()
        
        return features
    
    def _extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features.
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            LBP feature vector (histogram)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        lbp_config = self.feature_config['lbp']
        
        if HAS_SKIMAGE:
            lbp = local_binary_pattern(
                gray,
                lbp_config['n_points'],
                lbp_config['radius'],
                method=lbp_config['method']
            )
        else:
            # Simple LBP implementation using NumPy
            radius = lbp_config['radius']
            n_points = lbp_config['n_points']
            
            # Create circular sampling points
            angles = 2 * np.pi * np.arange(n_points) / n_points
            y_offsets = np.round(radius * np.sin(angles)).astype(int)
            x_offsets = np.round(radius * np.cos(angles)).astype(int)
            
            # Pad image
            padded = np.pad(gray, radius, mode='edge')
            h, w = gray.shape
            lbp = np.zeros_like(gray, dtype=np.uint8)
            
            # Compute LBP
            for i in range(n_points):
                y_coords = np.arange(radius, h + radius)[:, None] + y_offsets[i]
                x_coords = np.arange(radius, w + radius)[None, :] + x_offsets[i]
                neighbor_values = padded[y_coords, x_coords]
                lbp |= ((neighbor_values >= gray) << i).astype(np.uint8)
        
        # Compute histogram
        n_bins = int(lbp.max()) + 1
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-7)  # Normalize
        
        return hist
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features.
        
        Args:
            image: Input image (H, W, C) in RGB format
            
        Returns:
            Color histogram feature vector
        """
        hist_config = self.feature_config['color_histogram']
        bins = hist_config['bins']
        color_space = hist_config['color_space']
        
        if color_space == "RGB":
            channels = cv2.split(image)
        elif color_space == "HSV":
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            channels = cv2.split(hsv)
        elif color_space == "LAB":
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            channels = cv2.split(lab)
        else:
            raise ValueError(f"Unknown color space: {color_space}")
        
        # Compute histogram for each channel
        histograms = []
        for channel in channels:
            hist = cv2.calcHist([channel], [0], None, [bins], [0, 256])
            hist = hist.flatten()
            hist = hist / (hist.sum() + 1e-7)  # Normalize
            histograms.append(hist)
        
        return np.concatenate(histograms)
    
    def _extract_combined(self, image: np.ndarray) -> np.ndarray:
        """
        Extract combined features (HOG + LBP + Color Histogram).
        
        Args:
            image: Input image (H, W, C)
            
        Returns:
            Combined feature vector
        """
        features = []
        combined_config = self.feature_config['combined']
        
        if combined_config['use_hog']:
            hog_features = self._extract_hog(image)
            features.append(hog_features)
        
        if combined_config['use_lbp']:
            lbp_features = self._extract_lbp(image)
            features.append(lbp_features)
        
        if combined_config['use_color_hist']:
            color_features = self._extract_color_histogram(image)
            features.append(color_features)
        
        return np.concatenate(features)
    
    def get_feature_dimension(self, image_shape: Tuple[int, int, int]) -> int:
        """
        Calculate the dimension of the feature vector for a given image shape.
        
        Args:
            image_shape: Shape of input image (H, W, C)
            
        Returns:
            Feature vector dimension
        """
        # Create dummy image to calculate feature size
        dummy_image = np.zeros(image_shape, dtype=np.uint8)
        features = self.extract_features(dummy_image)
        return len(features)


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    
    # Test feature extraction
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    features = extractor.extract_features(test_image)
    
    print(f"Image shape: {test_image.shape}")
    print(f"Feature vector shape: {features.shape}")
    print(f"Feature vector dimension: {len(features)}")
