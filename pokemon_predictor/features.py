import cv2
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional, Tuple

from pokemon_predictor import config


def extract_kmeans_features(img_path: str, k: int = 5) -> Optional[np.ndarray]:
    """
    Extracts dominant colors from an image using K-Means clustering in CIELAB space.

    Args:
        img_path (str): Path to the image file.
        k (int, optional): Number of clusters. Defaults to 5.

    Returns:
        Optional[np.ndarray]: Flattened array [L1, A1, B1, P1, ... Lk, Ak, Bk, Pk]
    """
    try:
        # Load image (BGR)
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, config.IMG_SIZE)
        
        # Reshape to list of pixels
        pixels = img.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and counts
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        counts = np.bincount(labels)
        percentages = counts / len(pixels)
        
        # Sort by percentage (descending)
        sorted_indices = np.argsort(percentages)[::-1]
        sorted_colors = colors[sorted_indices]
        sorted_percentages = percentages[sorted_indices]
        
        # Flatten: [L1, A1, B1, Pct1, ... Lk, Ak, Bk, Pctk]
        feature_vector = []
        for i in range(k):
            feature_vector.extend(sorted_colors[i])
            feature_vector.append(sorted_percentages[i])
            
        return np.array(feature_vector)
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def extract_histogram_features(img_path: str, bins: Tuple[int, int, int] = (8, 8, 8)) -> Optional[np.ndarray]:
    """
    Extracts a 3D color histogram from an image.

    Args:
        img_path (str): Path to the image file.
        bins (Tuple[int, int, int], optional): Number of bins for R, G, B channels. Defaults to (8, 8, 8).

    Returns:
        Optional[np.ndarray]: A flattened normalized histogram array. Returns None if image processing fails.
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate Histogram
        # channels [0, 1, 2], mask None, histSize bins, ranges [0, 256, 0, 256, 0, 256]
        hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
        
        # Normalize
        cv2.normalize(hist, hist)
        
        return hist.flatten()
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None
