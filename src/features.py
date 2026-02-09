import cv2
import numpy as np
import os
import requests
from sklearn.cluster import KMeans
from typing import Optional, List, Tuple, Union

IMG_SIZE = (100, 100)

def extract_kmeans_features(img_path: str, k: int = 5) -> Optional[np.ndarray]:
    """
    Extracts dominant colors from an image using K-Means clustering.

    Args:
        img_path (str): Path to the image file.
        k (int, optional): Number of clusters (dominant colors). Defaults to 5.

    Returns:
        Optional[np.ndarray]: A flattened array containing [R, G, B, Percentage] for each of the k clusters,
                              sorted by percentage in descending order. Returns None if image processing fails.
    """
    try:
        # Load image (BGR)
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize
        img = cv2.resize(img, IMG_SIZE)
        
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
        
        # Flatten: [R1, G1, B1, Pct1, ... R5, G5, B5, Pct5]
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

def download_image_from_url(url: str, save_path: str) -> bool:
    """
    Downloads an image from a URL to a specified path.

    Args:
        url (str): The URL of the image.
        save_path (str): Local path to save the image.

    Returns:
        bool: True if download was successful or file already exists, False otherwise.
    """
    if os.path.exists(save_path):
        return True
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"Error downloading: {e}")
        return False
