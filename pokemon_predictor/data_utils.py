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
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pokemon_predictor import config

def load_data(
    feature_type: str = 'rgb', # 'rgb', 'lab' (if reverted?), 'hist', 'hybrid'
    split_data: bool = True,
    test_size: float = 0.2,
    random_state: int = None
) -> Tuple:
    """
    Loads features and labels, encodes labels, and optionally splits data.
    
    Args:
        feature_type: 'rgb' (KMeans), 'hist' (Histogram), 'hybrid' (Concatenated).
        split_data: Whether to return X_train, X_test, y_train, y_test.
        test_size: Fraction of test data.
        random_state: Seed for reproducibility. Defaults to config.RANDOM_SEED.
        
    Returns:
        If split_data=True: (X_train, X_test, y_train, y_test, mlb_classes)
        If split_data=False: (X, y_encoded, mlb_classes)
    """
    if random_state is None:
        random_state = config.RANDOM_SEED

    # 1. Load Features
    if feature_type == 'rgb':
        X = pd.read_csv(config.PROCESSED_DATA_DIR / "X_kmeans.csv")
    elif feature_type == 'hist':
        X = pd.read_csv(config.PROCESSED_DATA_DIR / "X_hist.csv")
    elif feature_type == 'hybrid':
        X_rgb = pd.read_csv(config.PROCESSED_DATA_DIR / "X_kmeans.csv")
        X_hist = pd.read_csv(config.PROCESSED_DATA_DIR / "X_hist.csv")
        # Ensure alignment (assuming generated in same order)
        #Ideally we join on ID, but files are aligned by index 0..889
        X = pd.concat([X_rgb, X_hist], axis=1)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    # 1.5 Load and Calculate Advanced Base Stat Ratios
    meta = pd.read_csv(config.PROCESSED_DATA_DIR / "pokemon_metadata.csv")
    y_labels_temp = pd.read_csv(config.PROCESSED_DATA_DIR / "y_labels.csv")
    stats_df = y_labels_temp[['id']].merge(meta[['id', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']], on='id', how='left')
    stats_raw = stats_df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].fillna(60)
    
    # Calculate Biological Ratios
    epsilon = 1e-5
    ratios = pd.DataFrame(index=stats_raw.index)
    ratios['phys_spec'] = stats_raw['attack'] / (stats_raw['sp_attack'] + epsilon)
    ratios['bulk'] = (stats_raw['hp'] + stats_raw['defense'] + stats_raw['sp_defense']) / (stats_raw['speed'] + epsilon)
    ratios['glass_cannon'] = (stats_raw['attack'] + stats_raw['sp_attack'] + stats_raw['speed']) / (stats_raw['hp'] + stats_raw['defense'] + stats_raw['sp_defense'] + epsilon)
    ratios['phys_pillar'] = stats_raw['defense'] / (stats_raw['speed'] + epsilon)
    ratios['sweeper'] = stats_raw['speed'] / (stats_raw['hp'] + epsilon)
    
    # Scale ratios with RobustScaler to handle extreme outliers
    from sklearn.preprocessing import RobustScaler
    import joblib
    scaler = RobustScaler()
    scaled_ratios = pd.DataFrame(scaler.fit_transform(ratios), columns=ratios.columns)
    joblib.dump(scaler, config.MODELS_DIR / "robust_scaler.pkl")
    
    X = pd.concat([X, scaled_ratios], axis=1)

    # 2. Load and Encode Labels
    y_labels = pd.read_csv(config.PROCESSED_DATA_DIR / "y_labels.csv")
    y_list = []
    for _, row in y_labels.iterrows():
        types = [row['type1']]
        if pd.notna(row['type2']):
            types.append(row['type2'])
        y_list.append(types)

    mlb = MultiLabelBinarizer()
    y_encoded = mlb.fit_transform(y_list)
    
    if not split_data:
        return X, y_encoded, mlb.classes_
        
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test, mlb.classes_

def load_metadata() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DATA_DIR / "pokemon_metadata.csv")
