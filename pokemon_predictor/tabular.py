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

    # 1.5 Load and Append Base Stats
    meta = pd.read_csv(config.PROCESSED_DATA_DIR / "pokemon_metadata.csv")
    y_labels_temp = pd.read_csv(config.PROCESSED_DATA_DIR / "y_labels.csv")
    stats_df = y_labels_temp[['id']].merge(meta[['id', 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']], on='id', how='left')
    stats_raw = stats_df[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']].fillna(60)
    
    def bin_stat(val):
        if val < 50: return 0.0
        if val < 90: return 0.5
        return 1.0
        
    stats_features = stats_raw.map(bin_stat) if hasattr(stats_raw, 'map') else stats_raw.applymap(bin_stat)
    X = pd.concat([X, stats_features], axis=1)

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
