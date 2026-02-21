import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pokemon_predictor.tabular import load_data

@pytest.fixture
def mock_csvs(tmp_path):
    # Mock X_kmeans.csv
    x_kmeans = pd.DataFrame(np.random.rand(10, 20), columns=[f'feat_{i}' for i in range(20)])
    x_hist = pd.DataFrame(np.random.rand(10, 512), columns=[f'hist_{i}' for i in range(512)])
    
    # Mock y_labels.csv
    y_labels = pd.DataFrame({
        'id': range(1, 11),
        'type1': ['Fire'] * 5 + ['Water'] * 5,
        'type2': [None] * 10
    })
    
    return x_kmeans, x_hist, y_labels

def test_load_data_rgb(mock_csvs):
    x_k, _, y = mock_csvs
    
    with patch('pokemon_predictor.tabular.pd.read_csv') as mock_read:
        # Configure mock to return our dataframes
        # Order of calls: X_kmeans, y_labels
        mock_read.side_effect = [x_k, y]
        
        X_train, X_test, y_train, y_test, classes = load_data('rgb', split_data=True, test_size=0.2)
        
        assert len(classes) == 2 # Fire, Water
        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert X_train.shape[1] == 20

def test_load_data_hybrid(mock_csvs):
    x_k, x_h, y = mock_csvs
    
    with patch('pokemon_predictor.tabular.pd.read_csv') as mock_read:
        # Order: X_kmeans, X_hist, y_labels
        mock_read.side_effect = [x_k, x_h, y]
        
        X, y_enc, classes = load_data('hybrid', split_data=False)
        
        assert X.shape == (10, 532) # 20 + 512
        assert y_enc.shape[0] == 10
