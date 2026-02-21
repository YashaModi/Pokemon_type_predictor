import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from pokemon_predictor.predict import PokemonPredictor

@pytest.fixture
def dummy_image(tmp_path):
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    p = tmp_path / "dummy_predict.png"
    cv2.imwrite(str(p), img)
    return str(p)

def test_predictor_initialization():
    predictor = PokemonPredictor()
    assert hasattr(predictor, 'xgb_model')
    assert hasattr(predictor, 'mlp_threshold')

@patch('pokemon_predictor.predict.PokemonPredictor._load_models')
@patch('pokemon_predictor.predict.extract_kmeans_features')
@patch('pokemon_predictor.predict.extract_histogram_features')
def test_predict_returns_correct_format(mock_hist, mock_kmeans, mock_load, dummy_image):
    # Setup mocks
    mock_kmeans.return_value = np.zeros(20)
    mock_hist.return_value = np.zeros(512)
    
    predictor = PokemonPredictor()
    
    # Inject XGBoost mock
    predictor.xgb_model = MagicMock()
    predictor.xgb_model.predict.return_value = np.array([[1, 0, 0]])
    
    # Inject MLB mock
    predictor.mlb = MagicMock()
    predictor.mlb.inverse_transform.return_value = [('Fire',)]
    predictor.mlb.classes_ = ['Fire', 'Water', 'Grass']
    
    # Inject single MLP mock (Top-3 combinations combinatorial logic)
    predictor.mlp_model = MagicMock()
    # Mock predicting 3 classes with decreasing probability
    predictor.mlp_model.predict.return_value = np.array([[0.8, 0.4, 0.1]])
    
    # Run prediction
    res = predictor.predict(dummy_image)
    
    # Assert format
    assert res is not None
    assert 'xgboost' in res
    assert 'mlp' in res
    assert res['xgboost'] == ('Fire',)
    
    # MLP should return a list of ranked combination tuples (max 3)
    assert isinstance(res['mlp'], list)
    assert len(res['mlp']) > 0
    assert len(res['mlp']) <= 3
    assert isinstance(res['mlp'][0], tuple)
