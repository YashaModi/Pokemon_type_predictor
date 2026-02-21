import pytest
import numpy as np
import cv2
from pokemon_predictor.data_utils import extract_kmeans_features, extract_histogram_features

@pytest.fixture
def dummy_image(tmp_path):
    # Create a dummy 100x100 RGB image (random)
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    p = tmp_path / "dummy_pokemon.png"
    cv2.imwrite(str(p), img) # Writes BGR
    return str(p)

def test_kmeans_features_shape(dummy_image):
    k = 5
    features = extract_kmeans_features(dummy_image, k=k)
    assert features is not None
    # Output: [R, G, B, P] * k = 4 * 5 = 20
    assert features.shape == (20,)
    
def test_kmeans_features_values(dummy_image):
    features = extract_kmeans_features(dummy_image, k=5)
    # Check if proportions sum to ~1 (or close, due to float precision)
    # Proportions are at indices 3, 7, 11, 15, 19
    proportions = features[3::4]
    assert np.isclose(np.sum(proportions), 1.0, atol=0.01)

def test_histogram_features_shape(dummy_image):
    features = extract_histogram_features(dummy_image, bins=(8, 8, 8))
    assert features is not None
    assert features.shape == (512,) # 8*8*8

def test_invalid_image_path():
    features = extract_kmeans_features("non_existent.png")
    assert features is None
