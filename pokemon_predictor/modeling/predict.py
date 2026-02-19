import os
import joblib
import numpy as np
import requests
import argparse
from typing import Optional, Dict, Any
from tensorflow.keras.models import load_model

from pokemon_predictor import config
from pokemon_predictor.features import extract_kmeans_features, extract_histogram_features


class PokemonPredictor:
    def __init__(self) -> None:
        self.xgb_model = None
        self.mlp_threshold = 0.5
        self._load_models()

    def _load_models(self) -> None:
        """Loads trained models from disk."""
        try:
            self.xgb_model = joblib.load(config.MODELS_DIR / "xgboost_model.pkl")
            self.mlb = joblib.load(config.MODELS_DIR / "mlb.pkl")
            
            # Try loading optimized MLP
            opt_path = config.MODELS_DIR / "mlp_model_optimized.h5"
            if opt_path.exists():
                self.mlp_model = load_model(opt_path)
                thresh_path = config.MODELS_DIR / "best_threshold.pkl"
                if thresh_path.exists():
                    self.mlp_threshold = joblib.load(thresh_path)
                    print(f"Loaded optimized MLP with threshold: {self.mlp_threshold:.2f}")
            else:
                self.mlp_model = load_model(config.MODELS_DIR / "mlp_model.h5")
                print("Loaded baseline MLP.")
                
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Ensure models are trained and saved in the 'models/' directory.")

    def download_image(self, url: str, save_path: str) -> bool:
        """Helper to download image for inference."""
        if os.path.exists(save_path):
            return True
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
        except Exception as e:
            print(f"Error downloading: {e}")
            return False
        return False

    def predict(self, image_path_or_url: str) -> Optional[dict]:
        """
        Runs prediction on a single image.
        
        Args:
            image_path_or_url: Local path or URL to image.
            
        Returns:
            Dictionary with 'xgboost' and 'mlp' predictions, or None if failed.
        """
        if self.xgb_model is None:
            print("Models not loaded.")
            return

        # Handle URL
        if image_path_or_url.startswith("http"):
            filename = image_path_or_url.split("/")[-1]
            temp_dir = config.DATA_DIR / "temp_inference"
            os.makedirs(temp_dir, exist_ok=True)
            local_path = temp_dir / filename
            if not self.download_image(image_path_or_url, str(local_path)):
                print("Failed to download image.")
                return
            img_path = str(local_path)
        else:
            img_path = image_path_or_url

        # Feature Extraction
        feat_kmeans = extract_kmeans_features(img_path)
        feat_hist = extract_histogram_features(img_path)
        
        if feat_kmeans is None or feat_hist is None:
            print("Feature extraction failed.")
            return

        # Inference
        pred_xgb = self.xgb_model.predict([feat_kmeans])
        labels_xgb = self.mlb.inverse_transform(pred_xgb)[0]

        pred_probs_mlp = self.mlp_model.predict(np.array([feat_hist]), verbose=0)
        pred_mlp = (pred_probs_mlp > self.mlp_threshold).astype(int)
        labels_mlp = self.mlb.inverse_transform(pred_mlp)[0]

        return {
            "xgboost": labels_xgb,
            "mlp": labels_mlp
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Pokemon Type from Image')
    parser.add_argument('image', nargs='?', help='URL or path to image')
    args = parser.parse_args()

    predictor = PokemonPredictor()
    
    # Default to Charizard if no argument provided
    # Using a shorter URL to avoid line length issues if possible, or just keeping it.
    target = args.image if args.image else "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/6.png"
    
    print(f"Predicting for: {target}")
    try:
        results = predictor.predict(target)
        if results:
            print("\nPredictions:")
            print(f"  XGBoost: {results['xgboost']}")
            print(f"  MLP:     {results['mlp']}")
        else:
            print("Prediction failed.")
    except Exception as e:
        print(f"An error occurred: {e}")
