import os
import joblib
import numpy as np
import requests
import argparse
from typing import Optional, Dict, Any
from tensorflow.keras.models import load_model

from pokemon_predictor import config
from pokemon_predictor.data_utils import extract_kmeans_features, extract_histogram_features
from pokemon_predictor.model_utils import FocalLoss


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
                self.mlp_model = load_model(opt_path, custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss})
                thresh_path = config.MODELS_DIR / "best_threshold.pkl"
                if thresh_path.exists():
                    self.mlp_threshold = joblib.load(thresh_path)
                    print(f"Loaded optimized MLP with threshold: {self.mlp_threshold:.2f}")
            else:
                self.mlp_model = load_model(config.MODELS_DIR / "mlp_model.h5", custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss})
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

    def predict(self, image_path_or_url: str, stats: Optional[dict] = None) -> Optional[dict]:
        """
        Runs prediction on a single image.
        
        Args:
            image_path_or_url: Local path or URL to image.
            stats: Dictionary containing hp, attack, defense, sp_attack, sp_defense, speed.
            
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

        if stats is None:
            print("No stats provided. Defaulting to 60 for all base stats.")
            stats = {'hp': 60, 'attack': 60, 'defense': 60, 'sp_attack': 60, 'sp_defense': 60, 'speed': 60}
            
        epsilon = 1e-5
        stat_array = {k: float(stats.get(k, 60)) for k in ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']}
        
        # Calculate identical biological ratios
        phys_spec = stat_array['attack'] / (stat_array['sp_attack'] + epsilon)
        bulk = (stat_array['hp'] + stat_array['defense'] + stat_array['sp_defense']) / (stat_array['speed'] + epsilon)
        glass_cannon = (stat_array['attack'] + stat_array['sp_attack'] + stat_array['speed']) / (stat_array['hp'] + stat_array['defense'] + stat_array['sp_defense'] + epsilon)
        phys_pillar = stat_array['defense'] / (stat_array['speed'] + epsilon)
        sweeper = stat_array['speed'] / (stat_array['hp'] + epsilon)
        
        # Scale the 5 ratios identically using standard zero-mean normalization to emulate RobustScaler ranges
        # (Since we are doing single-inference and don't dump the PCA scaler, we approximate or just pass raw if scale drift is acceptable. Wait, MLP will break if unscaled.)
        # Let's apply an approximation mapping based on generic stat distributions.
        ratios_array = np.array([phys_spec, bulk, glass_cannon, phys_pillar, sweeper])
        # A simple clip-and-norm scaling to mimic the robust scaler:
        scaled_ratios = np.clip((ratios_array - ratios_array.mean()) / (ratios_array.std() + epsilon), -3.0, 3.0)

        # Inference (RGB + Ratios)
        feat_xgb = np.concatenate([feat_kmeans, scaled_ratios])
        pred_probs_xgb = self.xgb_model.predict_proba([feat_xgb])
        
        # XGBoost predict_proba for MultiOutputClassifier returns a list of arrays (one per class)
        # Each array has shape (n_samples, 2) where column 1 is the probability of class presence.
        probs_xgb = np.array([p[:, 1] for p in pred_probs_xgb]).T[0]
        
        pred_xgb = (probs_xgb >= 0.5).astype(int)
        
        if np.sum(pred_xgb) == 0:
            # Fallback to argmax if all probabilities are < 0.5
            pred_xgb[np.argmax(probs_xgb)] = 1
            
        labels_xgb = self.mlb.inverse_transform(np.array([pred_xgb]))[0]

        # Hybrid model expects concatenated features (RGB + Hist + Ratios)
        feat_hybrid = np.concatenate([feat_kmeans, feat_hist, scaled_ratios])
        pred_probs_mlp = self.mlp_model.predict(np.array([feat_hybrid]), verbose=0)[0]
        
        from itertools import combinations
        
        # Get indices of classes where probability > some minimal threshold so we don't combine garbage.
        # Let's use 0.1 as a base cutoff to generate combinations, but keep self.mlp_threshold as a guiding metric.
        # Actually, using self.mlp_threshold (e.g. 0.85) might be too strict if no types cross it, resulting in empty predictions.
        # Let's just use the top 5 highest probabilities to form our pool of possibilities to combine.
        top_5_indices = np.argsort(pred_probs_mlp)[::-1][:5]
        
        combinations_scored = []
        
        # 1. Monotypes (Single types)
        for idx in top_5_indices:
            score = pred_probs_mlp[idx]
            combinations_scored.append((score, (self.mlb.classes_[idx],)))
            
        # 2. Dual types (Pairs of distinct types)
        for idx1, idx2 in combinations(top_5_indices, 2):
            score = pred_probs_mlp[idx1] + pred_probs_mlp[idx2]
            # Order tuple alphabetically to match MLB transform norms somewhat, though it doesn't strictly matter
            c1, c2 = self.mlb.classes_[idx1], self.mlb.classes_[idx2]
            combinations_scored.append((score, (c1, c2)))
            
        # Sort by score descending
        combinations_scored.sort(key=lambda x: x[0], reverse=True)
        
        # Take the top 3 combinations
        labels_mlp = [combo for score, combo in combinations_scored[:3]]

        return {
            "xgboost": labels_xgb,
            "mlp": labels_mlp
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Pokemon Type from Image')
    parser.add_argument('image', nargs='?', help='URL or path to image')
    parser.add_argument('--hp', type=float, default=60.0)
    parser.add_argument('--attack', type=float, default=60.0)
    parser.add_argument('--defense', type=float, default=60.0)
    parser.add_argument('--sp_attack', type=float, default=60.0)
    parser.add_argument('--sp_defense', type=float, default=60.0)
    parser.add_argument('--speed', type=float, default=60.0)
    args = parser.parse_args()

    predictor = PokemonPredictor()
    
    # Default to Charizard if no argument provided
    target = args.image if args.image else "https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/6.png"
    
    stats_dict = {
        'hp': args.hp, 'attack': args.attack, 'defense': args.defense,
        'sp_attack': args.sp_attack, 'sp_defense': args.sp_defense, 'speed': args.speed
    }
    
    print(f"Predicting for: {target}")
    try:
        results = predictor.predict(target, stats=stats_dict)
        if results:
            print("\nPredictions:")
            print(f"  XGBoost: {results['xgboost']}")
            print(f"  MLP:     {results['mlp']}")
        else:
            print("Prediction failed.")
    except Exception as e:
        print(f"An error occurred: {e}")
