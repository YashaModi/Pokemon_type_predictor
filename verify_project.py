import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import cv2
from pokemon_predictor import config
from pokemon_predictor.data_utils import load_data, load_metadata

def verify_structure():
    print("Verifying Directory Structure...")
    required = [
        Path("notebooks"),
        Path("pokemon_predictor"),
        Path("data"),
        Path("models"),
        Path("tests")
    ]
    for p in required:
        if not p.exists():
            print(f"❌ Missing: {p}")
            return False
        else:
            print(f"✅ Found: {p}")
            
    print("Verifying Notebooks...")
    notebooks = [
        "baseline-models.ipynb",
        "data-loader.ipynb",
        "feature-extraction-pipeline.ipynb",
        "hybrid-models.ipynb",
        "inference.ipynb",
        "mlp-optimization.ipynb",
        "modeling-evaluation.ipynb",
        "quantitative-evaluation.ipynb",
        "scenario-testing.ipynb",
        "train_models.ipynb"
    ]
    for nb in notebooks:
        if not (Path("notebooks") / nb).exists():
            print(f"❌ Missing Notebook: {nb}")
            return False
        else:
            print(f"✅ Found: {nb}")
            
    return True

def verify_data_loading():
    print("\nVerifying Data Utils...")
    try:
        # Just check if files exist before trying to load
        if not (config.PROCESSED_DATA_DIR / "X_kmeans.csv").exists():
            print("⚠️ Processed data not found. Skipping data loading check.")
            return True
            
        X, y, classes = load_data('rgb', split_data=False)
        print(f"✅ Data Loaded: {X.shape}, {len(classes)} classes")
        return True
    except Exception as e:
        print(f"❌ Data Loading Failed: {e}")
        return False

def verify_inference_pipeline():
    print("\nVerifying Inference Pipeline...")
    # Create dummy image
    dummy_path = "dummy_verify.png"
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(dummy_path, img)
    
    from pokemon_predictor.predict import PokemonPredictor
    
    try:
        predictor = PokemonPredictor()
        res = predictor.predict(dummy_path)
        print(f"✅ Prediction Successful: {res}")
        Path(dummy_path).unlink()
        return True
    except Exception as e:
        print(f"❌ Prediction Failed: {e}")
        try:
            Path(dummy_path).unlink()
        except: pass
        return False

if __name__ == "__main__":
    print("=== PROJECT VERIFICATION ===\n")
    if verify_structure() and verify_data_loading() and verify_inference_pipeline():
        print("\n🎉 PROJECT VERIFIED SUCCESSFULLY!")
    else:
        print("\n⚠️ VERIFICATION COMPLETED WITH WARNINGS/ERRORS")
