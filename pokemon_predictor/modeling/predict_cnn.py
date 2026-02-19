import tensorflow as tf
import cv2
import numpy as np
import joblib
import pandas as pd
from pathlib import Path
from pokemon_predictor import config
from pokemon_predictor.modeling.train_hybrid import FocalLoss # Custom object

from typing import Optional, Dict

def predict_cnn(image_path_or_name: str, threshold: float = 0.5) -> Optional[Dict[str, float]]:
    # Load Model
    model_path = config.MODELS_DIR / "cnn_mobilenet.keras"
    if not model_path.exists():
        print("Model not found. Run train_cnn.py first.")
        return
    
    # Custom Objects
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load Classes
    classes_path = config.MODELS_DIR / "cnn_classes.pkl"
    if not classes_path.exists():
        print("Classes file not found.")
        return
    classes = joblib.load(classes_path)
    
    # Load Image
    img_path = Path(image_path_or_name)
    if not img_path.exists():
        # Try raw dir
        possible_path = config.RAW_DATA_DIR / f"{image_path_or_name}.png"
        if possible_path.exists():
            img_path = possible_path
        else:
            print(f"Image not found: {image_path_or_name}")
            return

    img = cv2.imread(str(img_path))
    if img is None:
        print("Failed to load image.")
        return
    
    # Preprocess
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, config.IMG_SIZE) # (100, 100)
    img = img.astype('float32')
    
    # Batch dimension
    img_batch = np.expand_dims(img, axis=0) # (1, 100, 100, 3)
    
    # Predict
    preds = model.predict(img_batch, verbose=0)[0]
    
    # Decode
    results = {}
    for i, prob in enumerate(preds):
        if prob > threshold:
            results[classes[i]] = float(prob)
            
    # Sort by confidence
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    print(f"Predictions for {img_path.name}:")
    if not sorted_results:
        print("No types detected above threshold.")
    else:
        for cls, prob in sorted_results.items():
            print(f"  {cls}: {prob:.4f}")
            
    return sorted_results

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        predict_cnn(sys.argv[1])
    else:
        print("Usage: python -m pokemon_predictor.modeling.predict_cnn <image_name_or_path>")
