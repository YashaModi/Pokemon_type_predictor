import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import load_model

from pokemon_predictor import config
from pokemon_predictor.tabular import load_data
from pokemon_predictor.losses import FocalLoss

def evaluate_models():
    print("=== MODEL EVALUATION REPORT ===")
    
    # 1. Evaluate XGBoost
    print("\n--- XGBoost Evaluation ---")
    try:
        X_train, X_test, y_train, y_test, classes = load_data('rgb', split_data=True)
        xgb_model = joblib.load(config.MODELS_DIR / "xgboost_model.pkl")
        
        y_pred = xgb_model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"Accuracy (Subset): {acc:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
        
    except Exception as e:
        print(f"XGBoost evaluation failed: {e}")

    # 2. Evaluate MLP (Hybrid)
    print("\n--- MLP (Hybrid) Evaluation ---")
    try:
        X_train, X_test, y_train, y_test, classes = load_data('hybrid', split_data=True)
        
        mlp_path = config.MODELS_DIR / "mlp_model_optimized.h5"
        if not mlp_path.exists():
            print("Optimized MLP model not found.")
            return

        mlp_model = load_model(mlp_path, custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss})
        
        # Load threshold
        try:
            threshold = joblib.load(config.MODELS_DIR / "best_threshold.pkl")
        except:
            threshold = 0.5
        print(f"Using Threshold: {threshold:.2f}")

        y_pred_probs = mlp_model.predict(X_test, verbose=0)
        
        # Apply Top-3 logic per sample
        y_pred = np.zeros_like(y_pred_probs, dtype=int)
        for i in range(len(y_pred_probs)):
            probs = y_pred_probs[i]
            passing = np.where(probs > threshold)[0]
            if len(passing) > 0:
                passing_probs = probs[passing]
                sorted_idx_relative = np.argsort(passing_probs)[::-1]
                top_indices = passing[sorted_idx_relative][:3]
                y_pred[i, top_indices] = 1
        
        acc = accuracy_score(y_test, y_pred)
        f1_micro = f1_score(y_test, y_pred, average='micro')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        print(f"Accuracy (Subset): {acc:.4f}")
        print(f"F1 Score (Micro): {f1_micro:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))
        
    except Exception as e:
        print(f"MLP evaluation failed: {e}")

if __name__ == "__main__":
    evaluate_models()
