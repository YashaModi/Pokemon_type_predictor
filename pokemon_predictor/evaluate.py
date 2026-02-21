import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tensorflow.keras.models import load_model

from pokemon_predictor import config
from pokemon_predictor.data_utils import load_data
from pokemon_predictor.model_utils import FocalLoss

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
        
        # Calculate Partial Accuracy
        partial_matches = 0
        for true, pred in zip(y_test, y_pred):
            if np.sum(np.logical_and(true, pred)) > 0:
                partial_matches += 1
        acc_partial = partial_matches / len(y_test)
        
        print(f"Exact Match Accuracy: {acc:.4f}")
        print(f"Partial Match Accuracy: {acc_partial:.4f}")
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
        
        from itertools import combinations
        
        # Apply Top-3 Combinatorial logic per sample
        # Since the output is now a list of tuples, calculating standard sklearn metrics is tricky for the 'Top-3' case.
        # We will calculate two things:
        # 1. Top-1 F1/Accuracy (treating the highest scoring combination as the absolute prediction)
        # 2. Top-3 Any-Match Accuracy (is the true combination ANYWHERE in the top 3?)
        
        y_pred_top1 = np.zeros_like(y_pred_probs, dtype=int)
        y_pred_top3_any_match = 0
        
        for i in range(len(y_pred_probs)):
            probs = y_pred_probs[i]
            
            top_5_indices = np.argsort(probs)[::-1][:5]
            combinations_scored = []
            
            for idx in top_5_indices:
                combinations_scored.append((probs[idx], (idx,)))
                
            for idx1, idx2 in combinations(top_5_indices, 2):
                combinations_scored.append((probs[idx1] + probs[idx2], (idx1, idx2)))
                
            combinations_scored.sort(key=lambda x: x[0], reverse=True)
            top_3_combos_indices = [combo for score, combo in combinations_scored[:3]]
            
            # 1. Top-1 logic for F1/Acc
            if len(top_3_combos_indices) > 0:
                y_pred_top1[i, list(top_3_combos_indices[0])] = 1
                
            # 2. Top-3 Any-Match logic
            true_indices = np.where(y_test[i] == 1)[0]
            true_combo = tuple(sorted(true_indices))
            
            # Sort predicted indices for fair tuple comparison
            preds_sorted = [tuple(sorted(c)) for c in top_3_combos_indices]
            
            if true_combo in preds_sorted:
                y_pred_top3_any_match += 1
                
        
        acc_top1 = accuracy_score(y_test, y_pred_top1)
        acc_top3 = y_pred_top3_any_match / len(y_test)
        
        # Calculate Partial Accuracy for MLP Top-1
        partial_matches_mlp = 0
        for true, pred in zip(y_test, y_pred_top1):
            if np.sum(np.logical_and(true, pred)) > 0:
                partial_matches_mlp += 1
        acc_partial_mlp = partial_matches_mlp / len(y_test)
        
        f1_micro = f1_score(y_test, y_pred_top1, average='micro')
        f1_macro = f1_score(y_test, y_pred_top1, average='macro')
        
        print(f"Top-1 Exact Match Accuracy: {acc_top1:.4f}")
        print(f"Top-1 Partial Match Accuracy: {acc_partial_mlp:.4f}")
        print(f"Top-3 Accuracy (Any Exact Match): {acc_top3:.4f}")
        print(f"Top-1 F1 Score (Micro): {f1_micro:.4f}")
        print(f"Top-1 F1 Score (Macro): {f1_macro:.4f}")
        print("\nClassification Report (Based on Top-1 Prediction):")
        print(classification_report(y_test, y_pred_top1, target_names=classes, zero_division=0))
        
    except Exception as e:
        print(f"MLP evaluation failed: {e}")

if __name__ == "__main__":
    evaluate_models()
