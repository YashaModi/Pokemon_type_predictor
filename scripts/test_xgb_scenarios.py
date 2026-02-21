from pokemon_predictor.tabular import load_data
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def test_scenarios():
    print("Loading Hybrid Dataset (Color + 5 Biological Ratios)...")
    X_train, X_test, y_train, y_test, classes = load_data('hybrid', split_data=True, test_size=0.2)
    
    print(f"X_train shape: {X_train.shape}")
    
    scenarios = [1.0, 0.75, 0.50, 0.25]
    results = []
    
    for penalty in scenarios:
        print(f"\n--- Testing Scenario: colsample_bytree = {penalty} ---")
        
        # Pure XGBoost test to determine baseline feature engineering success
        model = MultiOutputClassifier(XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            learning_rate=0.1, 
            colsample_bytree=penalty, 
            n_jobs=-1, 
            random_state=42
        ))
        
        print(f"Training XGBoost (Penalty: {penalty})...")
        model.fit(X_train, y_train)
        
        pred_probs = model.predict_proba(X_test)
        pred_array = np.array([p[:, 1] for p in pred_probs]).T
        preds = (pred_array > 0.5).astype(int)
        
        f1_micro = f1_score(y_test, preds, average='micro')
        prec = precision_score(y_test, preds, average='micro')
        
        print(f"Result: F1 Micro: {f1_micro:.4f} | Precision: {prec:.4f}")
        results.append({'Penalty': penalty, 'F1': f1_micro, 'Precision': prec})
        
    print("\n=== FINAL SCENARIO RANKING ===")
    df = pd.DataFrame(results).sort_values(by='F1', ascending=False)
    print(df.to_string(index=False))

if __name__ == "__main__":
    test_scenarios()
