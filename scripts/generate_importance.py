import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pokemon_predictor import config

def generate_feature_importance():
    print("Loading XGBoost Model...")
    xgb_model = joblib.load(config.MODELS_DIR / "xgboost_model.pkl")
    
    # Define the 28 precise feature names
    feature_names = []
    for i in range(1, 6):
        feature_names.extend([f'Color_{i}_L', f'Color_{i}_A', f'Color_{i}_B', f'Color_{i}_Pct'])
        
    bio_ratios = [
        'phys_spec_ratio', 'bulk_ratio', 'glass_cannon_ratio', 
        'phys_pillar_ratio', 'sweeper_ratio', 
        'total_base_stats', 'total_physical', 'total_special'
    ]
    feature_names.extend(bio_ratios)
    
    print(f"Total defined features: {len(feature_names)}")
    
    # Extract feature importances from the MultiOutputClassifier
    # A MultiOutputClassifier holds multiple independent XGBClassifier estimators
    # We will compute the mean importance across all estimators
    importances_list = []
    for estimator in xgb_model.estimators_:
        importances_list.append(estimator.feature_importances_)
        
    avg_importances = np.mean(importances_list, axis=0)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': avg_importances
    })
    
    # Sort and take top 20
    df = df.sort_values(by='Importance', ascending=False).head(20)
    
    # Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df, x='Importance', y='Feature', palette='magma')
    
    plt.title('Top 20 Feature Importances (XGBoost Hybrid Baseline)', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Mean Gain Importance (across 18 Type Classifiers)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    save_path = config.REPORTS_DIR / "figures" / "xgboost_feature_importance.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved to: {save_path}")

if __name__ == "__main__":
    generate_feature_importance()
