import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report

from pokemon_predictor import config
from pokemon_predictor.data_utils import load_data
from tensorflow.keras.models import load_model

def generate_model_comparison():
    print("Generating Model Comparison Bar Chart...")
    # These are our finalized empirical metrics from the 28-dim XGBoost and 540-dim MLP
    data = {
        'Model': ['XGBoost (Hybrid 28-dim)', 'XGBoost (Hybrid 28-dim)', 'XGBoost (Hybrid 28-dim)', 
                  'MLP (Hybrid 540-dim)', 'MLP (Hybrid 540-dim)', 'MLP (Hybrid 540-dim)'],
        'Metric': ['Exact Match (%)', 'Partial Match (%)', 'F1 Micro Score', 
                   'Exact Match (%)', 'Partial Match (%)', 'F1 Micro Score'],
        'Score': [44.55, 90.45, 74.25, 
                  0.61, 21.36, 12.46] # Scaled to 100 for percentage visualization
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=df, x='Metric', y='Score', hue='Model', palette=['#1f77b4', '#ff7f0e'])
    plt.title('Final Model Performance Comparison (Hold-Out Test Set)', fontsize=16, fontweight='bold', pad=15)
    plt.ylabel('Score (out of 100)', fontsize=12)
    plt.xlabel('Evaluation Metric', fontsize=12)
    plt.ylim(0, 100)
    
    # Add data labels
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.tight_layout()
    save_path = config.REPORTS_DIR / "figures" / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")

def generate_class_distributions_and_f1():
    print("Generating Class Distributions and Per-Class F1 Scores...")
    # Load Models
    xgb_model = joblib.load(config.MODELS_DIR / "xgboost_model.pkl")
    mlp_model = load_model(config.MODELS_DIR / "mlp_model_clean.h5", compile=False)
    
    # Load Data
    X_train_xgb, X_test_xgb, y_train, y_test, classes = load_data('rgb', split_data=True)
    X_train_mlp, X_test_mlp, _, _, _ = load_data('hybrid', split_data=True)
    
    # --- XGBoost Predictions ---
    pred_probs_xgb = xgb_model.predict_proba(X_test_xgb)
    probs_xgb_all = np.array([p[:, 1] for p in pred_probs_xgb]).T
    y_pred_xgb = np.zeros_like(probs_xgb_all, dtype=int)
    for i in range(len(probs_xgb_all)):
        probs = probs_xgb_all[i]
        valid_indices = np.where(probs >= 0.5)[0]
        if len(valid_indices) == 0:
            y_pred_xgb[i, np.argmax(probs)] = 1
        elif len(valid_indices) > 2:
            top_2_indices = np.argsort(probs)[-2:]
            y_pred_xgb[i, top_2_indices] = 1
        else:
            y_pred_xgb[i, valid_indices] = 1
            
    # --- MLP Predictions ---
    y_pred_probs_mlp = mlp_model.predict(X_test_mlp, verbose=0)
    y_pred_mlp = np.zeros_like(y_pred_probs_mlp, dtype=int)
    from itertools import combinations
    for i in range(len(y_pred_probs_mlp)):
        probs = y_pred_probs_mlp[i]
        top_5_indices = np.argsort(probs)[::-1][:5]
        combinations_scored = []
        for idx in top_5_indices:
            combinations_scored.append((probs[idx], (idx,)))
        for idx1, idx2 in combinations(top_5_indices, 2):
            combinations_scored.append((probs[idx1] + probs[idx2], (idx1, idx2)))
        combinations_scored.sort(key=lambda x: x[0], reverse=True)
        if combinations_scored:
            top_combo = combinations_scored[0][1]
            y_pred_mlp[i, list(top_combo)] = 1

    # --- Plot: Class Distribution XGBoost ---
    true_counts = np.sum(y_test, axis=0)
    xgb_counts = np.sum(y_pred_xgb, axis=0)
    mlp_counts = np.sum(y_pred_mlp, axis=0)
    
    def plot_dist(counts, title, filename, color):
        df = pd.DataFrame({
            'Class': classes,
            'True Count': true_counts,
            'Predicted Count': counts
        })
        df_melt = df.melt(id_vars='Class', var_name='Type', value_name='Count')
        plt.figure(figsize=(14, 6))
        sns.barplot(data=df_melt, x='Class', y='Count', hue='Type', palette=['#34495e', color])
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.xticks(rotation=45)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Pokemon Type', fontsize=12)
        plt.tight_layout()
        save_path = config.REPORTS_DIR / "figures" / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {save_path}")

    plot_dist(xgb_counts, "Class Distribution: Ground Truth vs XGBoost Predictions", "class_dist_xgboost.png", "#2ecc71")
    plot_dist(mlp_counts, "Class Distribution: Ground Truth vs Neural Network Predictions", "class_dist_mlp.png", "#e74c3c")
    
    # --- Plot: Per-Class F1 Score (XGBoost) ---
    print("Generating Per-Class F1 Score Chart for XGBoost...")
    report = classification_report(y_test, y_pred_xgb, target_names=classes, output_dict=True, zero_division=0)
    
    f1_scores = []
    for cls in classes:
        f1_scores.append(report[cls]['f1-score'])
        
    df_f1 = pd.DataFrame({'Class': classes, 'F1_Score': f1_scores})
    df_f1 = df_f1.sort_values(by='F1_Score', ascending=False)
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_f1, x='Class', y='F1_Score', palette='mako')
    plt.title('XGBoost: F1 Score per Pokemon Type', fontsize=16, fontweight='bold', pad=15)
    plt.xticks(rotation=45)
    plt.ylabel('F1 Score (0.0 to 1.0)', fontsize=12)
    plt.xlabel('Pokemon Type', fontsize=12)
    plt.ylim(0, 1.05)
    
    # Add data labels
    for p in plt.gca().patches:
        plt.gca().annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points')

    plt.tight_layout()
    save_path = config.REPORTS_DIR / "figures" / "xgboost_per_class_f1.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")

if __name__ == "__main__":
    generate_model_comparison()
    generate_class_distributions_and_f1()
