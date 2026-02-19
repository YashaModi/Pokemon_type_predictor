import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

def plot_model_comparison(metrics_df: pd.DataFrame, save_path: Path = None):
    """
    Plots model comparison metrics.
    """
    sns.set_theme(style="whitegrid")
    
    # Melt for seaborn
    df_melt = metrics_df.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melt, x='Metric', y='Score', hue='Model', palette='viridis')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_feature_importance(model, feature_names=None, top_n=20, save_path: Path = None):
    """
    Plots feature importance for tree-based models (XGBoost).
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        print("Model does not have feature_importances_ attribute.")
        return

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
        
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(data=df, x='Importance', y='Feature', palette='magma')
    plt.title(f'Top {top_n} Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()

def plot_class_distribution_comparison(y_true, y_pred, classes, model_name="Model", save_path: Path = None):
    """
    Plots the distribution of predicted vs true classes.
    """
    # Sum counts per class
    true_counts = np.sum(y_true, axis=0)
    pred_counts = np.sum(y_pred, axis=0)
    
    df = pd.DataFrame({
        'Class': classes,
        'True Count': true_counts,
        'Predicted Count': pred_counts
    })
    
    df_melt = df.melt(id_vars='Class', var_name='Type', value_name='Count')
    
    plt.figure(figsize=(14, 6))
    sns.barplot(data=df_melt, x='Class', y='Count', hue='Type', palette='muted')
    plt.title(f'Class Distribution: Truth vs {model_name} Prediction')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Figure saved to {save_path}")
        plt.close()
    else:
        plt.show()
