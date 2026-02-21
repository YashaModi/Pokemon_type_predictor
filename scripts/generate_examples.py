import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from pokemon_predictor import config
from pokemon_predictor.predict import PokemonPredictor

def generate_examples():
    df_meta = pd.read_csv(config.PROCESSED_DATA_DIR / "pokemon_metadata.csv")
    predictor = PokemonPredictor()

    xgb_examples = []
    mlp_examples = []
    
    # Shuffle predictably
    for idx, row in df_meta.sample(frac=1, random_state=42).iterrows():
        if len(xgb_examples) >= 8 and len(mlp_examples) >= 8:
            break
            
        name = row['name']
        img_path = str(config.RAW_DATA_DIR / f"{name}.png")
        
        if not Path(img_path).exists():
            continue
            
        pred = predictor.predict(img_path)
        if not pred:
            continue
            
        t1 = row['type1']
        t2 = row['type2']
        true_t = f"{t1}" + (f"+{t2}" if pd.notna(t2) else "")
        item = {'name': name, 'img_path': img_path, 'true': true_t}

        if len(xgb_examples) < 8:
            pred_xgb = "+".join(pred['xgboost']) if pred['xgboost'] else "None"
            item_xgb = item.copy()
            item_xgb['pred'] = pred_xgb
            xgb_examples.append(item_xgb)
            
        if len(mlp_examples) < 8:
            if pred['mlp']:
                top_mlp = pred['mlp'][0]
                pred_mlp = "+".join(top_mlp)
            else:
                pred_mlp = "None"
            item_mlp = item.copy()
            item_mlp['pred'] = pred_mlp
            mlp_examples.append(item_mlp)

    # Plot 4x4 grid
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Prediction Examples (Top 2 Rows: XGBoost, Bottom 2 Rows: MLP)', fontsize=22, y=1.02)
    
    for i, ax in enumerate(axes.flat):
        if i < 8:
            item = xgb_examples[i]
            prefix = "[XGB]"
        else:
            item = mlp_examples[i - 8]
            prefix = "[MLP]"
            
        img = cv2.imread(item['img_path'])
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        
        true_set = set(item['true'].split('+'))
        pred_set = set(item['pred'].split('+')) if item['pred'] != "None" else set()
        
        is_correct = (true_set == pred_set)
        color = 'green' if is_correct else 'red'
        status = "CORRECT" if is_correct else "INCORRECT"
        
        ax.set_title(f"{prefix} {item['name'].capitalize()}\nTrue: {item['true']}\nPred: {item['pred']}\n[{status}]", color=color, fontweight='bold', fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    out_path = config.FIGURES_DIR / "prediction_examples.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved 16 examples to {out_path}")

if __name__ == "__main__":
    generate_examples()
