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
    
    # Shuffle randomly
    for idx, row in df_meta.sample(frac=1, random_state=None).iterrows():
        if len(xgb_examples) >= 8 and len(mlp_examples) >= 8:
            break
            
        name = row['name']
        img_path = str(config.RAW_DATA_DIR / f"{name}.png")
        
        if not Path(img_path).exists():
            continue
            
        stats = {
            'hp': row['hp'],
            'attack': row['attack'],
            'defense': row['defense'],
            'sp_attack': row['sp_attack'],
            'sp_defense': row['sp_defense'],
            'speed': row['speed']
        }
        
        pred = predictor.predict(img_path, stats=stats)
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
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_bgr)
        
        true_types = item['true'].split('+')
        
        # Always run predictions
        pred = predictor.predict(item['img_path'])
        pred_xgb = pred["xgboost"]
        
        # Format prediction text (XGBoost Only)
        true_text = f"True: {', '.join(true_types)}"
        xgb_text = f"XGB: {', '.join(pred_xgb)}"
        
        # Add a subtle dark backing panel for text readability
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (224, 60), (0, 0, 0), -1)
        alpha = 0.6
        img_bgr = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
        
        # Determine exact match color for XGBoost
        # Sort both tuples to ensure order doesn't fail the match
        pred_sorted = tuple(sorted(pred_xgb))
        true_sorted = tuple(sorted(true_types))
        
        if pred_sorted == true_sorted:
            color_xgb = (0, 255, 0) # Green for perfect match
        elif set(pred_xgb).intersection(set(true_types)):
            color_xgb = (0, 255, 255) # Yellow for partial match
        else:
            color_xgb = (0, 0, 255) # Red for complete miss
            
        cv2.putText(img_bgr, true_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(img_bgr, xgb_text, (5, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_xgb, 1)
        
        ax.imshow(img_bgr) # Display the image with text overlay
        ax.set_title(f"{prefix} {item['name'].capitalize()}", fontsize=12) # Only name and prefix in title
        ax.axis('off')

    plt.tight_layout()
    out_path = config.FIGURES_DIR / "prediction_examples.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved 16 examples to {out_path}")

if __name__ == "__main__":
    generate_examples()
