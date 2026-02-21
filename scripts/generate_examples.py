import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from pokemon_predictor import config
from pokemon_predictor.predict import PokemonPredictor

def generate_examples():
    df_meta = pd.read_csv(config.PROCESSED_DATA_DIR / "pokemon_metadata.csv")
    predictor = PokemonPredictor()

    good_examples = []
    bad_examples = []
    
    # Let's just scan through the first few hundred to find clear 
    # examples of correct and incorrect predictions from the XGBoost baseline.
    for idx, row in df_meta.sample(frac=1, random_state=42).iterrows():
        if len(good_examples) >= 2 and len(bad_examples) >= 2:
            break
            
        name = row['name']
        img_path = str(config.RAW_DATA_DIR / f"{name}.png")
        
        # Some images might be missing in raw if download failed
        if not Path(img_path).exists():
            continue
            
        pred = predictor.predict(img_path)
        if not pred:
            continue
            
        t1 = row['type1']
        t2 = row['type2']
        true_t = f"{t1}" + (f", {t2}" if pd.notna(t2) else "")
        
        pred_xgb = ", ".join(pred['xgboost'])
        if not pred_xgb:
            pred_xgb = "None"
            
        t_set = set([t1]) if pd.isna(t2) else set([t1, t2])
        p_set = set(pred['xgboost'])
        
        item = {'name': name, 'img_path': img_path, 'true': true_t, 'pred': pred_xgb}
        
        if t_set == p_set and len(good_examples) < 2:
            good_examples.append(item)
        elif t_set != p_set and len(bad_examples) < 2:
            # Prefer cases where it predicted something, rather than just "None"
            if p_set:
                bad_examples.append(item)

    # Plot a 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle('Prediction Examples (XGBoost)', fontsize=16, y=1.02)

    # 2 good
    for i in range(2):
        if i < len(good_examples):
            item = good_examples[i]
            img = cv2.imread(item['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f"{item['name'].capitalize()}\nTrue: {item['true']}\nPred: {item['pred']}\n[CORRECT]", color='green', fontweight='bold')
            axes[0, i].axis('off')

    # 2 bad
    for i in range(2):
        if i < len(bad_examples):
            item = bad_examples[i]
            img = cv2.imread(item['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f"{item['name'].capitalize()}\nTrue: {item['true']}\nPred: {item['pred']}\n[INCORRECT]", color='red', fontweight='bold')
            axes[1, i].axis('off')

    plt.tight_layout()
    out_path = config.FIGURES_DIR / "prediction_examples.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    print(f"Saved examples to {out_path}")

if __name__ == "__main__":
    generate_examples()
