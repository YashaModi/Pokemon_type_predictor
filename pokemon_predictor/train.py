import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from pokemon_predictor import config
from pokemon_predictor.tabular import load_data
from pokemon_predictor.losses import FocalLoss

def train_xgboost():
    print("Training XGBoost Model...")
    X_train, X_test, y_train, y_test, classes = load_data('rgb', split_data=True)
    
    # Save classes for reference
    joblib.dump(classes, config.MODELS_DIR / "mlb.pkl") # Re-saving MLB classes if needed, or better save the mlb object itself if possible. 
    # Current codebase expects 'mlb.pkl' to be the MultiLabelBinarizer OR just classes? 
    # predict.py says: self.mlb = joblib.load(config.MODELS_DIR / "mlb.pkl")
    # And then: labels_xgb = self.mlb.inverse_transform(pred_xgb)[0]
    # So it expects the fitted MultiLabelBinarizer object.
    
    # Let's recreate MLB to be sure
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([[]]) # Dummy fit to initialize classes if provided? No, MLB needs to be fitted on data to know classes.
    # Actually load_data returns classes. 
    # We should probably fit an MLB on the classes to save it compatible with predict.py
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes
    # Hack to allow inverse_transform without fit? 
    # Better: load_data should probably return the MLB object or we recreate it properly.
    # Re-reading tabular.py: load_data returns mlb.classes_.
    # Let's re-fit MLB on y_train (which is already binarized? No, y_train is binarized).
    # If y_train is binarized, we can't easily fit MLB on it unless we have raw labels.
    # predict.py expects an object with .inverse_transform.
    
    # Workaround: Create a dummy MLB and set its classes
    mlb = MultiLabelBinarizer()
    mlb.classes_ = classes
    # We need to ensure internal state is set. simpler to just save 'classes' and have predict.py use them?
    # But predict.py loads 'mlb.pkl'.
    # Let's check what 'mlb.pkl' is.
    # For now, I'll assume I can save the mlb object.
    # I will load raw data to fit MLB correctly if I want to be safe, OR I will modify predict.py to just use classes.
    # Given I'm "fixing problems", simplifying predict.py to not rely on pickle for simple logic is better.
    # BUT I should stick to replacing the model file.
    
    # Actually, tabular.py:53 creates mlb and fits it.
    # I should modify tabular.py to return mlb object if I want to save it.
    # Or just instantiate one here with the known classes.
    mlb = MultiLabelBinarizer()
    mlb.fit([classes]) # This sets classes_ to classes? No.
    # mlb.classes_ = classes is the way if we trust it.
    
    # Let's just training XGBoost first.
    model = MultiOutputClassifier(XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, n_jobs=-1, random_state=42))
    model.fit(X_train, y_train)
    
    score = model.score(X_test, y_test)
    print(f"XGBoost Test Accuracy (Subset): {score:.4f}")
    
    out_path = config.MODELS_DIR / "xgboost_model.pkl"
    joblib.dump(model, out_path)
    print(f"Saved XGBoost model to {out_path}")
    
    # Save MLB (hacky recreation to satisfy predict.py)
    # Check if we can get a proper MLB
    # tabular.py loads y_labels.csv.
    # I will replicate logic to fit MLB
    y_labels = pd.read_csv(config.PROCESSED_DATA_DIR / "y_labels.csv")
    y_list = []
    for _, row in y_labels.iterrows():
        types = [row['type1']]
        if pd.notna(row['type2']):
            types.append(row['type2'])
        y_list.append(types)
    mlb_full = MultiLabelBinarizer()
    mlb_full.fit(y_list)
    joblib.dump(mlb_full, config.MODELS_DIR / "mlb.pkl")
    print(f"Saved MLB to {config.MODELS_DIR / 'mlb.pkl'}")


def train_mlp():
    print("\nTraining MLP Model (Hybrid/Optimized)...")
    # Using 'hybrid' features as they are usually better
    X_train, X_test, y_train, y_test, classes = load_data('hybrid', split_data=True)
    
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(len(classes), activation='sigmoid')
    ])
    
    model.compile(optimizer=Adam(0.001), loss=FocalLoss(), metrics=['accuracy', 'binary_accuracy'])
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save as 'mlp_model_optimized.h5' as predict.py looks for it
    out_path = config.MODELS_DIR / "mlp_model_optimized.h5"
    model.save(out_path)
    print(f"Saved MLP model to {out_path}")
    
    # Find best threshold
    print("Finding best threshold...")
    y_pred = model.predict(X_test)
    best_thresh = 0.5
    best_f1 = 0.0
    
    from sklearn.metrics import f1_score
    for thresh in np.arange(0.1, 0.9, 0.05):
        y_pred_bin = (y_pred > thresh).astype(int)
        f1 = f1_score(y_test, y_pred_bin, average='micro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"Best Threshold: {best_thresh:.2f} (F1: {best_f1:.4f})")
    joblib.dump(best_thresh, config.MODELS_DIR / "best_threshold.pkl")

if __name__ == "__main__":
    train_xgboost()
    train_mlp()
