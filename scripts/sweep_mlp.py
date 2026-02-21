import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from pokemon_predictor import config
from pokemon_predictor.data_utils import load_data
from pokemon_predictor.model_utils import FocalLoss

def build_model(architecture="medium", input_shape=540):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    
    if architecture == "shallow":
        model.add(Dense(1024, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    elif architecture == "medium":
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
    elif architecture == "deep":
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        
    model.add(Dense(18, activation='sigmoid'))
    return model

def run_mlp_sweep():
    print("Loading Hybrid Data (RGB + Hist + Ratios)...")
    X_train, X_test, y_train, y_test, _ = load_data('hybrid', split_data=True)
    
    architectures = ["shallow", "medium", "deep"]
    best_loss = float('inf')
    best_arch = None
    
    print("\nStarting MLP Architecture Sweep:")
    print("-" * 30)
    
    for arch in architectures:
        print(f"\nTraining Architecture: {arch.upper()}")
        model = build_model(arch, input_shape=X_train.shape[1])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                      loss=FocalLoss(), 
                      metrics=['accuracy', 'binary_accuracy'])
        
        # Modern Keras 2 formats only
        temp_path = config.MODELS_DIR / f"temp_{arch}.h5"
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=str(temp_path), monitor='val_loss', save_best_only=True, verbose=0)
        ]
        
        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=2
        )
        
        # Get the best validation loss
        min_val_loss = min(history.history['val_loss'])
        print(f"{arch.upper()} Best Val Loss: {min_val_loss:.4f}")
        
        if min_val_loss < best_loss:
            best_loss = min_val_loss
            best_arch = arch
            
    print("\n" + "="*40)
    print(f"Sweep Complete! Best Architecture: {best_arch.upper()} (Val Loss: {best_loss:.4f})")
    
    # Save the winner as the official model
    winner_path = config.MODELS_DIR / f"temp_{best_arch}.h5"
    final_path = config.MODELS_DIR / "mlp_model_clean.h5"
    
    import shutil
    shutil.copy2(str(winner_path), str(final_path))
    print(f"Saved winning architecture to {final_path}")
    
    # Clean up temp files
    for arch in architectures:
        p = config.MODELS_DIR / f"temp_{arch}.h5"
        if p.exists():
            os.remove(p)

if __name__ == "__main__":
    run_mlp_sweep()
