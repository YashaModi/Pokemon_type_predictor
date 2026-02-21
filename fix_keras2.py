import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input
import h5py

try:
    # 1. Load the broken model bypassing custom objects and compilation
    old_model = tf.keras.models.load_model("models/mlp_model.keras", compile=False)
    
    # 2. Rebuild the exact same architecture natively in this environment
    new_model = Sequential([
        Input(shape=(540,)),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(18, activation='sigmoid')
    ])
    
    # 3. Transfer weights exactly
    new_model.set_weights(old_model.get_weights())
    
    # 4. Save as stable .h5
    new_model.save("models/mlp_model_clean.h5")
    print("Successfully transferred weights and saved clean h5 model.")

except Exception as e:
    print(f"Failed to fix model: {e}")
