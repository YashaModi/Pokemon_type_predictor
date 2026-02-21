import tensorflow as tf
from pokemon_predictor.model_utils import FocalLoss

try:
    print("Trying to load mlp_model.keras...")
    model = tf.keras.models.load_model("models/mlp_model.keras", custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss}, compile=False)
    print("Loaded successfully!")
    model.save_weights("models/mlp_weights.h5")
    print("Saved weights to models/mlp_weights.h5")
except Exception as e:
    print(f"Error: {e}")
