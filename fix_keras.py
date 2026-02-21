import tensorflow as tf
from tensorflow.keras.models import load_model

# Load without custom objects to bypass compile/quantization errors if possible
try:
    model = load_model("models/mlp_model.keras", compile=False)
    model.save("models/mlp_model.keras")
    print("Successfully resaved model.")
except Exception as e:
    print(f"Failed to resave: {e}")
