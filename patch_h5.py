import h5py
import json
import tensorflow as tf
from pokemon_predictor.model_utils import FocalLoss

try:
    with h5py.File("models/mlp_model_optimized.h5", "r+") as f:
        model_config = json.loads(f.attrs.get("model_config"))
        for layer in model_config["config"]["layers"]:
            if "quantization_config" in layer["config"]:
                del layer["config"]["quantization_config"]
        f.attrs.modify("model_config", json.dumps(model_config).encode("utf-8"))
    print("Patched the h5 file!")
    
    # Test load
    model = tf.keras.models.load_model("models/mlp_model_optimized.h5", custom_objects={'FocalLoss': FocalLoss, 'focal_loss': FocalLoss})
    print("Loaded model successfully!")
except Exception as e:
    print(f"Error: {e}")
