import sys
try:
    import pokemon_predictor
    import pokemon_predictor.config
    import pokemon_predictor.features
    from pokemon_predictor.data_pipeline.download import fetch_pokemon_metadata
    import tensorflow as tf
    import xgboost as xgb
    print(f"TensorFlow version: {tf.__version__}")
    print(f"XGBoost version: {xgb.__version__}")
    print("SUCCESS: pokemon_predictor package is installed and importable.")
except ImportError as e:
    print(f"FAILURE: Could not import pokemon_predictor. {e}")
    sys.exit(1)
except Exception as e:
    print(f"FAILURE: An error occurred during verification. {e}")
    sys.exit(1)
