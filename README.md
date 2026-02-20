# Pokémon Type Predictor (XGBoost vs MLP)

This project compares the performance of two machine learning models in predicting Pokémon types based on their official artwork color palettes.

## Project Structure

```
pokemon_type_predictor/
├── README.md          <- The top-level README for developers
├── data               <- Data directory
│   ├── external       <- Data from third party sources
│   ├── processed      <- Final canonical data sets
│   └── raw            <- Original immutable data dump
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks for experiments
├── pyproject.toml     <- Project configuration file
├── requirements.txt   <- Pinned dependencies
├── setup.cfg          <- Configuration file for flake8
├── tests              <- Unit tests for the package
├── pokemon_predictor  <- Source code package
│   ├── __init__.py    <- Makes pokemon_predictor a Python module
│   ├── config.py      <- Project configuration
│   ├── download.py    <- Data acquisition scripts
│   ├── features.py    <- Feature extraction logic
│   ├── images.py      <- Image handling utilities
│   ├── losses.py      <- Custom loss functions (FocalLoss)
│   ├── plots.py       <- Visualization utilities
│   ├── predict.py     <- XGBoost/MLP Inference
│   ├── predict_cnn.py <- CNN Inference
│   ├── tabular.py     <- Tabular data loading
│   └── train.py       <- Model training script
└── verify_project.py  <- Project verification script
```

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Acquire Data:**
   Run the data acquisition script:
   ```bash
   python -m pokemon_predictor.download
   ```

3. **Train Models:**
   Train and evaluate the models (XGBoost & MLP):
   ```bash
   python -m pokemon_predictor.train
   ```

4. **Run Inference:**
   ```bash
   # XGBoost/MLP
   python -m pokemon_predictor.predict <path_to_image>
   
   # CNN
   python -m pokemon_predictor.predict_cnn <path_to_image>
   ```

5. **Run Notebooks:**
   The notebooks in `notebooks/` are for exploration and prototyping:
   - `data-loader.ipynb`: Initial data exploration and verification.
   - `feature-extraction.ipynb`: Development of feature extraction logic.
   - `baseline-models.ipynb`: Training baseline models for comparison.
   - `xgboost-tuning.ipynb`: Hyperparameter optimization for XGBoost.
   - `mlp-optimization.ipynb`: Architecture search for MLP.
   - `hybrid-models.ipynb`: Training the Hybrid MLP (RGB + Histogram).
   - `cnn-transfer-learning.ipynb`: Experiments with MobileNetV2.
   - `quantitative-evaluation.ipynb`: Detailed metrics and confusion matrices.
   - `scenario-testing.ipynb`: Testing model on specific edge cases.

   To use the project code within notebooks, ensure the package is installed in editable mode (`pip install -e .`).

## Methodology

### Track A: XGBoost
- **Input:** Top 5 dominant colors (L*a*b* space converted from RGB) and their percentage coverage. Flattened vector of size 20 (5 colors * 4 features each).
- **Pipeline:** `MultiOutputClassifier(XGBClassifier)`.
- **Hypothesis:** Interpretable, fast, and robust for limited data. Good at capturing dominant color themes.

### Track B: MLP (Neural Network)
- **Input:** **Hybrid Feature Vector** (Size: 532).
    - Concatenation of Top 5 Dominant Colors (Size 20) + Flattened 3D Color Histogram (8x8x8 bins = 512 size).
- **Architecture:** 
    - `Input(532) -> Dense(512, ReLU) -> BN -> Dropout(0.4) -> Dense(256, ReLU) -> BN -> Dropout(0.3) -> Dense(18, Sigmoid)`.
- **Loss Function:** `FocalLoss` (to handle class imbalance).
- **Hypothesis:** Combining dominant colors with detailed color distribution provides the richest signal for prediction.

### Status: Experimental (code available in `predict_cnn.py` and notebooks).

## Sample Results

Here are current inputs and outputs from the trained models:

| Pokemon | Image | XGBoost Prediction | MLP Prediction (Top-3) |
| :---: | :---: | :--- | :--- |
| **Charizard** (#6) | <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/6.png" width="100"> | `('Fire', 'Flying')` | `('Flying', 'Bug', 'Fire')` |
| **Pikachu** (#25) | <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/25.png" width="100"> | `('Electric')` | `('Electric', 'Fairy')` |
| **Bulbasaur** (#1) | <img src="https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/other/official-artwork/1.png" width="100"> | `('Grass', 'Poison')` | `('Poison', 'Normal', 'Grass')` |

*Note: The MLP model is configured to output at most the **Top 3** highest probability types that cross the confidence threshold. Expanding to 3 guesses improves recall but often includes an incorrect third type since Pokemon have at most 2.*

