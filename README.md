# Pokémon Type Predictor (XGBoost vs MLP)

This project compares the performance of two machine learning models in predicting Pokémon types based on their official artwork color palettes.

## Project Structure

```
pokemon_type_predictor/
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers
├── data               <- Data directory
│   ├── external       <- Data from third party sources
│   ├── processed      <- Final canonical data sets
│   └── raw            <- Original immutable data dump
├── models             <- Trained and serialized models
├── notebooks          <- Jupyter notebooks for experiments
│   ├── baseline-models.ipynb
│   ├── xgboost-tuning.ipynb
│   ├── mlp-optimization.ipynb
│   ├── feature-extraction-pipeline.ipynb
│   ├── hybrid-models.ipynb
│   ├── cnn-transfer-learning.ipynb
│   ├── quantitative-evaluation.ipynb
│   └── scenario-testing.ipynb
├── pyproject.toml     <- Project configuration file
├── reports            <- Generated analysis
│   └── figures        <- Generated graphics
├── requirements.txt   <- Pinned dependencies for reproducibility
├── setup.cfg          <- Configuration file for flake8
├── tests              <- Unit tests for the package
└── pokemon_predictor  <- Source code package
    ├── __init__.py    <- Makes pokemon_predictor a Python module
    ├── config.py      <- Project configuration
    ├── dataset.py     <- Data acquisition scripts
    ├── features.py    <- Feature extraction logic
    ├── modeling       <- Modeling scripts
    │   ├── predict.py     <- XGBoost/MLP Inference
    │   └── predict_cnn.py <- CNN Inference
    └── plots.py       <- Visualization utilities
├── verify_project.py  <- Project verification script
```

## Setup

1. **Install Dependencies:**
   ```bash
   make requirements
   ```

2. **Acquire Data:**
   Run the data acquisition script:
   ```bash
   make data
   ```
   Or run `python -m pokemon_predictor.dataset.py`.

3. **Train Models:**
   Train and evaluate the models:
   ```bash
   make train
   ```
   Or run `python -m pokemon_predictor.modeling.train`.

4. **Run Notebooks:**
   The notebooks in `notebooks/` are for exploration and prototyping.
   - `1.0-data-loader.ipynb`: Initial data exploration.
   - `2.0-feature-extraction.ipynb`: Feature engineering steps.
   - `3.0-modeling-evaluation.ipynb`: detailed modeling analysis.
   - `4.0-inference.ipynb`: Interactive inference.

   To use the project code within notebooks, ensure the package is installed in editable mode (`pip install -e .`).

## Methodology

### Track A: XGBoost
- **Input:** Top 5 dominant colors (R, G, B) and their percentage coverage (Size: 20).
- **Pipeline:** `MultiOutputClassifier(XGBClassifier)`.
- **Hypothesis:** Interpretable, fast and robust for limited data.

### Track B: MLP (Neural Network)
- **Input:** Flattened 3D Color Histogram (8x8x8 bins = 512 size).
- **Architecture:** `Dense(128) -> Dropout(0.3) -> Dense(64) -> Dense(18, Sigmoid)`.
- **Hypothesis:** Captures richer color distribution information.
