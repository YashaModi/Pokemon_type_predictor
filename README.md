# Pokémon Type Predictor (XGBoost vs MLP)

This project compares the performance of two machine learning models in predicting Pokémon types based on their official artwork color palettes.

## Project Structure

- **`data/`**: 
    - `raw/`: Downloaded images from PokéAPI.
    - `processed/`: Extracted features (CSV files).
- **`notebooks/`**:
    - `1_data_loader.ipynb`: Fetches Pokémon metadata and images.
    - `2_feature_extraction.ipynb`: Extracts K-Means dominant colors (for XGBoost) and Color Histograms (for MLP).
    - `3_modeling_evaluation.ipynb`: Trains, evaluates, and compares the models.
    - `4_inference.ipynb`: Validation script to predict types for new images.
- **`src/`**:
    - `features.py`: Shared feature extraction logic.
- **`models/`**: Saved models and encoders.

## Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Notebooks:**
   Execute the notebooks in order:
   
   1. **Data Acquisition:**
      Open `notebooks/1_data_loader.ipynb` and run all cells. This will download ~386 images (Gen 1-3).
      
   2. **Feature Extraction:**
      Run `notebooks/2_feature_extraction.ipynb` to process the images into feature vectors.
      
   3. **Modeling:**
      Run `notebooks/3_modeling_evaluation.ipynb` to train both models and view the comparison metrics.
      
   4. **Inference:**
      Use `notebooks/4_inference.ipynb` to test the models on new images by providing a URL.

## Methodology

### Track A: XGBoost
- **Input:** Top 5 dominant colors (R, G, B) and their percentage coverage (Size: 20).
- **Pipeline:** `MultiOutputClassifier(XGBClassifier)`.
- **Hypothesis:** Interpretable, fast and robust for limited data.

### Track B: MLP (Neural Network)
- **Input:** Flattened 3D Color Histogram (8x8x8 bins = 512 size).
- **Architecture:** `Dense(128) -> Dropout(0.3) -> Dense(64) -> Dense(18, Sigmoid)`.
- **Hypothesis:** Captures richer color distribution information.

## Results
(To be populated after running the evaluation notebook)
