# Model Performance Report

## 1. Data Overview
- **Dataset Source**: Official Artwork images from PokeAPI.
- **Total Samples**: ~1,112 Pokemon.
- **Train/Test Split**: 80% Training (~890 samples), 20% Testing (~222 samples).
- **Features**:
    - **RGB**: Top 5 dominant colors extracted via K-Means.
    - **Histogram**: 3D Color Histogram (8x8x8 bins).
    - **Hybrid**: Concatenation of RGB and Histogram features.

## 2. Models Evaluated
- **XGBoost**: `MultiOutputClassifier(XGBClassifier)`. Trained on RGB features.
- **MLP (Hybrid)**: Neural Network trained on Hybrid features with `FocalLoss` to handle class imbalance.

## 3. Evaluation Results

Performance metrics on the Test Set (20% hold-out):

| Metric | XGBoost | MLP (Hybrid) |
| :--- | :--- | :--- |
| **Exact Match Accuracy** | 3.93% | 0.00% |
| **F1 Score (Micro)** | 0.1042 | 0.1837 |
| **F1 Score (Macro)** | 0.0685 | 0.1794 |
| **Precision (Micro)** | 0.40 | 0.10 |
| **Recall (Micro)** | 0.06 | 0.80 |

### detailed Analysis

#### XGBoost (Baseline)
- **Behavior**: Extremely conservative.
- **Pros**: High precision (0.40) - when it predicts a type, it's often correct.
- **Cons**: Very low recall (0.06) - it misses the vast majority of types, often predicting nothing or only the most obvious dominant color matches (e.g., Grass for green).

#### MLP (Hybrid)
- **Behavior**: Aggressive internal scoring, but bounded output.
- **Pros**: High internal recall (0.80) - it successfully retrieves most true types across its probability spectrum.
- **Cons**: Still struggles with precision when raw thresholding is used. To counter this, the `predict.py` script now strictly filters the output to only return the **Top 2** highest probability classes. This prevents the model from predicting unreasonable numbers of types (e.g., 5-6 types) for a single Pokemon.

## 4. Visual Analysis & Observations
- **Input Data Limitations**: The models rely solely on *color*. This is insufficient for distinguishing types that share color palettes (e.g., Water vs Ice, Ground vs Rock, Ghost vs Poison).
- **Class Imbalance**: Rare types (Ice, Ghost, Fairy) are difficult to learn. The MLP's Focal Loss attempted to fix this but resulted in over-correction (predicting rare types too often).

## 5. Recommendations
1.  **Switch to CNNs**: Color histograms lose spatial information. A CNN (like the experimental MobileNetV2) can learn shapes (wings -> Flying, flames -> Fire) which are critical for accurate classification.
2.  **Tune Thresholds**: The MLP threshold (0.85) is arguably too low for practical use given the over-prediction. Raising it would trade recall for precision.
3.  **Feature Augmentation**: If sticking to tabular models, add features like "edge density" or "texture entropy" to distinguish smooth Pokemon (Normal/Water) from rough ones (Rock/Ground).
