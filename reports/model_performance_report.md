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
- **MLP (Hybrid) [Top-3]**: Neural Network trained on Hybrid features. Uses `FocalLoss` internally to score classes, but the output is strictly bounded to the **Top 3** highest probabilities that cross the 0.85 threshold.

## 3. Evaluation Results

Performance metrics on the Test Set (20% hold-out):

| Metric | XGBoost | MLP (Hybrid) [Top-3] |
| :--- | :--- | :--- |
| **Exact Match Accuracy** | 3.93% | 0.00% |
| **F1 Score (Micro)** | 0.1042 | 0.1973 |
| **F1 Score (Macro)** | 0.0685 | 0.1664 |
| **Precision (Micro)** | 0.40 | 0.15 |
| **Recall (Micro)** | 0.06 | 0.30 |

### Detailed Analysis

#### XGBoost (Baseline)
- **Behavior**: Extremely conservative.
- **Pros**: High precision (0.40) - when it predicts a type, it's often correct.
- **Cons**: Very low recall (0.06) - it misses the vast majority of types, often predicting nothing or only the most obvious dominant color matches (e.g., Grass for green).

#### MLP (Hybrid) [Top-3]
- **Behavior**: More balanced recall/precision trade-off via output bounding.
- **Impact of Top-3 vs Top-2 Logic**: 
    - Expanding the limit from Top-2 to Top-3 increased **Recall** significantly (from 0.19 up to 0.30) and overall **F1 Score** (from 0.16 to 0.19), because the model is allowed an extra guess to catch correct secondary types.
    - However, **Exact Match Accuracy** dropped to 0%. Since a Pokemon can only have at most 2 actual types, a Top-3 prediction is mathematically guaranteed to be a "partial match" (never an "exact match") if the model outputs 3 types!
- **Summary**: Top-3 provides a higher likelihood of capturing the true types (better recall), but creates inherently "imprecise" sets (since it often includes a 3rd incorrect type).

## 4. Visual Analysis & Observations
- **Input Data Limitations**: The models rely solely on *color*. This is insufficient for distinguishing types that share color palettes (e.g., Water vs Ice, Ground vs Rock, Ghost vs Poison).
- **Metric Context**: "Exact Match Accuracy" requires predicting the *entire set* of types perfectly (e.g., guessing both Fire AND Flying). Because many Pokemon are dual-type, and color is a weak signal, getting the exact combination perfectly right is incredibly difficult. F1 scores provide a better picture of partial correctness.

## 5. Recommendations
1.  **Switch to CNNs**: Color histograms lose spatial information. A CNN (like the experimental MobileNetV2) can learn shapes (wings -> Flying, flames -> Fire) which are critical for accurate classification.
2.  **Feature Augmentation**: If sticking to tabular models, add features like "edge density" or "texture entropy" to distinguish smooth Pokemon (Normal/Water) from rough ones (Rock/Ground).
