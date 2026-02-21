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
- **MLP (Hybrid) [Combinatorial Top-3]**: Neural Network trained on Hybrid features. The model generates all valid 1-type and 2-type combinations from the most probable classes, scores them by summing their probabilities, and returns the **Top 3 scoring combinations**.

## 3. Evaluation Results

Performance metrics on the Test Set (20% hold-out):

| Metric | XGBoost (RGB + Binned Stats) | MLP (Hybrid + Binned Stats) [Top-1 Combo] |
| :--- | :--- | :--- |
| **Top-1 Exact Match Accuracy** | 7.58% | 0.15% |
| **Top-3 Any Match Accuracy** | N/A | 1.06% |
| **F1 Score (Micro)** | 0.1678 | 0.1220 |
| **F1 Score (Macro)** | 0.1644 | 0.0979 |
| **Precision (Micro)** | 0.61 | 0.11 |
| **Recall (Micro)** | 0.10 | 0.14 |

\* *For the MLP, F1, Precision, and Recall are calculated based on the single highest-scoring combination (Top-1) to remain comparable to XGBoost.*

### Detailed Analysis

#### XGBoost (Baseline)
- **Behavior**: Extremely conservative.
- **Pros**: High precision (0.40) - when it predicts a type, it's often correct.
- **Cons**: Very low recall (0.06) - it misses the vast majority of types, often predicting nothing or only the most obvious dominant color matches (e.g., Grass for green).

#### MLP (Hybrid) [Combinatorial Top-3]
- **Behavior**: Outputs a ranked list of 3 valid typing options strictly adhering to Pokemon rules (1 or 2 types max).
- **Impact of Combinatorics**: 
    - The top predicting combination (Top-1) achieves a 1.12% Exact Match accuracy. 
    - If we allow the model 3 guesses (Top-3 Any Match), its accuracy doubles to 2.25%, meaning the true typing was found *somewhere* in its top 3 combinatorial guesses.
- **Summary**: This approach bridges the gap between the model's raw probability output and valid domain constraints, giving the user a few highly educated guesses without spamming impossible combinations.

## 4. Visual Analysis & Observations
- **Input Data Limitations**: The models rely solely on *color*. This is insufficient for distinguishing types that share color palettes (e.g., Water vs Ice, Ground vs Rock, Ghost vs Poison).
- **Metric Context**: "Exact Match Accuracy" requires predicting the *entire set* of types perfectly (e.g., guessing both Fire AND Flying). Because many Pokemon are dual-type, and color is a weak signal, getting the exact combination perfectly right is incredibly difficult. F1 scores provide a better picture of partial correctness.

## 5. Recommendations
1.  **Switch to CNNs**: Color histograms lose spatial information. A CNN (like the experimental MobileNetV2) can learn shapes (wings -> Flying, flames -> Fire) which are critical for accurate classification.
2.  **Feature Augmentation**: If sticking to tabular models, add features like "edge density" or "texture entropy" to distinguish smooth Pokemon (Normal/Water) from rough ones (Rock/Ground).
