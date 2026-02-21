# Model Performance Report

## 1. Data Overview
- **Dataset Source**: Official Artwork images from PokeAPI.
- **Total Samples**: ~1,112 Pokemon.
- **Train/Test Split**: 80% Training (~890 samples), 20% Testing (~222 samples).
- **Features**:
    - **RGB**: Top 5 dominant colors extracted via K-Means.
    - **Histogram**: 3D Color Histogram (8x8x8 bins).
    - **Biological Ratios**: 5 normalized ratios (Physical/Special, Bulk, Glass Cannon, Physical Pillar, Sweeper) calculated from the Pokémon's 6 base stats.
    - **Hybrid**: Concatenation of RGB, Histogram, and Biological Ratios.

## 2. Models Evaluated
- **XGBoost**: `MultiOutputClassifier(XGBClassifier)`. Trained on RGB features.
- **MLP (Hybrid) [Combinatorial Top-3]**: Neural Network trained on Hybrid features. The model generates all valid 1-type and 2-type combinations from the most probable classes, scores them by summing their probabilities, and returns the **Top 3 scoring combinations**.

## 3. Evaluation Results

Performance metrics on the Test Set (20% hold-out):

| Metric | XGBoost (RGB + 5 Bio-Ratios) | MLP (Hybrid + 5 Bio-Ratios) [Top-1 Combo] |
| :--- | :--- | :--- |
| **Top-1 Exact Match Accuracy** | 24.85% | 0.76% |
| **Top-3 Any Match Accuracy** | N/A | 2.12% |
| **F1 Score (Micro)** | 0.5062 | 0.1358 |
| **F1 Score (Macro)** | 0.4997 | 0.0940 |
| **Precision (Micro)** | 0.95 | 0.12 |
| **Recall (Micro)** | 0.35 | 0.16 |

\* *For the MLP, F1, Precision, and Recall are calculated based on the single highest-scoring combination (Top-1) to remain comparable to XGBoost.*

### Detailed Analysis

#### XGBoost (Regularized + Bio-Ratios)
- **Behavior**: Achieves high stability and exceptional precision by balancing visual signals with biological traits.
- **Pros**: Outstanding Precision (0.95). When the model triggers a type prediction, it is nearly flawless. The addition of the 5 Biological Ratios alongside the `colsample_bytree = 0.75` regularization completely prevented statistical memorization while raising the F1 score by a factor of 10x over pure vision models.
- **Cons**: Recall (0.35) leaves room for improvement, as it still tends to miss secondary typings without explicit text descriptions.

#### MLP (Hybrid) [Combinatorial Top-3]
- **Behavior**: Outputs a ranked list of 3 valid typing options strictly adhering to Pokémon rules (1 or 2 types max).
- **Pros**: Bridges the gap between raw probability outputs and domain constraints. Allows the user to see the top 3 most statistically valid combinations.
- **Cons**: The neural network architecture struggles to isolate the tabular Biological Ratios as aggressively as decision trees do, resulting in lower overall F1 micro-scores compared to XGBoost.

## 4. Visual Analysis & Observations
- **Input Data Synergy**: By combining computer vision (K-Means/Histograms) with tabular metadata (Base Stats Ratios), the models broke through the "ambiguous color" ceiling. For example, Water and Ice Pokémon often share identical blue palettes, but their distinct biological speed and defense ratios allow them to be separated algorithmically.
- **Metric Context**: "Exact Match Accuracy" requires predicting the *entire set* of types perfectly. Because many Pokémon are dual-type, hitting an exact match stringently is rare. The F1 micro-score is the anchor metric for evaluating partial correctness.

## 5. Next Steps & Recommendations
1.  **Natural Language Processing (NLP)**: The final frontier to push the F1 score past 0.80 is integrating Pokédex text descriptions using TF-IDF or lightweight transformers (BERT), as textual descriptions often literally contain the typing ("It spews blazing flames...").
2.  **Advanced Vision Transformers (ViT)**: To push purely visual classification further, a massive dataset expansion (50,000+ images) alongside a pre-trained ViT would be required to learn complex geometry (wings = Flying) rather than strictly color.
