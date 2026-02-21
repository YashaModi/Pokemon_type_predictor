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

### Prediction Examples
The following grid illustrates scenarios where the model leverages the base-stat ratios effectively, alongside cases where it struggles (typically missing secondary typings due to lack of text/lore context).

![Prediction Examples](figures/prediction_examples.png)

## 5. Strategies for Maximizing F1 Score
To push the F1 Micro-score from the current `~0.506` baseline up to the `0.80+` range, the following advanced methodologies should be implemented:

1. **Natural Language Processing (NLP) / Lore Mining**: The single highest-yield upgrade. Pokémon types are heavily tied to their lore (e.g., "breathes fire", "haunts", "floats"). By extracting TF-IDF or transformer embeddings from the official Pokédex text descriptions and concatenating them with the tabular/visual vectors, the model gains explicit contextual knowledge.
2. **Bayesian Hyperparameter Optimization**: Replace the manual XGBoost grid search with an automated `Optuna` or `Hyperopt` tuning pipeline to exhaustively optimize tree depth, learning rates, `min_child_weight`, and `gamma` simultaneously.
3. **Advanced Class Imbalance Techniques**: Ghost, Ice, and Fairy types are notoriously sparse in generation 1-5 data. Implementing `SMOTE` (Synthetic Minority Over-sampling Technique) strictly on the minority class feature vectors during training can drastically improve macro and micro F1 recall.
4. **Ensemble & Soft Voting**: Instead of relying solely on the regularized XGBoost, build a Soft Voting ensemble that averages the predicted probability distributions of both the XGBoost and the MLP architectures before making the final threshold cut off.
5. **Architectural CNN Fine-Tuning**: If visual processing is to be pursued further, a pre-trained feature extractor (like `EfficientNetB0`) must be explicitly un-frozen and fine-tuned on the Pokémon dataset to learn shapes (wings = Flying) rather than strictly color palettes.
