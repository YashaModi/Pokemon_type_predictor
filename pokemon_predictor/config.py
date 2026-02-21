from pathlib import Path

# Paths
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Parameters
IMG_SIZE = (224, 224)
RANDOM_SEED = 42

# Model Hyperparameters
TEST_SPLIT = 0.2
KMEANS_CLUSTERS = 5
XGB_PENALTY = 0.75
MLP_THRESHOLD = 0.85

# External Data
EXTERNAL_DATA_PATH = DATA_DIR / "external" / "archive" / "pokedex_(Update_05.20).csv"
