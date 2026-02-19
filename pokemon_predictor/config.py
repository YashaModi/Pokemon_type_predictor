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
IMG_SIZE = (100, 100)
RANDOM_SEED = 42

# External Data
EXTERNAL_DATA_PATH = DATA_DIR / "external" / "archive" / "pokedex_(Update_05.20).csv"
