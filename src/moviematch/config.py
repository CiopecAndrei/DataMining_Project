from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "ml-latest-small"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MOVIES_PATH = RAW_DATA_DIR / "movies.csv"
RATINGS_PATH = RAW_DATA_DIR / "ratings.csv"

MIN_LIKED_RATING = 4.0

MIN_SUPPORT = 0.03
MIN_CONFIDENCE = 0.30
MIN_LIFT = 1.0

MAX_ITEMSET_SIZE = 3
