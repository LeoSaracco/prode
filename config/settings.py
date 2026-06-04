from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

KAGGLE_DB_PATH = Path(os.getenv("KAGGLE_DB_PATH", str(DATA_RAW / "kaggle" / "database.sqlite")))
INTERNATIONAL_RESULTS_PATH = Path(
    os.getenv("INTERNATIONAL_RESULTS_PATH", str(DATA_RAW / "international" / "results.csv"))
)
INTERNATIONAL_RESULTS_URL = os.getenv(
    "INTERNATIONAL_RESULTS_URL",
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
)
KAGGLE_ELO_DIR = Path(
    os.getenv("KAGGLE_ELO_DIR", str(DATA_RAW / "kaggle" / "international_elo"))
)
KAGGLE_MATCH_FEATURES_DIR = Path(
    os.getenv("KAGGLE_MATCH_FEATURES_DIR", str(DATA_RAW / "kaggle" / "match_features"))
)
STATSBOMB_SHOTS_DIR = Path(
    os.getenv("STATSBOMB_SHOTS_DIR", str(DATA_RAW / "statsbomb" / "shots"))
)
STATSBOMB_SHOTS_DATASET = os.getenv(
    "STATSBOMB_SHOTS_DATASET",
    "ClementeH/statsbomb-open-data-shots",
)

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Ligas para recolección de datos de clubes (proxy para selecciones nacionales)
FBREF_LEAGUES = [
    "ESP-La Liga",
    "ENG-Premier League",
    "DEU-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
    "POR-Primeira Liga",
    "NED-Eredivisie",
]
FBREF_SEASONS = ["2022-2023", "2023-2024", "2024-2025"]

UNDERSTAT_LEAGUES = ["La liga", "EPL", "Bundesliga", "Serie A", "Ligue 1"]
UNDERSTAT_SEASONS = [2022, 2023, 2024]

CACHE_TTL_HOURS = 72

MC_ITERATIONS = 100_000

# Pesos del ensemble (deben sumar 1.0)
ENSEMBLE_WEIGHTS = {
    "xgb": 0.35,
    "lgbm": 0.30,
    "elo": 0.20,
    "poisson": 0.15,
}

# Umbrales de confianza para clasificar nivel HIGH/MEDIUM/LOW
CONFIDENCE_HIGH_PROB = 0.55
CONFIDENCE_HIGH_ELO_DIFF = 150
CONFIDENCE_MEDIUM_PROB = 0.45
CONFIDENCE_MEDIUM_ELO_DIFF = 75

# Umbral de confianza para "predicción firme" (accuracy > 80%)
HIGH_CONFIDENCE_THRESHOLD = 0.65

# Parámetro de decaimiento para momentum_score (más reciente = más peso)
MOMENTUM_DECAY = 0.85

# Máximo de goles a considerar en la matriz de Poisson
POISSON_MAX_GOALS = 9

# Número de terceros clasificados en WC2026
WC2026_BEST_THIRDS = 8
WC2026_TOTAL_TEAMS = 48
WC2026_GROUPS = 12
WC2026_TEAMS_PER_GROUP = 4
