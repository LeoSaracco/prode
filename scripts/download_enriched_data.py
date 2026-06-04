"""Download and cache enriched public datasets used by prode-ML."""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config.settings import (
    KAGGLE_ELO_DIR,
    KAGGLE_MATCH_FEATURES_DIR,
    STATSBOMB_SHOTS_DIR,
)
from src.data.cache_manager import CacheManager
from src.data.collectors.enriched_training_collectors import (
    KaggleInternationalEloCollector,
    KaggleMatchFeaturesCollector,
    StatsBombShotsCollector,
)
from src.data.collectors.wc2026_collectors import (
    FIFAWorldCupScheduleCollector,
    FIFASquadListCollector,
    TransfermarktWC2026Collector,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

KAGGLE_DATASETS = {
    "international_elo": ("saifalnimri/international-football-elo-ratings", KAGGLE_ELO_DIR),
    "match_features": ("lchikry/international-football-match-features-and-statistics", KAGGLE_MATCH_FEATURES_DIR),
}


def _kaggle_json_path() -> Path:
    configured = os.getenv("KAGGLE_CONFIG_DIR")
    base = Path(configured) if configured else Path.home() / ".kaggle"
    return base / "kaggle.json"


def _download_kaggle(force: bool) -> None:
    kaggle_json = _kaggle_json_path()
    if not kaggle_json.exists():
        logger.warning(
            "Kaggle credentials not found at %s. Create a new Kaggle token and place kaggle.json there.",
            kaggle_json,
        )
        return

    for name, (slug, target_dir) in KAGGLE_DATASETS.items():
        target_dir.mkdir(parents=True, exist_ok=True)
        has_files = any(target_dir.rglob("*"))
        if has_files and not force:
            logger.info("Kaggle %s already present at %s", name, target_dir)
            continue
        logger.info("Downloading Kaggle dataset %s -> %s", slug, target_dir)
        cmd = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            slug,
            "-p",
            str(target_dir),
            "--unzip",
        ]
        subprocess.run(cmd, check=True)


def _save_if_available(cache: CacheManager, name: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        logger.warning("%s produced no rows", name)
        return
    cache.save(df, name)
    logger.info("%s cached: %s rows", name, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean enriched prode-ML datasets")
    parser.add_argument("--force", action="store_true", help="Re-download Kaggle files when local files exist")
    parser.add_argument("--skip-web", action="store_true", help="Skip FIFA/Transfermarkt web sources")
    args = parser.parse_args()

    cache = CacheManager()
    _download_kaggle(force=args.force)

    _save_if_available(cache, "kaggle_international_elo", KaggleInternationalEloCollector().collect())
    _save_if_available(cache, "kaggle_match_features", KaggleMatchFeaturesCollector().collect())
    _save_if_available(cache, "statsbomb_team_xg", StatsBombShotsCollector(directory=STATSBOMB_SHOTS_DIR).collect_team_xg_profiles())

    if args.skip_web:
        return

    _save_if_available(cache, "wc2026_schedule", FIFAWorldCupScheduleCollector().collect())
    _save_if_available(cache, "wc2026_squads", FIFASquadListCollector().collect())
    _save_if_available(cache, "transfermarkt_wc2026_team_values", TransfermarktWC2026Collector().collect())


if __name__ == "__main__":
    main()
