"""Train all prediction models."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.cache_manager import CacheManager
from src.data.collectors.enriched_training_collectors import (
    KaggleInternationalEloCollector,
    KaggleMatchFeaturesCollector,
    StatsBombShotsCollector,
)
from src.data.collectors.international_results_collector import InternationalResultsCollector
from src.data.collectors.kaggle_collector import KaggleCollector
from src.models.trainer import ModelTrainer


def main() -> None:
    print("Loading training datasets...")
    cache = CacheManager()
    international_df = InternationalResultsCollector().collect_match_history()
    kaggle_elo_df = KaggleInternationalEloCollector().collect()
    kaggle_features_df = KaggleMatchFeaturesCollector().collect()
    statsbomb_xg_df = StatsBombShotsCollector().collect_team_xg_profiles()
    kaggle_df = KaggleCollector().collect_match_history()
    elo_df = cache.load("elo_ratings")

    print(f"International matches: {len(international_df)}")
    print(f"Kaggle Elo rows: {len(kaggle_elo_df)}")
    print(f"Kaggle enriched matches: {len(kaggle_features_df)}")
    print(f"StatsBomb xG teams: {len(statsbomb_xg_df)}")
    print(f"Kaggle club fallback matches: {len(kaggle_df)}")

    trainer = ModelTrainer(elo_df=elo_df)
    metadata = trainer.train_all(
        kaggle_df=kaggle_df,
        international_df=international_df,
        enriched_match_df=kaggle_features_df,
        elo_history_df=kaggle_elo_df,
        statsbomb_xg_df=statsbomb_xg_df,
    )

    print("\nTraining results:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
