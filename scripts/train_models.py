"""Train all prediction models."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from config.settings import MATCH_HISTORY_START_YEAR
from src.data.cache_manager import CacheManager
from src.data.collectors.elo_history_collector import EloHistoryCollector
from src.data.collectors.international_results_collector import InternationalResultsCollector
from src.models.trainer import ModelTrainer


def main() -> None:
    print("Loading training datasets...")
    cache = CacheManager()

    international_df = InternationalResultsCollector().collect_match_history(
        start_year=MATCH_HISTORY_START_YEAR
    )
    print(f"International matches: {len(international_df)}")

    elo_df = cache.load("elo_ratings")
    fifa_rankings_df = cache.load("fifa_rankings")
    kaggle_elo_df = cache.load("kaggle_international_elo")
    enriched_match_df = cache.load("kaggle_match_features")
    statsbomb_xg_df = cache.load("statsbomb_team_xg")
    elo_history = EloHistoryCollector()
    elo_history_df = elo_history.compute_from_matches(international_df) if not international_df.empty else pd.DataFrame()
    print(f"Elo history rows: {len(elo_history_df)}")
    print(f"Elo ratings: {len(elo_df) if elo_df is not None else 0} teams")
    print(f"FIFA rankings rows: {len(fifa_rankings_df) if fifa_rankings_df is not None else 0}")
    print(f"Kaggle Elo rows: {len(kaggle_elo_df) if kaggle_elo_df is not None else 0}")
    print(f"Kaggle match-feature rows: {len(enriched_match_df) if enriched_match_df is not None else 0}")
    print(f"StatsBomb xG teams: {len(statsbomb_xg_df) if statsbomb_xg_df is not None else 0}")

    trainer = ModelTrainer(elo_df=elo_df)
    metadata = trainer.train_all(
        kaggle_df=None,
        international_df=international_df,
        enriched_match_df=enriched_match_df,
        elo_history_df=elo_history_df,
        statsbomb_xg_df=statsbomb_xg_df,
        fifa_rankings_df=fifa_rankings_df,
        kaggle_elo_df=kaggle_elo_df,
    )

    print("\n=== TRAINING RESULTS ===")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    if "accuracy_high_elo_diff_200" in metadata:
        print(f"\n  *** High Elo diff (>=200) accuracy: {metadata['accuracy_high_elo_diff_200']:.2%} "
              f"(n={metadata.get('n_high_elo_matches', 0)})")


if __name__ == "__main__":
    main()
