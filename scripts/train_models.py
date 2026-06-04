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
from src.data.collectors.enriched_training_collectors import (
    KaggleInternationalEloCollector,
    KaggleMatchFeaturesCollector,
    StatsBombShotsCollector,
)
from src.data.collectors.fifa_rankings_collector import FIFARankingsCollector
from src.data.collectors.international_results_collector import InternationalResultsCollector
from src.data.collectors.kaggle_collector import KaggleCollector
from src.models.trainer import ModelTrainer


def main() -> None:
    print("Loading training datasets...")
    cache = CacheManager()

    international_df = InternationalResultsCollector().collect_match_history()
    start_date = pd.Timestamp(f"{MATCH_HISTORY_START_YEAR}-01-01")
    if "date" in international_df.columns and international_df["date"].notna().any():
        international_df = international_df[international_df["date"] >= start_date].copy()

    kaggle_elo_df = KaggleInternationalEloCollector().collect()
    kaggle_features_df = KaggleMatchFeaturesCollector().collect()
    statsbomb_xg_df = StatsBombShotsCollector().collect_team_xg_profiles()
    kaggle_df = KaggleCollector().collect_match_history()
    elo_df = cache.load("elo_ratings")

    elo_history = EloHistoryCollector()
    elo_history_df = elo_history.compute_from_matches(international_df) if not international_df.empty else pd.DataFrame()

    fifa_rankings = FIFARankingsCollector()
    fifa_df = fifa_rankings.collect()

    combined_elo = _merge_elo_sources(elo_history_df, kaggle_elo_df, elo_df)

    print(f"International matches: {len(international_df)}")
    print(f"Computed Elo history rows: {len(combined_elo)}")
    print(f"Kaggle Elo rows: {len(kaggle_elo_df)}")
    print(f"Kaggle enriched matches: {len(kaggle_features_df)}")
    print(f"StatsBomb xG teams: {len(statsbomb_xg_df)}")
    print(f"Kaggle club fallback matches: {len(kaggle_df)}")
    print(f"FIFA rankings rows: {len(fifa_df)}")

    trainer = ModelTrainer(elo_df=elo_df)
    metadata = trainer.train_all(
        kaggle_df=kaggle_df,
        international_df=international_df,
        enriched_match_df=kaggle_features_df,
        elo_history_df=combined_elo,
        statsbomb_xg_df=statsbomb_xg_df,
    )

    print("\nTraining results:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    if "accuracy_high_elo_diff_200" in metadata:
        print(f"\n  *** High Elo diff (>=200) accuracy: {metadata['accuracy_high_elo_diff_200']:.2%} "
              f"(n={metadata.get('n_high_elo_matches', 0)})")


def _merge_elo_sources(
    computed: pd.DataFrame,
    kaggle: pd.DataFrame,
    elo_ratings: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    if not computed.empty and "team" in computed.columns and "elo_rating" in computed.columns:
        frames.append(computed[["team", "elo_rating"]])
    if not kaggle.empty and "team" in kaggle.columns and "elo_rating" in kaggle.columns:
        frames.append(kaggle[["team", "elo_rating"]])
    if not elo_ratings.empty and "team" in elo_ratings.columns and "elo" in elo_ratings.columns:
        frames.append(elo_ratings[["team", "elo"]].rename(columns={"elo": "elo_rating"}))
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="team", keep="first")
    return merged


if __name__ == "__main__":
    main()
