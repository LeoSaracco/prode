"""Data collection and processing pipeline."""

import logging

import pandas as pd

from src.data.cache_manager import CacheManager
from src.data.collectors.eloratings_collector import EloRatingsCollector
from src.data.collectors.enriched_training_collectors import (
    KaggleInternationalEloCollector,
    KaggleMatchFeaturesCollector,
    StatsBombShotsCollector,
)
from src.data.collectors.fbref_collector import FBrefCollector
from src.data.collectors.international_results_collector import InternationalResultsCollector
from src.data.collectors.kaggle_collector import KaggleCollector
from src.data.collectors.sofifa_collector import SoFIFACollector
from src.data.collectors.understat_collector import UnderstatCollector
from src.data.national_team_proxy import NationalTeamProxy
from src.data.validators import report_coverage

logger = logging.getLogger(__name__)


class DataPipeline:
    """Orchestrates collectors. Each source can fail independently."""

    def __init__(self):
        self.cache = CacheManager()
        self.elo_collector = EloRatingsCollector()
        self.international_collector = InternationalResultsCollector()
        self.kaggle_elo_collector = KaggleInternationalEloCollector()
        self.kaggle_match_features_collector = KaggleMatchFeaturesCollector()
        self.statsbomb_shots_collector = StatsBombShotsCollector()
        self.kaggle_collector = KaggleCollector()
        self.fbref_collector = FBrefCollector()
        self.understat_collector = UnderstatCollector()
        self.sofifa_collector = SoFIFACollector()
        self.proxy = NationalTeamProxy()

    def run(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        logger.info("=" * 60)
        logger.info("Starting data pipeline...")
        logger.info("=" * 60)
        results: dict[str, pd.DataFrame] = {}

        logger.info("[1/9] International national-team results...")
        try:
            results["international_matches"] = self.international_collector.collect_match_history()
        except Exception as e:
            logger.warning("International results failed: %s", e)
            results["international_matches"] = pd.DataFrame()

        logger.info("[2/9] Kaggle international Elo...")
        try:
            results["kaggle_international_elo"] = self.kaggle_elo_collector.collect()
        except Exception as e:
            logger.warning("Kaggle international Elo failed: %s", e)
            results["kaggle_international_elo"] = pd.DataFrame()

        logger.info("[3/9] Kaggle international match features...")
        try:
            results["kaggle_match_features"] = self.kaggle_match_features_collector.collect()
        except Exception as e:
            logger.warning("Kaggle match features failed: %s", e)
            results["kaggle_match_features"] = pd.DataFrame()

        logger.info("[4/9] StatsBomb shot xG...")
        try:
            results["statsbomb_team_xg"] = self.statsbomb_shots_collector.collect_team_xg_profiles()
            if not results["statsbomb_team_xg"].empty:
                self.cache.save(results["statsbomb_team_xg"], "statsbomb_team_xg")
        except Exception as e:
            logger.warning("StatsBomb shots failed: %s", e)
            results["statsbomb_team_xg"] = pd.DataFrame()

        logger.info("[5/9] Kaggle club history fallback...")
        try:
            results["kaggle_matches"] = self.kaggle_collector.collect_match_history()
        except Exception as e:
            logger.warning("Kaggle failed: %s", e)
            results["kaggle_matches"] = pd.DataFrame()

        logger.info("[6/9] National Elo ratings...")
        try:
            results["elo_ratings"] = self.elo_collector.collect_current_ratings(force_refresh)
            if not results["elo_ratings"].empty:
                self.cache.save(results["elo_ratings"], "elo_ratings")
        except Exception as e:
            logger.warning("Elo ratings failed: %s", e)
            results["elo_ratings"] = pd.DataFrame()

        logger.info("[7/9] FBref club stats...")
        try:
            results["fbref_schedule"] = self.fbref_collector.collect_match_schedule(
                force_refresh=force_refresh
            )
        except Exception as e:
            logger.warning("FBref failed: %s", e)
            results["fbref_schedule"] = pd.DataFrame()

        logger.info("[8/9] Understat xG data...")
        try:
            results["understat"] = self.understat_collector.collect_team_match_stats(
                force_refresh=force_refresh
            )
        except Exception as e:
            logger.warning("Understat failed: %s", e)
            results["understat"] = pd.DataFrame()

        logger.info("[9/9] SoFIFA squad ratings...")
        try:
            results["sofifa_teams"] = self.sofifa_collector.collect_team_ratings(
                force_refresh=force_refresh
            )
        except Exception as e:
            logger.warning("SoFIFA failed (optional): %s", e)
            results["sofifa_teams"] = pd.DataFrame()

        logger.info("Building national-team profiles...")
        profiles = self.proxy.build_team_profiles(
            elo_df=results.get("elo_ratings"),
            fbref_df=results.get("fbref_schedule"),
        )
        self.cache.save(profiles, "team_profiles")
        results["team_profiles"] = profiles
        report_coverage(profiles)

        logger.info("Pipeline completed.")
        return results

    def run_fast(self, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
        logger.info("Running fast pipeline (international + Elo + Kaggle fallback)...")
        results: dict[str, pd.DataFrame] = {}

        try:
            results["international_matches"] = self.international_collector.collect_match_history()
        except Exception as e:
            logger.warning("International results failed: %s", e)
            results["international_matches"] = pd.DataFrame()

        try:
            results["kaggle_international_elo"] = self.kaggle_elo_collector.collect()
        except Exception as e:
            logger.warning("Kaggle international Elo failed: %s", e)
            results["kaggle_international_elo"] = pd.DataFrame()

        try:
            results["kaggle_match_features"] = self.kaggle_match_features_collector.collect()
        except Exception as e:
            logger.warning("Kaggle match features failed: %s", e)
            results["kaggle_match_features"] = pd.DataFrame()

        try:
            results["statsbomb_team_xg"] = self.statsbomb_shots_collector.collect_team_xg_profiles()
            if not results["statsbomb_team_xg"].empty:
                self.cache.save(results["statsbomb_team_xg"], "statsbomb_team_xg")
        except Exception as e:
            logger.warning("StatsBomb shots failed: %s", e)
            results["statsbomb_team_xg"] = pd.DataFrame()

        try:
            results["kaggle_matches"] = self.kaggle_collector.collect_match_history()
        except Exception as e:
            logger.warning("Kaggle failed: %s", e)
            results["kaggle_matches"] = pd.DataFrame()

        try:
            results["elo_ratings"] = self.elo_collector.collect_current_ratings(force_refresh)
            if not results["elo_ratings"].empty:
                self.cache.save(results["elo_ratings"], "elo_ratings")
        except Exception as e:
            logger.warning("Elo failed: %s", e)
            results["elo_ratings"] = pd.DataFrame()

        profiles = self.proxy.build_team_profiles(elo_df=results.get("elo_ratings"))
        self.cache.save(profiles, "team_profiles")
        results["team_profiles"] = profiles
        return results
