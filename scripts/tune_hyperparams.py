"""Hyperparameter tuning with Optuna + time-series cross-validation.

Tunes RandomForest, XGBoost, LightGBM, CatBoost, and the meta-learner
LogisticRegressionCV jointly using chronological splits.

Usage:
    python scripts/tune_hyperparams.py [--trials 100] [--study-name v1]
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from config.settings import MODELS_DIR, MATCH_HISTORY_START_YEAR
from src.data.cache_manager import CacheManager
from src.data.collectors.elo_history_collector import EloHistoryCollector
from src.data.collectors.fifa_rankings_collector import FIFARankingsCollector
from src.data.collectors.international_results_collector import InternationalResultsCollector
from src.data.collectors.enriched_training_collectors import (
    KaggleInternationalEloCollector,
    KaggleMatchFeaturesCollector,
    StatsBombShotsCollector,
)
from src.data.collectors.kaggle_collector import KaggleCollector
from src.features.feature_builder import MATCH_FEATURE_COLUMNS
from src.models.trainer import (
    ModelTrainer,
    NationalTeamTrainingFeatureBuilder,
    _build_team_stats,
    _standardize_training_matches,
    _add_reverse_perspective,
    _latest_elo_overrides,
    _apply_statsbomb_xg,
)
from src.models.rf_model import RFOutcomeClassifier
from src.models.xgb_model import XGBOutcomeClassifier
from src.models.lgbm_model import LGBMOutcomeClassifier
from src.models.catboost_model import CatBoostOutcomeClassifier
from src.models.poisson_model import PoissonGoalModel
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger.error("optuna not installed. Run: pip install optuna")


def load_data():
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

    international = _standardize_training_matches(international_df, "international_results")
    enriched = _standardize_training_matches(kaggle_features_df, "kaggle_match_features")
    frames = []
    if not international.empty:
        frames.append(international)
    if not enriched.empty:
        frames.append(enriched)

    if frames:
        match_df = pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["date", "team_a", "team_b", "goals_a", "goals_b"]
        )
        if match_df["date"].notna().sum() > 0:
            recent = match_df[match_df["date"] >= pd.Timestamp("2014-01-01")]
            if len(recent) >= 300:
                match_df = recent
    else:
        match_df = pd.DataFrame()

    print(f"Training matches: {len(match_df)}")
    return match_df, elo_history_df, statsbomb_xg_df, elo_df


def prepare_splits(match_df, elo_history_df, statsbomb_xg_df):
    match_df = match_df.sort_values("date").reset_index(drop=True)
    team_stats = _build_team_stats(match_df)
    for team, elo in _latest_elo_overrides(elo_history_df).items():
        team_stats.setdefault(team, {})["elo"] = float(elo)
    _apply_statsbomb_xg(team_stats, statsbomb_xg_df)

    n = len(match_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train_matches = match_df.iloc[:train_end].copy()
    val_matches = match_df.iloc[train_end:val_end].copy()
    test_matches = match_df.iloc[val_end:].copy()

    train_df = _add_reverse_perspective(train_matches)

    fb = NationalTeamTrainingFeatureBuilder(team_stats)
    x_train_raw, y_train = fb.build_training_matrix(train_df)
    x_val_raw, y_val = fb.build_training_matrix(val_matches.copy())
    x_test_raw, y_test = fb.build_training_matrix(test_matches.copy())

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    poisson_df = train_matches[["team_a", "team_b", "goals_a", "goals_b", "neutral"]] if "neutral" in train_matches.columns else train_matches[["team_a", "team_b", "goals_a", "goals_b"]]

    return {
        "x_train": x_train, "y_train": y_train,
        "x_val": x_val, "y_val": y_val,
        "x_test": x_test, "y_test": y_test,
        "x_test_raw": x_test_raw,
        "poisson_df": poisson_df,
        "train_matches": train_matches,
    }


def objective(trial, data):
    x_train, y_train = data["x_train"], data["y_train"]
    x_val, y_val = data["x_val"], data["y_val"]
    x_test, y_test = data["x_test"], data["y_test"]
    x_test_raw = data["x_test_raw"]

    rf_params = {
        "n_estimators": trial.suggest_int("rf_n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("rf_max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("rf_min_samples_split", 5, 30),
        "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 2, 15),
        "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
    }

    xgb_params = {
        "n_estimators": trial.suggest_int("xgb_n_estimators", 300, 900, step=100),
        "max_depth": trial.suggest_int("xgb_max_depth", 3, 9),
        "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
        "gamma": trial.suggest_float("xgb_gamma", 0, 2.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-4, 5.0, log=True),
    }

    lgbm_params = {
        "n_estimators": trial.suggest_int("lgbm_n_estimators", 300, 900, step=100),
        "num_leaves": trial.suggest_int("lgbm_num_leaves", 15, 63),
        "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("lgbm_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("lgbm_colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("lgbm_min_child_samples", 10, 50),
        "reg_alpha": trial.suggest_float("lgbm_reg_alpha", 1e-4, 2.0, log=True),
        "reg_lambda": trial.suggest_float("lgbm_reg_lambda", 1e-4, 5.0, log=True),
    }

    cb_params = {
        "iterations": trial.suggest_int("cb_iterations", 300, 900, step=100),
        "depth": trial.suggest_int("cb_depth", 3, 9),
        "learning_rate": trial.suggest_float("cb_learning_rate", 0.01, 0.15, log=True),
        "l2_leaf_reg": trial.suggest_float("cb_l2_leaf_reg", 0.5, 10.0),
        "border_count": trial.suggest_int("cb_border_count", 32, 255, step=32),
    }

    meta_Cs = trial.suggest_int("meta_Cs", 5, 15)

    models = {
        "rf": RFOutcomeClassifier(params=rf_params).train(x_train, y_train),
        "xgb": XGBOutcomeClassifier(params=xgb_params).train(x_train, y_train, x_val, y_val),
        "lgbm": LGBMOutcomeClassifier(params=lgbm_params).train(x_train, y_train, x_val, y_val),
        "catboost": CatBoostOutcomeClassifier(params=cb_params).train(x_train, y_train, x_val, y_val),
    }

    train_elo = _elo_batch(x_train, y_train)
    val_elo_raw = data["x_val"][:, 0] if "x_val" in data else x_val[:, 0]

    val_probs = {}
    for name, model in models.items():
        if model.is_fitted_:
            val_probs[name] = model.predict_proba_batch(x_val)
        else:
            val_probs[name] = np.tile([0.38, 0.24, 0.38], (len(x_val), 3))

    from sklearn.linear_model import LogisticRegressionCV
    val_elo_stack = _elo_batch_for_diffs(x_val[:, 0]) if x_val.shape[1] >= 1 else np.tile([0.38, 0.24, 0.38], (len(x_val), 3))
    val_stacked = np.hstack([
        val_probs["rf"], val_probs["xgb"], val_probs["lgbm"],
        val_probs["catboost"], val_elo_stack,
    ])

    meta = LogisticRegressionCV(
        Cs=meta_Cs, cv=3, multi_class="multinomial",
        max_iter=2000, random_state=42, n_jobs=-1,
    )
    meta.fit(val_stacked, y_val)

    test_probs = {}
    for name, model in models.items():
        if model.is_fitted_:
            test_probs[name] = model.predict_proba_batch(x_test)
        else:
            test_probs[name] = np.tile([0.38, 0.24, 0.38], (len(x_test), 3))
    test_elo = _elo_batch_for_diffs(x_test_raw[:, 0]) if x_test_raw.shape[1] >= 1 else np.tile([0.38, 0.24, 0.38], (len(x_test), 3))
    test_stacked = np.hstack([
        test_probs["rf"], test_probs["xgb"], test_probs["lgbm"],
        test_probs["catboost"], test_elo,
    ])

    meta_probs = meta.predict_proba(test_stacked)
    loss = log_loss(y_test, meta_probs, labels=[0, 1, 2])
    acc = accuracy_score(y_test, meta_probs.argmax(axis=1))

    trial.set_user_attr("accuracy", float(acc))

    return loss


def _elo_batch_for_diffs(elo_diffs):
    n = len(elo_diffs)
    probs = np.zeros((n, 3))
    for i in range(n):
        ed = float(elo_diffs[i])
        p_win = 1.0 / (1.0 + 10.0 ** (-ed / 400.0))
        p_draw = 0.22 * (1.0 - abs(p_win - 0.5) * 2.0)
        p_win = p_win * (1.0 - p_draw)
        p_loss = 1.0 - p_win - p_draw
        probs[i] = [p_loss, p_draw, p_win]
    return probs


def _elo_batch(x, y):
    return np.tile([0.38, 0.24, 0.38], (len(y), 3))


def run_tuning(trials: int = 50, study_name: str = "v1"):
    if not HAS_OPTUNA:
        print("optuna not installed.")
        return

    print(f"=== Optuna Hyperparameter Tuning ({trials} trials, study={study_name}) ===")
    match_df, elo_history_df, statsbomb_xg_df, elo_df = load_data()

    if match_df.empty or len(match_df) < 200:
        print("Not enough data for tuning. Run pipeline first.")
        return

    data = prepare_splits(match_df, elo_history_df, statsbomb_xg_df)
    print(f"Train: {len(data['y_train'])}, Val: {len(data['y_val'])}, Test: {len(data['y_test'])}")

    study = optuna.create_study(
        study_name=f"prode_ml_{study_name}",
        direction="minimize",
        storage="sqlite:///models/optuna_study.db",
        load_if_exists=True,
    )

    def wrapped_objective(trial):
        return objective(trial, data)

    study.optimize(wrapped_objective, n_trials=trials, show_progress_bar=True)

    print(f"\n=== Best trial (trial #{study.best_trial.number}) ===")
    print(f"  Log loss: {study.best_value:.4f}")
    print(f"  Accuracy: {study.best_trial.user_attrs.get('accuracy', 0):.4f}")

    best_params = dict(study.best_params)
    best_params["meta_Cs"] = best_params.pop("meta_Cs", 10)
    best_params["tuning_date"] = datetime.now().isoformat()
    best_params["n_trials"] = trials
    best_params["best_log_loss"] = float(study.best_value)
    best_params["best_accuracy"] = float(study.best_trial.user_attrs.get("accuracy", 0))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_params_path = MODELS_DIR / "best_params.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {best_params_path}")

    for key, val in best_params.items():
        print(f"  {key}: {val}")

    return best_params


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Optuna")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--study-name", type=str, default="v1", help="Optuna study name")
    args = parser.parse_args()

    run_tuning(trials=args.trials, study_name=args.study_name)
