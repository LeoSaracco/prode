"""Model training pipeline."""

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR, ENSEMBLE_WEIGHTS, HIGH_CONFIDENCE_THRESHOLD
from src.features.attack_features import compute_offensive_power
from src.features.defense_features import compute_defensive_stability
from src.features.elo_features import compute_elo_win_probability
from src.features.feature_builder import MATCH_FEATURE_COLUMNS
from src.features.historical_features import compute_world_cup_history_score
from src.features.risk_features import compute_tactical_advantage
from src.features.squad_features import compute_squad_depth_from_market_value
from src.models.lgbm_model import LGBMOutcomeClassifier
from src.models.poisson_model import PoissonGoalModel
from src.models.xgb_model import XGBOutcomeClassifier

logger = logging.getLogger(__name__)


def _standardize_training_matches(df: pd.DataFrame | None, source: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    required = {"team_a", "team_b", "goals_a", "goals_b"}
    if required - set(out.columns):
        return pd.DataFrame()
    out = out.dropna(subset=list(required)).copy()
    out["date"] = pd.to_datetime(out.get("date", pd.NaT), errors="coerce")
    out["goals_a"] = out["goals_a"].astype(int)
    out["goals_b"] = out["goals_b"].astype(int)
    out["result"] = out.apply(
        lambda r: "W" if r.goals_a > r.goals_b else ("L" if r.goals_a < r.goals_b else "D"),
        axis=1,
    )
    out["neutral"] = out.get("neutral", True)
    out["tournament"] = out.get("tournament", "Unknown")
    out["source"] = out.get("source", source)
    return out


def _add_reverse_perspective(df: pd.DataFrame) -> pd.DataFrame:
    away = df.copy()
    away[["team_a", "team_b"]] = df[["team_b", "team_a"]].values
    away[["goals_a", "goals_b"]] = df[["goals_b", "goals_a"]].values
    away["result"] = df["result"].map({"W": "L", "L": "W", "D": "D"})
    if "xg_a" in away.columns and "xg_b" in away.columns:
        away[["xg_a", "xg_b"]] = df[["xg_b", "xg_a"]].values
    return pd.concat([df, away], ignore_index=True)


def _build_team_stats(match_df: pd.DataFrame) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    buckets: dict[str, dict[str, list[float]]] = {}
    for _, row in match_df.sort_values("date").iterrows():
        ta, tb = row["team_a"], row["team_b"]
        buckets.setdefault(ta, {"gf": [], "ga": [], "results": [], "xg": []})
        buckets.setdefault(tb, {"gf": [], "ga": [], "results": [], "xg": []})
        ga, gb = float(row["goals_a"]), float(row["goals_b"])
        buckets[ta]["gf"].append(ga)
        buckets[ta]["ga"].append(gb)
        buckets[tb]["gf"].append(gb)
        buckets[tb]["ga"].append(ga)
        if ga > gb:
            buckets[ta]["results"].append(3)
            buckets[tb]["results"].append(0)
        elif ga < gb:
            buckets[ta]["results"].append(0)
            buckets[tb]["results"].append(3)
        else:
            buckets[ta]["results"].append(1)
            buckets[tb]["results"].append(1)
        if not pd.isna(row.get("xg_a", np.nan)):
            buckets[ta]["xg"].append(float(row["xg_a"]))
        if not pd.isna(row.get("xg_b", np.nan)):
            buckets[tb]["xg"].append(float(row["xg_b"]))

    for team, values in buckets.items():
        results = values["results"]
        if not results:
            continue
        form_5 = sum(results[-5:]) / max(min(len(results), 5) * 3, 1)
        gf = float(np.mean(values["gf"]))
        ga = float(np.mean(values["ga"]))
        xg_pg = float(np.mean(values["xg"])) if values["xg"] else gf
        win_rate = results.count(3) / len(results)
        stats[team] = {
            "xg_pg": max(0.35, min(2.8, xg_pg)),
            "xga_pg": max(0.35, min(2.8, ga)),
            "form_5": float(form_5),
            "ppda": float(max(8.0, 14.0 - (gf - 1.3) * 2.0)),
            "elo": float(1450 + win_rate * 650),
            "games": len(results),
        }
    return stats


def _latest_elo_overrides(elo_history_df: pd.DataFrame | None) -> dict[str, float]:
    if elo_history_df is None or elo_history_df.empty:
        return {}
    df = elo_history_df.dropna(subset=["team", "elo_rating"]).copy()
    if "date" in df.columns:
        df = df.sort_values("date")
    latest = df.groupby("team").tail(1)
    return dict(zip(latest["team"], latest["elo_rating"].astype(float)))


def _apply_statsbomb_xg(stats: dict[str, dict], statsbomb_df: pd.DataFrame | None) -> None:
    if statsbomb_df is None or statsbomb_df.empty:
        return
    for _, row in statsbomb_df.iterrows():
        team = row.get("team")
        if team not in stats:
            continue
        shots = float(row.get("statsbomb_shots", 0) or 0)
        if shots < 50:
            continue
        xg_per_shot = float(row.get("statsbomb_xg_per_shot", 0.10) or 0.10)
        estimated_xg_pg = max(0.5, min(2.4, xg_per_shot * 11.0))
        stats[team]["xg_pg"] = 0.7 * stats[team]["xg_pg"] + 0.3 * estimated_xg_pg


class NationalTeamTrainingFeatureBuilder:
    def __init__(self, team_stats: dict[str, dict]):
        self.team_stats = team_stats

    def get_team_stats(self, team: str) -> dict:
        return self.team_stats.get(team, {
            "xg_pg": 1.1,
            "xga_pg": 1.1,
            "form_5": 0.50,
            "ppda": 13.0,
            "elo": 1600.0,
        })

    def build_match_features(self, team_a: str, team_b: str) -> np.ndarray:
        from src.data.national_team_proxy import MARKET_VALUE_EUR_M

        sa = self.get_team_stats(team_a)
        sb = self.get_team_stats(team_b)
        elo_a = float(sa["elo"])
        elo_b = float(sb["elo"])
        elo_diff = elo_a - elo_b
        op_a = compute_offensive_power(sa["xg_pg"])
        op_b = compute_offensive_power(sb["xg_pg"])
        ds_a = compute_defensive_stability(sa["xga_pg"], sa["ppda"])
        ds_b = compute_defensive_stability(sb["xga_pg"], sb["ppda"])
        wc_a = compute_world_cup_history_score(team_a)
        wc_b = compute_world_cup_history_score(team_b)
        mv_a = min(MARKET_VALUE_EUR_M.get(team_a, 100.0) / 1500, 1.0)
        mv_b = min(MARKET_VALUE_EUR_M.get(team_b, 100.0) / 1500, 1.0)
        tactical_adv = compute_tactical_advantage(elo_diff, op_a, ds_b, op_b, ds_a)
        form_diff = float(sa["form_5"]) - float(sb["form_5"])
        return np.array([
            elo_diff,
            compute_elo_win_probability(elo_a, elo_b),
            sa["xg_pg"] - sb["xg_pg"],
            sa["xga_pg"] - sb["xga_pg"],
            form_diff,
            op_a - op_b,
            ds_a - ds_b,
            compute_squad_depth_from_market_value(team_a) - compute_squad_depth_from_market_value(team_b),
            0.0,
            0.0,
            0.0,
            wc_a - wc_b,
            mv_a - mv_b,
            tactical_adv,
            0.0,
            form_diff * (elo_diff / 400),
            op_a - ds_b,
        ], dtype=np.float32)

    def build_training_matrix(self, match_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        label_map = {"W": 2, "D": 1, "L": 0}
        x_rows, y_rows = [], []
        for _, row in match_df.iterrows():
            feats = self.build_match_features(row["team_a"], row["team_b"])
            is_tournament = int(row.get("is_tournament", 0) or 0)
            is_wc = int(row.get("is_wc", 0) or 0)
            is_qualifier = int(row.get("is_qualifier", 0) or 0)
            is_neutral = int(not row.get("neutral", True))
            extra = np.array([is_tournament, is_wc, is_qualifier, is_neutral], dtype=np.float32)
            x_rows.append(np.concatenate([feats, extra]))
            y_rows.append(label_map[row["result"]])
        return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


class ModelTrainer:
    def __init__(self, elo_df: pd.DataFrame | None = None):
        self.elo_df = elo_df
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def train_all(
        self,
        kaggle_df: pd.DataFrame | None = None,
        international_df: pd.DataFrame | None = None,
        enriched_match_df: pd.DataFrame | None = None,
        elo_history_df: pd.DataFrame | None = None,
        statsbomb_xg_df: pd.DataFrame | None = None,
    ) -> dict:
        training_sources = []
        frames = []
        international = _standardize_training_matches(international_df, "international_results")
        enriched = _standardize_training_matches(enriched_match_df, "kaggle_match_features")
        if not international.empty:
            frames.append(international)
            training_sources.append("international_results")
        if not enriched.empty:
            frames.append(enriched)
            training_sources.append("kaggle_match_features")

        if frames:
            match_df = pd.concat(frames, ignore_index=True).drop_duplicates(
                subset=["date", "team_a", "team_b", "goals_a", "goals_b"]
            )
            if match_df["date"].notna().sum() > 0:
                recent = match_df[match_df["date"] >= pd.Timestamp("2014-01-01")]
                if len(recent) >= 300:
                    match_df = recent
            return self._train_from_national_matches(
                match_df=match_df,
                training_sources=training_sources,
                elo_history_df=elo_history_df,
                statsbomb_xg_df=statsbomb_xg_df,
            )

        logger.warning("No national-team datasets available. Falling back to Kaggle club data.")
        return self._train_from_kaggle_fallback(kaggle_df)

    def _train_from_national_matches(
        self,
        match_df: pd.DataFrame,
        training_sources: list[str],
        elo_history_df: pd.DataFrame | None,
        statsbomb_xg_df: pd.DataFrame | None,
    ) -> dict:
        if len(match_df) < 100:
            raise RuntimeError("Not enough national-team matches to train without defaults.")
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
        val_df = train_matches.copy()
        test_df = test_matches.copy()

        feature_builder = NationalTeamTrainingFeatureBuilder(team_stats)
        x_train, y_train = feature_builder.build_training_matrix(train_df)
        x_val, y_val = feature_builder.build_training_matrix(val_df)
        x_test, y_test = feature_builder.build_training_matrix(test_df)

        return self._fit_models(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            poisson_match_df=train_matches,
            metadata_extra={
                "training_source": "+".join(training_sources),
                "n_source_matches": len(match_df),
                "date_min": str(match_df["date"].min().date()) if match_df["date"].notna().any() else None,
                "date_max": str(match_df["date"].max().date()) if match_df["date"].notna().any() else None,
                "val_date_min": str(val_matches["date"].min().date()) if len(val_matches) > 0 and val_matches["date"].notna().any() else None,
                "test_date_min": str(test_matches["date"].min().date()) if len(test_matches) > 0 and test_matches["date"].notna().any() else None,
                "statsbomb_xg_teams": 0 if statsbomb_xg_df is None else int(len(statsbomb_xg_df)),
                "elo_history_rows": 0 if elo_history_df is None else int(len(elo_history_df)),
            },
        )

    def _train_from_kaggle_fallback(self, kaggle_df: pd.DataFrame | None) -> dict:
        if kaggle_df is None or kaggle_df.empty:
            return self._train_default_models()
        from src.features.club_stats_builder import KaggleFeatureBuilder, build_club_stats_from_history

        df = kaggle_df.rename(columns={
            "home_team": "team_a",
            "away_team": "team_b",
            "home_team_goal": "goals_a",
            "away_team_goal": "goals_b",
        })
        df = _standardize_training_matches(df, "kaggle_club_fallback")
        if len(df) < 50:
            return self._train_default_models()
        df = df.sort_values("date").reset_index(drop=True)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        train_df = _add_reverse_perspective(df.iloc[:train_end])
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        club_stats = build_club_stats_from_history(train_df)
        fb = KaggleFeatureBuilder(club_stats=club_stats, elo_df=self.elo_df)
        x_train, y_train = fb.build_training_matrix(train_df)
        x_val, y_val = fb.build_training_matrix(val_df)
        x_test, y_test = fb.build_training_matrix(test_df)
        return self._fit_models(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            poisson_match_df=pd.DataFrame(),
            metadata_extra={"training_source": "kaggle_club_fallback", "n_source_matches": len(df)},
        )

    def _fit_models(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        poisson_match_df: pd.DataFrame,
        metadata_extra: dict,
    ) -> dict:
        if len(x_train) < 50:
            return self._train_default_models()

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_val_s = scaler.transform(x_val)
        x_test_s = scaler.transform(x_test)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

        xgb = XGBOutcomeClassifier().train(x_train_s, y_train, x_val_s, y_val)
        xgb.save()
        lgbm = LGBMOutcomeClassifier().train(x_train_s, y_train, x_val_s, y_val)
        lgbm.save()

        poisson = PoissonGoalModel()
        if poisson_match_df is not None and not poisson_match_df.empty:
            poisson.fit(poisson_match_df[["team_a", "team_b", "goals_a", "goals_b", "neutral"]])
        else:
            poisson._set_default_params(pd.DataFrame())
        poisson.save()

        xgb_probs = xgb.model_.predict_proba(x_test_s) if xgb.is_fitted_ else np.zeros((len(x_test_s), 3))
        lgbm_probs = lgbm.model_.predict_proba(x_test_s) if lgbm.is_fitted_ else np.zeros((len(x_test_s), 3))

        w_xgb = ENSEMBLE_WEIGHTS.get("xgb", 0.35)
        w_lgbm = ENSEMBLE_WEIGHTS.get("lgbm", 0.30)
        w_elo = ENSEMBLE_WEIGHTS.get("elo", 0.20)
        w_poisson = ENSEMBLE_WEIGHTS.get("poisson", 0.15)
        total_w = w_xgb + w_lgbm + w_elo + w_poisson
        if total_w > 0:
            w_xgb /= total_w
            w_lgbm /= total_w
            w_elo /= total_w
            w_poisson /= total_w

        elo_probs = np.zeros_like(xgb_probs)
        poisson_probs = np.zeros_like(xgb_probs)
        raw_elo_diff = x_test[:, 0]
        for i in range(len(x_test_s)):
            elo_diff_raw = float(raw_elo_diff[i])
            p_win = 1.0 / (1.0 + 10.0 ** (-elo_diff_raw / 400.0))
            p_draw = 0.22 * (1.0 - abs(p_win - 0.5) * 2.0)
            p_win = p_win * (1.0 - p_draw)
            p_loss = 1.0 - p_win - p_draw
            elo_probs[i] = [p_loss, p_draw, p_win]
            poisson_probs[i] = [0.333, 0.334, 0.333]

        blend_probs = (
            w_xgb * xgb_probs + w_lgbm * lgbm_probs +
            w_elo * elo_probs + w_poisson * poisson_probs
        )
        blend_pred = np.argmax(blend_probs, axis=1)

        elo_diff_from_features = x_test[:, 0]
        high_elo_mask = np.abs(elo_diff_from_features) >= 200

        metadata = {
            "train_date": datetime.now().isoformat(),
            "n_train_samples": int(len(x_train)),
            "n_val_samples": int(len(x_val)),
            "n_test_samples": int(len(x_test)),
            "features": MATCH_FEATURE_COLUMNS,
            "accuracy_xgb_global": round(xgb.score(x_test_s, y_test), 4),
            "accuracy_lgbm_global": round(lgbm.score(x_test_s, y_test), 4),
            "accuracy_blend_global": round(float(accuracy_score(y_test, blend_pred)), 4),
            "log_loss_blend": round(float(log_loss(y_test, blend_probs, labels=[0, 1, 2])), 4),
            "accuracy_high_confidence": round(self._segment_accuracy(blend_probs, y_test, threshold=HIGH_CONFIDENCE_THRESHOLD), 4),
            "accuracy_high_elo_diff_200": round(
                float(accuracy_score(y_test[high_elo_mask], blend_pred[high_elo_mask]))
                if high_elo_mask.sum() >= 10 else 0.0, 4
            ),
            "n_high_elo_matches": int(high_elo_mask.sum()),
            "baseline_random": 0.333,
            "split_type": "time_series_chronological",
            **metadata_extra,
        }
        with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Training completed: %s", metadata)
        return metadata

    def _train_default_models(self) -> dict:
        poisson = PoissonGoalModel()
        poisson._set_default_params(pd.DataFrame())
        poisson.save()
        metadata = {
            "train_date": datetime.now().isoformat(),
            "n_train_samples": 0,
            "status": "default_params",
            "training_source": "none",
            "note": "Default Poisson params only. No usable training data was available.",
        }
        with open(MODELS_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return metadata

    def _segment_accuracy(self, probs: np.ndarray, y_test: np.ndarray, threshold: float) -> float:
        mask = probs.max(axis=1) >= threshold
        if not mask.any():
            return 0.0
        preds = np.argmax(probs[mask], axis=1)
        return float(accuracy_score(y_test[mask], preds))
