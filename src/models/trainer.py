"""Model training pipeline."""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from config.settings import MODELS_DIR, HIGH_CONFIDENCE_THRESHOLD
from src.features.attack_features import compute_offensive_power
from src.features.defense_features import compute_defensive_stability
from src.features.elo_features import compute_elo_win_probability
from src.features.feature_builder import MATCH_FEATURE_COLUMNS
from src.features.historical_features import compute_world_cup_history_score
from src.features.risk_features import compute_tactical_advantage
from src.features.squad_features import compute_squad_depth_from_market_value, compute_squad_quality
from src.data.national_team_proxy import FALLBACK_STATS, MARKET_VALUE_EUR_M
from src.models.catboost_model import CatBoostOutcomeClassifier
from src.models.confederation_models import ConfederationModels
from src.models.ensemble import EnsemblePredictor
from src.models.lgbm_model import LGBMOutcomeClassifier
from src.models.poisson_model import PoissonGoalModel
from src.models.rf_model import RFOutcomeClassifier
from src.models.two_stage import TwoStageClassifier
from src.models.xgb_model import XGBOutcomeClassifier

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


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


def _context_flags(row: pd.Series) -> tuple[float, float, float, float]:
    tournament = str(row.get("tournament", "") or "").lower()
    is_friendly = "friendly" in tournament
    is_wc = "world cup" in tournament and "qualification" not in tournament and "qualifier" not in tournament
    is_qualifier = "qualification" in tournament or "qualifier" in tournament
    is_home = 1.0 if not bool(row.get("neutral", True)) else 0.0
    return float(not is_friendly), float(is_wc), float(is_qualifier), is_home


def _update_elo(elo_a: float, elo_b: float, goals_a: int, goals_b: int, k: float = 28.0) -> tuple[float, float]:
    expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
    score_a = 1.0 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0.0)
    delta = k * (score_a - expected_a)
    return elo_a + delta, elo_b - delta


def _latest_rank_lookup(rankings_df: pd.DataFrame | None) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    if rankings_df is None or rankings_df.empty or "team" not in rankings_df.columns:
        return {}
    if "fifa_rank" not in rankings_df.columns:
        return {}
    df = rankings_df.copy()
    df["date"] = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")
    df = df.dropna(subset=["team", "fifa_rank"]).sort_values(["team", "date"])
    lookup: dict[str, list[tuple[pd.Timestamp, float]]] = {}
    for team, group in df.groupby("team"):
        lookup[str(team)] = [(pd.Timestamp(r.date), float(r.fifa_rank)) for r in group.itertuples()]
    return lookup


def _rank_before(
    lookup: dict[str, list[tuple[pd.Timestamp, float]]],
    team: str,
    date: pd.Timestamp | None,
) -> float:
    rows = lookup.get(team)
    if not rows:
        return 75.0
    if date is None or pd.isna(date):
        return rows[-1][1]
    rank = rows[0][1]
    for row_date, row_rank in rows:
        if pd.isna(row_date) or row_date <= date:
            rank = row_rank
        else:
            break
    return float(rank)


def _confidence_thresholds_from_validation(probs: np.ndarray, y_true: np.ndarray) -> dict:
    max_probs = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    candidates = [round(v, 2) for v in np.arange(0.45, 0.86, 0.05)]
    rows = []
    for threshold in candidates:
        mask = max_probs >= threshold
        n = int(mask.sum())
        acc = float(accuracy_score(y_true[mask], preds[mask])) if n else 0.0
        rows.append({"threshold": threshold, "n": n, "accuracy": round(acc, 4)})

    min_high_prob = 0.70
    eligible = [r for r in rows if r["threshold"] >= min_high_prob and r["n"] >= 20 and r["accuracy"] >= 0.75]
    if eligible:
        high = eligible[-1]
    else:
        high_candidates = [r for r in rows if r["threshold"] >= min_high_prob and r["n"] >= 20]
        high = max(high_candidates, key=lambda r: (r["accuracy"], r["n"])) if high_candidates else max(rows, key=lambda r: (r["accuracy"], r["n"]))

    medium_candidates = [r for r in rows if r["threshold"] < high["threshold"] and r["n"] >= 20]
    medium = medium_candidates[-1] if medium_candidates else {"threshold": 0.45, "n": int(len(y_true)), "accuracy": 0.0}
    return {
        "high_prob": float(high["threshold"]),
        "medium_prob": float(medium["threshold"]),
        "target_high_accuracy": 0.75,
        "min_high_samples": 20,
        "min_high_prob": min_high_prob,
        "validation_bins": rows,
        "selected_high_validation_accuracy": high["accuracy"],
        "selected_high_validation_samples": high["n"],
    }


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
        consistency = float(sa.get("form_5", 0.5)) - float(sb.get("form_5", 0.5))

        return np.array([
            elo_diff,
            compute_elo_win_probability(elo_a, elo_b),
            sa["xg_pg"] - sb["xg_pg"],
            sa["xga_pg"] - sb["xga_pg"],
            form_diff,
            op_a - op_b,
            ds_a - ds_b,
            compute_squad_quality(team_a) - compute_squad_quality(team_b),
            0.0,  # consistency_diff — not available in simple builder
            wc_a - wc_b,
            mv_a - mv_b,
            tactical_adv,
            0.0,  # h2h_advantage — not available in simple builder
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ], dtype=np.float32)

    def build_training_matrix(self, match_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        label_map = {"W": 2, "D": 1, "L": 0}
        x_rows, y_rows = [], []
        for _, row in match_df.iterrows():
            feats = self.build_match_features(row["team_a"], row["team_b"])
            is_tournament = int(row.get("is_tournament", 0) or 0)
            is_wc = int(row.get("is_wc", 0) or 0)
            is_qualifier = int(row.get("is_qualifier", 0) or 0)
            is_home = 1.0 if not row.get("neutral", True) else 0.0
            feats[13] = float(is_tournament)
            feats[14] = float(is_wc)
            feats[15] = float(is_qualifier)
            feats[16] = is_home
            x_rows.append(feats)
            y_rows.append(label_map[row["result"]])
        return np.array(x_rows, dtype=np.float32), np.array(y_rows, dtype=np.int32)


class RollingNationalTeamFeatureBuilder:
    """Builds chronological features using only state available before each match."""

    def __init__(
        self,
        fifa_rankings_df: pd.DataFrame | None = None,
        statsbomb_xg_df: pd.DataFrame | None = None,
        initial_elo_df: pd.DataFrame | None = None,
    ):
        self.rank_lookup = _latest_rank_lookup(fifa_rankings_df)
        self.statsbomb_xg = self._statsbomb_lookup(statsbomb_xg_df)
        self.initial_elo = self._initial_elo_lookup(initial_elo_df)
        self.state: dict[str, dict] = {}
        self.h2h: dict[tuple[str, str], list[float]] = {}

    def _statsbomb_lookup(self, statsbomb_xg_df: pd.DataFrame | None) -> dict[str, dict]:
        if statsbomb_xg_df is None or statsbomb_xg_df.empty or "team" not in statsbomb_xg_df.columns:
            return {}
        out: dict[str, dict] = {}
        for _, row in statsbomb_xg_df.iterrows():
            team = str(row.get("team", ""))
            shots = float(row.get("statsbomb_shots", 0) or 0)
            if shots < 50:
                continue
            xg_per_shot = float(row.get("statsbomb_xg_per_shot", 0.10) or 0.10)
            out[team] = {"xg_pg": max(0.5, min(2.4, xg_per_shot * 11.0))}
        return out

    def _initial_elo_lookup(self, initial_elo_df: pd.DataFrame | None) -> dict[str, float]:
        if initial_elo_df is None or initial_elo_df.empty or "team" not in initial_elo_df.columns:
            return {}
        rating_col = "elo_rating" if "elo_rating" in initial_elo_df.columns else None
        if rating_col is None:
            return {}
        df = initial_elo_df.dropna(subset=["team", rating_col]).copy()
        if "date" in df.columns:
            df = df.sort_values("date")
        latest = df.groupby("team").tail(1)
        return dict(zip(latest["team"].astype(str), latest[rating_col].astype(float)))

    def _team_state(self, team: str) -> dict:
        if team not in self.state:
            fallback = FALLBACK_STATS.get(team, {
                "xg_pg": 1.1,
                "xga_pg": 1.1,
                "form_5": 0.50,
                "ppda": 13.0,
            }).copy()
            fallback.update(self.statsbomb_xg.get(team, {}))
            initial_elo = float(self.initial_elo.get(team, 1600.0))
            self.state[team] = {
                "gf": [],
                "ga": [],
                "points": [],
                "dates": [],
                "elo": initial_elo,
                "elo_history": [initial_elo],
                "fallback": fallback,
            }
        return self.state[team]

    def _profile(self, team: str, date: pd.Timestamp | None) -> dict:
        st = self._team_state(team)
        n = len(st["gf"])
        fallback = st["fallback"]
        if n:
            prior_weight = 8.0
            gf = (float(np.mean(st["gf"])) * n + fallback["xg_pg"] * prior_weight) / (n + prior_weight)
            ga = (float(np.mean(st["ga"])) * n + fallback["xga_pg"] * prior_weight) / (n + prior_weight)
            recent_points = st["points"][-5:]
            form_5 = float(sum(recent_points) / max(3 * len(recent_points), 1))
            last_date = st["dates"][-1]
            days_since = float(max((date - last_date).days, 0)) if date is not None and not pd.isna(date) else 30.0
        else:
            gf = float(fallback["xg_pg"])
            ga = float(fallback["xga_pg"])
            form_5 = float(fallback["form_5"])
            days_since = 30.0

        elo_hist = st["elo_history"]
        elo_momentum = float(st["elo"] - elo_hist[-6]) if len(elo_hist) >= 6 else 0.0
        recent_pts = st["points"][-5:]
        consistency = float(np.std(recent_pts)) if len(recent_pts) >= 2 else 0.0
        return {
            "xg_pg": max(0.35, min(2.8, gf)),
            "xga_pg": max(0.35, min(2.8, ga)),
            "form_5": form_5,
            "ppda": float(max(8.0, 14.0 - (gf - 1.3) * 2.0)),
            "elo": float(st["elo"]),
            "elo_momentum": elo_momentum,
            "days_since_last_match": days_since,
            "fifa_rank": _rank_before(self.rank_lookup, team, date),
            "games": n,
            "consistency": consistency,
        }

    def _h2h_advantage(self, team_a: str, team_b: str) -> float:
        results = self.h2h.get((team_a, team_b), [])
        if len(results) < 3:
            return 0.0
        return float(2.0 * np.mean(results[-10:]) - 1.0)  # -1=always lost, 0=balanced, +1=always won

    def build_match_features(self, row: pd.Series, reverse: bool = False) -> np.ndarray:
        team_a = str(row["team_b"] if reverse else row["team_a"])
        team_b = str(row["team_a"] if reverse else row["team_b"])
        date = pd.Timestamp(row["date"]) if not pd.isna(row.get("date", pd.NaT)) else None

        sa = self._profile(team_a, date)
        sb = self._profile(team_b, date)
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
        is_tournament, is_wc, is_qualifier, is_home = _context_flags(row)
        if reverse:
            is_home = 0.0

        return np.array([
            elo_diff,
            compute_elo_win_probability(elo_a, elo_b),
            sa["xg_pg"] - sb["xg_pg"],
            sa["xga_pg"] - sb["xga_pg"],
            float(sa["form_5"] - sb["form_5"]),
            op_a - op_b,
            ds_a - ds_b,
            compute_squad_quality(team_a) - compute_squad_quality(team_b),
            float(sb["consistency"] - sa["consistency"]),  # positive = A more consistent
            wc_a - wc_b,
            mv_a - mv_b,
            tactical_adv,
            self._h2h_advantage(team_a, team_b),
            is_tournament,
            is_wc,
            is_qualifier,
            is_home,
            float(sb["fifa_rank"] - sa["fifa_rank"]),
            float(sa["elo_momentum"] - sb["elo_momentum"]),
            float(sa["days_since_last_match"] - sb["days_since_last_match"]),
            float(sb["days_since_last_match"] - sa["days_since_last_match"]),
        ], dtype=np.float32)

    def update_from_match(self, row: pd.Series) -> None:
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        goals_a = int(row["goals_a"])
        goals_b = int(row["goals_b"])
        date = pd.Timestamp(row["date"]) if not pd.isna(row.get("date", pd.NaT)) else pd.Timestamp("1900-01-01")
        st_a = self._team_state(team_a)
        st_b = self._team_state(team_b)

        points_a = 3 if goals_a > goals_b else (1 if goals_a == goals_b else 0)
        points_b = 3 if goals_b > goals_a else (1 if goals_a == goals_b else 0)
        st_a["gf"].append(float(goals_a))
        st_a["ga"].append(float(goals_b))
        st_a["points"].append(points_a)
        st_a["dates"].append(date)
        st_b["gf"].append(float(goals_b))
        st_b["ga"].append(float(goals_a))
        st_b["points"].append(points_b)
        st_b["dates"].append(date)

        new_elo_a, new_elo_b = _update_elo(float(st_a["elo"]), float(st_b["elo"]), goals_a, goals_b)
        st_a["elo"] = new_elo_a
        st_b["elo"] = new_elo_b
        st_a["elo_history"].append(new_elo_a)
        st_b["elo_history"].append(new_elo_b)

        result_a = 1.0 if goals_a > goals_b else (0.5 if goals_a == goals_b else 0.0)
        self.h2h.setdefault((team_a, team_b), []).append(result_a)
        self.h2h.setdefault((team_b, team_a), []).append(1.0 - result_a)

    def build_split_matrices(
        self,
        match_df: pd.DataFrame,
        train_end: int,
        val_end: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        label_map = {"W": 2, "D": 1, "L": 0}
        x_train, y_train, train_rows = [], [], []
        x_val, y_val = [], []
        x_test, y_test, test_rows = [], [], []

        for idx, row in match_df.sort_values("date").reset_index(drop=True).iterrows():
            label = label_map.get(row["result"], 1)
            if idx < train_end:
                x_train.append(self.build_match_features(row, reverse=False))
                y_train.append(label)
                train_rows.append(row.to_dict())
                x_train.append(self.build_match_features(row, reverse=True))
                y_train.append({"W": 0, "D": 1, "L": 2}.get(row["result"], 1))
                rev = row.to_dict()
                rev["team_a"], rev["team_b"] = rev["team_b"], rev["team_a"]
                rev["goals_a"], rev["goals_b"] = rev["goals_b"], rev["goals_a"]
                rev["result"] = {"W": "L", "L": "W", "D": "D"}.get(rev["result"], "D")
                train_rows.append(rev)
            elif idx < val_end:
                x_val.append(self.build_match_features(row, reverse=False))
                y_val.append(label)
            else:
                x_test.append(self.build_match_features(row, reverse=False))
                y_test.append(label)
                test_rows.append(row.to_dict())
            self.update_from_match(row)

        return (
            np.array(x_train, dtype=np.float32),
            np.array(y_train, dtype=np.int32),
            np.array(x_val, dtype=np.float32),
            np.array(y_val, dtype=np.int32),
            np.array(x_test, dtype=np.float32),
            np.array(y_test, dtype=np.int32),
            pd.DataFrame(train_rows),
            pd.DataFrame(test_rows),
        )

    def latest_profiles(self) -> pd.DataFrame:
        rows = []
        today = pd.Timestamp.now()
        for team in sorted(self.state):
            p = self._profile(team, today)
            rows.append({
                "team": team,
                "elo_rating": p["elo"],
                "xg_per_game": p["xg_pg"],
                "xga_per_game": p["xga_pg"],
                "form_5": p["form_5"],
                "ppda": p["ppda"],
                "elo_momentum": p["elo_momentum"],
                "days_since_last_match": p["days_since_last_match"],
                "fifa_rank": p["fifa_rank"],
                "games": p["games"],
                "consistency": p["consistency"],
            })
        return pd.DataFrame(rows)

    def h2h_stats_for_inference(self) -> dict[str, dict]:
        out = {}
        for (ta, tb), results in self.h2h.items():
            if len(results) >= 3:
                out[f"{ta}||{tb}"] = {
                    "n": len(results),
                    "win_rate": round(float(np.mean(results[-10:])), 4),
                }
        return out


def _apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """p_cal[c] ∝ p[c]^(1/T); renormalized. T>1 softens, T<1 sharpens."""
    T = max(0.05, T)
    powered = np.power(np.clip(probs, 1e-9, 1.0), 1.0 / T)
    row_sums = powered.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return powered / row_sums


def _find_temperature(val_probs: np.ndarray, y_val: np.ndarray) -> float:
    """Find temperature T that minimizes log_loss on validation probabilities."""
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import log_loss as _log_loss

    def objective(T):
        cal = _apply_temperature(val_probs, T)
        return _log_loss(y_val, cal, labels=[0, 1, 2])

    result = minimize_scalar(objective, bounds=(0.1, 5.0), method="bounded")
    return float(result.x)


def _compute_recency_weights(train_df: pd.DataFrame, half_life_days: float = 1095.0) -> np.ndarray | None:
    """Exponential decay weights: matches from 3 years ago get ~0.5 weight vs latest."""
    if train_df is None or train_df.empty or "date" not in train_df.columns:
        return None
    dates = pd.to_datetime(train_df["date"], errors="coerce")
    if dates.isna().all():
        return None
    max_date = dates.max()
    days_back = (max_date - dates).dt.days.fillna(0).values.astype(float)
    weights = np.exp(-np.log(2) * days_back / half_life_days)
    mean_w = weights.mean()
    if mean_w > 0:
        weights /= mean_w
    return weights.astype(np.float32)


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
        fifa_rankings_df: pd.DataFrame | None = None,
        kaggle_elo_df: pd.DataFrame | None = None,
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
            match_df = match_df.sort_values("date").reset_index(drop=True)
            return self._train_from_national_matches(
                match_df=match_df,
                training_sources=training_sources,
                elo_history_df=elo_history_df,
                statsbomb_xg_df=statsbomb_xg_df,
                fifa_rankings_df=fifa_rankings_df,
                kaggle_elo_df=kaggle_elo_df,
            )

        logger.warning("No national-team datasets available. Falling back to Kaggle club data.")
        return self._train_from_kaggle_fallback(kaggle_df)

    def _train_from_national_matches(
        self,
        match_df: pd.DataFrame,
        training_sources: list[str],
        elo_history_df: pd.DataFrame | None,
        statsbomb_xg_df: pd.DataFrame | None,
        fifa_rankings_df: pd.DataFrame | None,
        kaggle_elo_df: pd.DataFrame | None = None,
    ) -> dict:
        if len(match_df) < 100:
            raise RuntimeError("Not enough national-team matches to train without defaults.")
        match_df = match_df.sort_values("date").reset_index(drop=True)

        n = len(match_df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        train_matches = match_df.iloc[:train_end].copy()
        val_matches = match_df.iloc[train_end:val_end].copy()
        test_matches = match_df.iloc[val_end:].copy()

        initial_elo_df = kaggle_elo_df if kaggle_elo_df is not None and not kaggle_elo_df.empty else elo_history_df
        feature_builder = RollingNationalTeamFeatureBuilder(
            fifa_rankings_df=fifa_rankings_df,
            initial_elo_df=initial_elo_df,
        )
        x_train, y_train, x_val, y_val, x_test, y_test, train_df, test_df = feature_builder.build_split_matrices(
            match_df=match_df,
            train_end=train_end,
            val_end=val_end,
        )
        latest_profiles = feature_builder.latest_profiles()
        if not latest_profiles.empty:
            latest_profiles.to_json(MODELS_DIR / "inference_team_profiles.json", orient="records", indent=2)
        h2h_stats = feature_builder.h2h_stats_for_inference()
        with open(MODELS_DIR / "inference_h2h_stats.json", "w", encoding="utf-8") as f:
            json.dump(h2h_stats, f, indent=2)

        sample_weight = _compute_recency_weights(train_df)

        return self._fit_models(
            x_train=x_train, y_train=y_train,
            x_val=x_val, y_val=y_val,
            x_test=x_test, y_test=y_test,
            poisson_match_df=train_matches,
            train_matches_df=train_df,
            val_matches_df=val_matches,
            test_matches_df=test_df,
            sample_weight=sample_weight,
            metadata_extra={
                "training_source": "+".join(training_sources),
                "n_source_matches": len(match_df),
                "date_min": str(match_df["date"].min().date()) if match_df["date"].notna().any() else None,
                "date_max": str(match_df["date"].max().date()) if match_df["date"].notna().any() else None,
                "val_date_min": str(val_matches["date"].min().date()) if len(val_matches) > 0 and val_matches["date"].notna().any() else None,
                "test_date_min": str(test_matches["date"].min().date()) if len(test_matches) > 0 and test_matches["date"].notna().any() else None,
                "statsbomb_xg_teams": 0 if statsbomb_xg_df is None else int(len(statsbomb_xg_df)),
                "statsbomb_xg_usage": "cached_not_used_for_training_static_aggregate",
                "elo_history_rows": 0 if elo_history_df is None else int(len(elo_history_df)),
                "kaggle_elo_rows": 0 if kaggle_elo_df is None else int(len(kaggle_elo_df)),
                "fifa_rankings_rows": 0 if fifa_rankings_df is None else int(len(fifa_rankings_df)),
                "feature_generation": "rolling_no_future_leakage",
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

    def _load_best_params(self) -> dict:
        path = MODELS_DIR / "best_params.json"
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _extract_model_params(self, best_params: dict, prefix: str) -> dict:
        result = {}
        for key, val in best_params.items():
            if key.startswith(prefix + "_"):
                result[key[len(prefix) + 1:]] = val
        return result

    def _fit_models(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        poisson_match_df: pd.DataFrame,
        train_matches_df: pd.DataFrame | None = None,
        val_matches_df: pd.DataFrame | None = None,
        test_matches_df: pd.DataFrame | None = None,
        sample_weight: np.ndarray | None = None,
        metadata_extra: dict | None = None,
    ) -> dict:
        if metadata_extra is None:
            metadata_extra = {}
        if len(x_train) < 50:
            return self._train_default_models()

        best_params = self._load_best_params()
        tuned = bool(best_params)

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_val_s = scaler.transform(x_val)
        x_test_s = scaler.transform(x_test)
        joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

        rf_params = self._extract_model_params(best_params, "rf")
        xgb_params = self._extract_model_params(best_params, "xgb")
        lgbm_params = self._extract_model_params(best_params, "lgbm")
        cb_params = self._extract_model_params(best_params, "cb")

        models = {}
        base_workers = _env_int("TRAIN_BASE_WORKERS", 4)
        logger.info(
            "Entrenando modelos base en paralelo: RF, XGB, LGBM, CatBoost "
            "(workers=%d, model_jobs=%s)",
            base_workers,
            os.getenv("TRAIN_MODEL_JOBS", "2"),
        )
        t0 = perf_counter()
        sw = sample_weight
        with ThreadPoolExecutor(max_workers=base_workers) as executor:
            futures = {}
            futures["rf"] = executor.submit(
                lambda sw=sw: RFOutcomeClassifier(params=rf_params).train(x_train_s, y_train, sample_weight=sw))
            futures["xgb"] = executor.submit(
                lambda sw=sw: XGBOutcomeClassifier(params=xgb_params).train(x_train_s, y_train, x_val_s, y_val, sample_weight=sw))
            futures["lgbm"] = executor.submit(
                lambda sw=sw: LGBMOutcomeClassifier(params=lgbm_params).train(x_train_s, y_train, x_val_s, y_val, sample_weight=sw))
            futures["cb"] = executor.submit(
                lambda sw=sw: CatBoostOutcomeClassifier(params=cb_params).train(x_train_s, y_train, x_val_s, y_val, sample_weight=sw))

            for future in as_completed(futures.values()):
                name = next(k for k, v in futures.items() if v is future)
                try:
                    models[name] = future.result(timeout=1)
                    models[name].save()
                    logger.info("Modelo base %s listo.", name)
                except Exception as e:
                    logger.warning("Model %s failed: %s. Skipping.", name, e)
                    models[name] = None
            for name in futures:
                models.setdefault(name, None)
        logger.info("Modelos base finalizados en %.1fs.", perf_counter() - t0)

        rf = models.get("rf")
        xgb = models.get("xgb")
        lgbm = models.get("lgbm")
        catboost = models.get("cb")

        logger.info("Entrenando Poisson...")
        t0 = perf_counter()
        poisson = PoissonGoalModel()
        if poisson_match_df is not None and not poisson_match_df.empty:
            try:
                poisson.fit(poisson_match_df[["team_a", "team_b", "goals_a", "goals_b", "neutral"]])
            except Exception:
                logger.warning("Poisson fit failed. Using defaults.")
                poisson._set_default_params(pd.DataFrame())
        else:
            poisson._set_default_params(pd.DataFrame())
        poisson.save()
        logger.info("Poisson listo en %.1fs.", perf_counter() - t0)

        def _poisson_probs_batch(matches_df: pd.DataFrame | None) -> np.ndarray | None:
            """Returns (N, 3) array in [L=0, D=1, W=2] order for each match."""
            if matches_df is None or matches_df.empty:
                return None
            rows = []
            for _, row in matches_df.iterrows():
                try:
                    p_win_a, p_draw, p_win_b = poisson.predict_outcome_probs(
                        str(row.get("team_a", "")), str(row.get("team_b", ""))
                    )
                    rows.append([p_win_b, p_draw, p_win_a])  # label order: L=0, D=1, W=2
                except Exception:
                    rows.append([0.38, 0.24, 0.38])
            if not rows:
                return None
            arr = np.array(rows, dtype=np.float32)
            s = arr.sum(axis=1, keepdims=True)
            s[s == 0] = 1.0
            return arr / s

        poisson_val_probs = _poisson_probs_batch(val_matches_df)
        poisson_test_probs = _poisson_probs_batch(test_matches_df)

        model_order = [("rf", "rf"), ("xgb", "xgb"), ("lgbm", "lgbm"), ("catboost", "cb")]
        val_acc = {}

        def _elo_probs_for_diff(elo_diff):
            p_win = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
            p_draw = 0.22 * (1.0 - abs(p_win - 0.5) * 2.0)
            p_win = p_win * (1.0 - p_draw)
            p_loss = 1.0 - p_win - p_draw
            return np.array([p_loss, p_draw, p_win])

        def _elo_probs_batch(diffs):
            out = np.zeros((len(diffs), 3))
            for i, d in enumerate(diffs):
                out[i] = _elo_probs_for_diff(float(d))
            return out

        logger.info("Calculando accuracies de validation y pesos del voting...")
        for metric_name, model_key in model_order:
            m = models.get(model_key)
            if m and m.is_fitted_:
                preds = m.predict_proba_batch(x_val_s).argmax(axis=1)
                val_acc[metric_name] = float(accuracy_score(y_val, preds))
            else:
                val_acc[metric_name] = 0.33
        val_acc["elo"] = float(accuracy_score(y_val, _elo_probs_batch(x_val[:, 0]).argmax(axis=1)))
        if poisson_val_probs is not None and len(poisson_val_probs) == len(y_val):
            val_acc["poisson"] = float(accuracy_score(y_val, poisson_val_probs.argmax(axis=1)))
        else:
            val_acc["poisson"] = 0.40  # synthetic fallback — gives weight ~0.07

        weights = {}
        for name in [m[0] for m in model_order] + ["elo"]:
            weights[name] = max(0.05, val_acc[name] - 0.33)
        weights["elo"] = 0.10  # fixed baseline
        if poisson_val_probs is not None and len(poisson_val_probs) == len(y_val):
            weights["poisson"] = 0.07  # fixed — orthogonal signal from goal distribution
        total = sum(weights.values()) or 1.0
        weights = {k: v / total for k, v in weights.items()}

        val_elo = _elo_probs_batch(x_val[:, 0])
        val_blend_probs = np.zeros((len(y_val), 3))
        for metric_name, model_key in model_order:
            m = models.get(model_key)
            if m and m.is_fitted_:
                val_blend_probs += weights[metric_name] * m.predict_proba_batch(x_val_s)
            else:
                val_blend_probs += weights[metric_name] * np.tile([0.38, 0.24, 0.38], (len(y_val), 3))
        val_blend_probs += weights["elo"] * val_elo
        if poisson_val_probs is not None and len(poisson_val_probs) == len(y_val):
            val_blend_probs += weights["poisson"] * poisson_val_probs
        row_sums = val_blend_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        val_blend_probs /= row_sums
        confidence_thresholds = _confidence_thresholds_from_validation(val_blend_probs, y_val)
        with open(MODELS_DIR / "confidence_thresholds.json", "w", encoding="utf-8") as f:
            json.dump(confidence_thresholds, f, indent=2)

        # Temperature scaling calibration fitted on validation probabilities.
        # More robust than isotonic regression when val set is small (<1000 samples):
        # isotonic overfits to extreme values and hurts log_loss on test.
        # Temperature T > 1 softens probabilities toward uniform; T < 1 sharpens them.
        temperature = _find_temperature(val_blend_probs, y_val)
        joblib.dump({"temperature": temperature}, MODELS_DIR / "probability_calibrators.pkl")
        logger.info("Temperature scaling calibrator fitted: T=%.4f", temperature)

        test_elo = _elo_probs_batch(x_test[:, 0])
        blend_probs = np.zeros((len(y_test), 3))
        for metric_name, model_key in model_order:
            m = models.get(model_key)
            if m and m.is_fitted_:
                blend_probs += weights[metric_name] * m.predict_proba_batch(x_test_s)
            else:
                blend_probs += weights[metric_name] * np.tile([0.38, 0.24, 0.38], (len(y_test), 3))
        blend_probs += weights["elo"] * test_elo
        if poisson_test_probs is not None and len(poisson_test_probs) == len(y_test):
            blend_probs += weights["poisson"] * poisson_test_probs
        row_sums = blend_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        blend_probs /= row_sums

        cal_blend_probs = _apply_temperature(blend_probs, temperature)

        blend_pred = cal_blend_probs.argmax(axis=1)

        elo_diff_test = x_test[:, 0]
        high_elo_mask = np.abs(elo_diff_test) >= 200

        logger.info("Entrenando TwoStage...")
        t0 = perf_counter()
        two_stage = TwoStageClassifier()
        two_stage.train(x_train_s, y_train)
        two_stage.save()
        ts_acc = round(float(accuracy_score(y_test, two_stage.predict_proba_batch(x_test_s).argmax(axis=1))), 4) if two_stage.is_fitted_ else 0.0
        logger.info("TwoStage listo en %.1fs. Accuracy test: %.4f", perf_counter() - t0, ts_acc)

        logger.info("Entrenando ConfederationModels...")
        t0 = perf_counter()
        conf_models = ConfederationModels()
        conf_acc = 0.0
        n_conf_pairs = 0
        if train_matches_df is not None and not train_matches_df.empty and \
           "team_a" in train_matches_df.columns:
            try:
                conf_models.train(x_train_s, y_train,
                                  list(train_matches_df["team_a"].astype(str)),
                                  list(train_matches_df["team_b"].astype(str)))
                conf_models.save()
                if conf_models.is_fitted_ and test_matches_df is not None and not test_matches_df.empty:
                    conf_pred = conf_models.predict_proba_batch(
                        x_test_s,
                        list(test_matches_df["team_a"].astype(str)),
                        list(test_matches_df["team_b"].astype(str)),
                    ).argmax(axis=1)
                    conf_acc = round(float(accuracy_score(y_test, conf_pred)), 4)
                n_conf_pairs = len(conf_models.models)
            except Exception as e:
                logger.warning("Confederation models failed: %s", e)
        logger.info(
            "ConfederationModels listo en %.1fs. Pares=%d, accuracy test=%.4f",
            perf_counter() - t0,
            n_conf_pairs,
            conf_acc,
        )

        ensemble = EnsemblePredictor()
        ensemble.set_weights_from_accuracies(val_acc)
        ensemble.save_weights(weights)

        accuracies = {}
        for metric_name, model_key in model_order:
            m = models.get(model_key)
            if m and m.is_fitted_:
                accuracies[metric_name] = round(m.score(x_test_s, y_test), 4)
            else:
                accuracies[metric_name] = 0.0
        accuracies["voting"] = round(float(accuracy_score(y_test, blend_pred)), 4)  # blend_pred from calibrated probs
        accuracies["two_stage"] = ts_acc
        accuracies["confederation"] = conf_acc

        metadata = {
            "train_date": datetime.now().isoformat(),
            "n_train_samples": int(len(x_train)),
            "n_val_samples": int(len(x_val)),
            "n_test_samples": int(len(x_test)),
            "n_features": len(MATCH_FEATURE_COLUMNS),
            "features": MATCH_FEATURE_COLUMNS,
            "accuracy_rf": accuracies.get("rf", 0),
            "accuracy_xgb": accuracies.get("xgb", 0),
            "accuracy_lgbm": accuracies.get("lgbm", 0),
            "accuracy_catboost": accuracies.get("catboost", 0),
            "accuracy_voting": accuracies["voting"],
            "accuracy_two_stage": accuracies["two_stage"],
            "accuracy_confederation": accuracies["confederation"],
            "n_confederation_pairs": n_conf_pairs,
            "log_loss_voting": round(float(log_loss(y_test, cal_blend_probs, labels=[0, 1, 2])), 4),
            "log_loss_uncalibrated": round(float(log_loss(y_test, blend_probs, labels=[0, 1, 2])), 4),
            "accuracy_high_confidence": round(self._segment_accuracy(cal_blend_probs, y_test, threshold=confidence_thresholds["high_prob"]), 4),
            "n_high_confidence_matches": int((cal_blend_probs.max(axis=1) >= confidence_thresholds["high_prob"]).sum()),
            "confidence_thresholds": confidence_thresholds,
            "accuracy_high_elo_diff_200": round(
                float(accuracy_score(y_test[high_elo_mask], blend_pred[high_elo_mask]))
                if high_elo_mask.sum() >= 10 else 0.0, 4
            ),
            "recency_weighted_training": sample_weight is not None,
            "n_high_elo_matches": int(high_elo_mask.sum()),
            "baseline_random": 0.333,
            "split_type": "time_series_chronological",
            "ensemble_architecture": "RF+XGB+LGBM+CB+Elo+Poisson -> accuracy-weighted voting",
            "accuracy_poisson_val": round(val_acc.get("poisson", 0.0), 4),
            "temperature_scaling": round(temperature, 4),
            "voting_weights": {k: round(v, 4) for k, v in weights.items()},
            "val_accuracies": {k: round(v, 4) for k, v in val_acc.items()},
            "tuned_hyperparameters": tuned,
            "parallel_training": True,
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
