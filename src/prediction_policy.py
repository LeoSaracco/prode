"""Shared prediction presentation policy.

The classifier predicts the match outcome. Poisson scorelines explain plausible
scores. The primary scoreline must therefore be compatible with the predicted
outcome, while the exact global mode can still be shown as an alternative.
"""

from __future__ import annotations

from typing import Any


Outcome = str
ScorelineTuple = tuple[int, int, float]


def predicted_outcome(result: dict[str, Any]) -> Outcome:
    """Return win_a/draw/win_b from the highest W/D/L probability."""
    p_win_a = float(result.get("p_win_a", 0.0))
    p_draw = float(result.get("p_draw", 0.0))
    p_win_b = float(result.get("p_win_b", 0.0))
    if p_win_a >= p_draw and p_win_a >= p_win_b:
        return "win_a"
    if p_win_b >= p_draw and p_win_b >= p_win_a:
        return "win_b"
    return "draw"


def normalize_scoreline(item: Any) -> ScorelineTuple:
    """Accept tuple/list/dict scoreline shapes and return a strict tuple."""
    if isinstance(item, dict):
        return (
            int(item.get("goals_a", 0)),
            int(item.get("goals_b", 0)),
            float(item.get("probability", 0.0)),
        )
    goals_a, goals_b, probability = item
    return int(goals_a), int(goals_b), float(probability)


def is_scoreline_compatible(scoreline: ScorelineTuple, outcome: Outcome) -> bool:
    goals_a, goals_b, _ = scoreline
    if outcome == "win_a":
        return goals_a > goals_b
    if outcome == "win_b":
        return goals_b > goals_a
    return goals_a == goals_b


def fallback_scoreline_for_outcome(result: dict[str, Any], outcome: Outcome) -> ScorelineTuple:
    """Build a deterministic plausible scoreline if no compatible top line exists."""
    xg_a = max(float(result.get("xg_a", 1.2)), 0.35)
    xg_b = max(float(result.get("xg_b", 1.0)), 0.35)
    if outcome == "win_a":
        goals_a = max(1, round(xg_a))
        goals_b = min(max(0, round(xg_b)), goals_a - 1)
        return int(goals_a), int(goals_b), 0.0
    if outcome == "win_b":
        goals_b = max(1, round(xg_b))
        goals_a = min(max(0, round(xg_a)), goals_b - 1)
        return int(goals_a), int(goals_b), 0.0
    goals = max(0, round((xg_a + xg_b) / 2.0))
    return int(goals), int(goals), 0.0


def _representative_scoreline_score(
    scoreline: ScorelineTuple,
    outcome: Outcome,
    xg_a: float,
    xg_b: float,
    max_probability: float,
) -> float:
    goals_a, goals_b, probability = scoreline
    if not is_scoreline_compatible(scoreline, outcome):
        return float("-inf")

    expected_total = xg_a + xg_b
    actual_total = goals_a + goals_b
    prob_score = probability / max(max_probability, 1e-9)
    xg_fit = 1.0 / (1.0 + abs(goals_a - xg_a) + abs(goals_b - xg_b))
    total_fit = 1.0 / (1.0 + abs(actual_total - expected_total))
    target_total = max(0, round(expected_total))
    total_floor_bonus = 0.0
    if expected_total >= 1.85 and actual_total >= target_total:
        total_floor_bonus += 0.16
    if expected_total >= 2.10 and actual_total >= 3:
        total_floor_bonus += 0.16
    if expected_total >= 2.65 and actual_total >= 3:
        total_floor_bonus += 0.10

    # Reward scorelines that communicate a high-volume favorite without
    # overwhelming probability and xG fit.
    volume_bonus = 0.0
    if outcome == "win_a" and xg_a >= 1.35 and goals_a >= 2:
        volume_bonus += 0.12
        if expected_total >= 1.95 and xg_b >= 0.70 and goals_b >= 1:
            volume_bonus += 0.18
        if xg_a >= 1.55 and actual_total >= 3:
            volume_bonus += 0.10
    elif outcome == "win_b" and xg_b >= 1.35 and goals_b >= 2:
        volume_bonus += 0.12
        if expected_total >= 1.95 and xg_a >= 0.70 and goals_a >= 1:
            volume_bonus += 0.18
        if xg_b >= 1.55 and actual_total >= 3:
            volume_bonus += 0.10
    elif outcome == "draw" and expected_total >= 1.8 and goals_a >= 1:
        volume_bonus += 0.12

    if expected_total >= 2.5 and actual_total >= 3:
        volume_bonus += 0.06
    if expected_total < 2.0 and actual_total <= 1:
        volume_bonus += 0.05

    return 0.34 * prob_score + 0.32 * xg_fit + 0.22 * total_fit + total_floor_bonus + volume_bonus


def representative_scoreline_for_outcome(
    scorelines: list[ScorelineTuple],
    outcome: Outcome,
    xg_a: float,
    xg_b: float,
) -> ScorelineTuple | None:
    """Pick a scoreline compatible with outcome and representative of xG volume."""
    compatible = [s for s in scorelines if is_scoreline_compatible(s, outcome)]
    if not compatible:
        return None
    max_probability = max(s[2] for s in compatible)
    return max(
        compatible,
        key=lambda s: (
            _representative_scoreline_score(s, outcome, xg_a, xg_b, max_probability),
            s[2],
        ),
    )


def outcome_scoreline(result: dict[str, Any], outcome: Outcome | None = None) -> ScorelineTuple | None:
    """Return a representative scoreline compatible with the predicted outcome."""
    selected_outcome = outcome or predicted_outcome(result)
    representative = result.get("representative_scoreline")
    if representative is not None:
        normalized = normalize_scoreline(representative)
        if is_scoreline_compatible(normalized, selected_outcome):
            return normalized

    top_scorelines = [normalize_scoreline(s) for s in result.get("top_scorelines", [])]
    if top_scorelines:
        xg_a = max(float(result.get("xg_a", 1.2)), 0.0)
        xg_b = max(float(result.get("xg_b", 1.0)), 0.0)
        representative = representative_scoreline_for_outcome(top_scorelines, selected_outcome, xg_a, xg_b)
        if representative is not None:
            return representative
    if top_scorelines or "xg_a" in result or "xg_b" in result:
        return fallback_scoreline_for_outcome(result, selected_outcome)
    return None


def exact_most_likely_scoreline(result: dict[str, Any]) -> ScorelineTuple | None:
    """Return the global scoreline mode, regardless of outcome compatibility."""
    top_scorelines = result.get("top_scorelines", [])
    if not top_scorelines:
        return None
    return normalize_scoreline(top_scorelines[0])


def enrich_prediction_result(result: dict[str, Any]) -> dict[str, Any]:
    """Add coherent presentation fields to a prediction result dict."""
    outcome = predicted_outcome(result)
    exact = exact_most_likely_scoreline(result)
    compatible = outcome_scoreline(result, outcome)
    result["predicted_outcome"] = outcome
    result["exact_most_likely_scoreline"] = exact
    result["outcome_scoreline"] = compatible
    result["most_likely_scoreline"] = compatible
    return result
