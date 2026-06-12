"""
Exporta datos estaticos del backend prode-ML a JSON para GitHub Pages.

Genera en frontend/public/data/:
  - teams.json, groups.json
  - fixtures.json (72 partidos con prediccion)
  - predictions/<Team>.json (48 archivos, uno por equipo "local")
  - groups/<A..L>.json (12 simulaciones de grupo)
  - tournament.json (simulacion lista, 2500 sims)
  - tournament_bracket.json (bracket tree, 5000 sims)

Uso:
  python scripts/export_static_data.py          # completo
  python scripts/export_static_data.py --quick  # n_sims reducidos para desarrollo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations, permutations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.team_aliases import TEAM_ALIASES, resolve_team_name
from config.wc2026_groups import CONFEDERATION, GROUPS, get_group_for_team
from src.features.elo_features import get_elo_rating
from src.runtime import load_prediction_runtime, predict_match
from src.simulation.group_simulator import GroupSimulator
from src.simulation.tournament_simulator import TournamentSimulator


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "frontend" / "public" / "data"
ALL_TEAMS = sorted(TEAM_ALIASES)


# ── Worker function (module-level for pickle) ──────────────────────────────────

def _predict_chunk(pairs: list[tuple[str, str]]) -> dict[str, dict]:
    """Worker: loads its own runtime and predicts a chunk of pairs.
    Returns a dict keyed by 'TeamA|TeamB' with raw result dicts (not Pydantic models).
    """
    runtime = load_prediction_runtime()
    results: dict[str, dict] = {}
    for team_a, team_b in pairs:
        try:
            result, _ = predict_match(runtime, team_a, team_b, include_shap=False, include_aux=False)
            results[f"{team_a}|{team_b}"] = result
        except Exception:
            results[f"{team_a}|{team_b}"] = None
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _result_to_match_dict(result: dict | None) -> dict | None:
    """Convert a raw predict_match result to the MatchResult JSON shape."""
    if result is None:
        return None
    from api.formatters import format_prediction
    model = format_prediction(result, [])
    return model.model_dump(mode="json")


def _load_wc_fixtures():
    import pandas as pd
    csv_path = PROJECT_ROOT / "data" / "raw" / "international" / "results.csv"
    df = pd.read_csv(str(csv_path), encoding="utf-8")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    wc = df[(df["tournament"] == "FIFA World Cup") & (df["date"] >= "2026-06-01")].copy()
    wc["team_a"] = wc["home_team"].apply(lambda x: resolve_team_name(str(x)))
    wc["team_b"] = wc["away_team"].apply(lambda x: resolve_team_name(str(x)))
    wc = wc[wc["team_a"].notna() & wc["team_b"].notna()]
    wc["group"] = wc["team_a"].apply(get_group_for_team)

    _MD1_END = pd.Timestamp("2026-06-17")
    _MD2_END = pd.Timestamp("2026-06-23")

    def _compute_matchday(date_val):
        if date_val <= _MD1_END:
            return 1
        if date_val <= _MD2_END:
            return 2
        return 3

    wc["matchday"] = wc["date"].apply(_compute_matchday)
    wc["venue"] = wc.apply(lambda r: f"{r['city']}, {r['country']}" if pd.notna(r["city"]) else None, axis=1)
    return wc


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Export steps ───────────────────────────────────────────────────────────────

def export_teams(runtime) -> dict:
    """Generate teams.json matching GET /teams shape."""
    items = []
    for team in ALL_TEAMS:
        items.append({
            "name": team,
            "group": get_group_for_team(team),
            "elo": get_elo_rating(team, runtime.elo_df),
            "confederation": CONFEDERATION.get(team),
        })
    print(f"  teams.json: {len(items)} equipos")
    return {"teams": items}


def export_groups() -> dict:
    """Generate groups.json matching GET /groups shape."""
    return {"groups": GROUPS}


def export_fixtures(runtime) -> dict:
    """Generate fixtures.json matching GET /fixtures shape."""
    from api.formatters import scoreline_tuple_to_schema

    df = _load_wc_fixtures()
    rows = df.to_dict(orient="records")

    # Deduplicate unique pairs and warm the cache
    pair_to_row: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["team_a"], row["team_b"])
        if key not in pair_to_row:
            pair_to_row[key] = row

    for team_a, team_b in pair_to_row:
        runtime.cached_predict(team_a, team_b)

    items = []
    for row in rows:
        p = runtime.cached_predict(row["team_a"], row["team_b"])
        scoreline = p.get("outcome_scoreline")
        items.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "matchday": int(row["matchday"]),
            "group": row["group"] or "",
            "team_a": row["team_a"],
            "team_b": row["team_b"],
            "venue": row.get("venue"),
            "predicted_outcome": p.get("predicted_outcome", "draw"),
            "outcome_scoreline": {
                "goals_a": int(scoreline[0]),
                "goals_b": int(scoreline[1]),
                "probability": round(float(scoreline[2]), 4),
            } if scoreline else None,
            "win_a": round(float(p.get("p_win_a", 0)), 4),
            "draw": round(float(p.get("p_draw", 0)), 4),
            "win_b": round(float(p.get("p_win_b", 0)), 4),
            "confidence": p.get("confidence", "BAJO"),
            "xg_a": round(float(p.get("xg_a", 0)), 2),
            "xg_b": round(float(p.get("xg_b", 0)), 2),
        })
    print(f"  fixtures.json: {len(items)} partidos")
    return {"fixtures": items}


def export_predictions() -> None:
    """Generate predictions/<Team>.json for all 48 teams (48 x 47 = 2256 pairs).
    Uses ProcessPoolExecutor for parallelism.
    """
    # Build all 2256 ordered pairs
    pairs: list[tuple[str, str]] = list(permutations(ALL_TEAMS, 2))
    n_workers = min(6, len(pairs))
    chunk_size = max(1, len(pairs) // n_workers)
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    print(f"  predictions/: {len(pairs)} pares, {n_workers} workers, {len(chunks)} chunks")

    all_results: dict[str, dict] = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_predict_chunk, chunk): i for i, chunk in enumerate(chunks)}

        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False

        if use_tqdm:
            with tqdm(total=len(pairs), desc="  Generando predicciones", unit="par") as pbar:
                for fut in as_completed(futures):
                    chunk_result = fut.result()
                    all_results.update(chunk_result)
                    pbar.update(len(chunk_result))
        else:
            for fut in as_completed(futures):
                chunk_result = fut.result()
                all_results.update(chunk_result)
                print(f"    chunk {futures[fut]+1}/{len(chunks)} completado ({len(chunk_result)} pares)")

    elapsed = time.perf_counter() - t0
    print(f"  Completado en {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Split by team_a
    team_data: dict[str, dict[str, dict]] = {team: {} for team in ALL_TEAMS}
    for key, raw in all_results.items():
        team_a, team_b = key.split("|", 1)
        if raw is not None:
            team_data[team_a][team_b] = _result_to_match_dict(raw)

    pred_dir = DATA_DIR / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for team in ALL_TEAMS:
        write_json(pred_dir / f"{team}.json", team_data[team])
    print(f"  predictions/: {len(ALL_TEAMS)} archivos generados")


def export_groups_sim(runtime, quick: bool) -> None:
    """Generate groups/<A..L>.json for all 12 groups."""
    poisson = runtime.poisson

    def _predictor(team_a, team_b):
        return runtime.cached_predict(team_a, team_b)

    simulator = GroupSimulator(poisson_model=poisson, predictor=_predictor)
    n_sims = 2000 if quick else 10000

    groups_dir = DATA_DIR / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)

    for group_name in sorted(GROUPS.keys()):
        df = simulator.simulate_group(group_name, n_sims=n_sims)
        fixtures = simulator.simulate_group_fixtures(group_name, n_sims=n_sims)
        data = {
            "group": group_name,
            "n_sims": n_sims,
            "results": df.to_dict(orient="records"),
            "fixtures": fixtures,
        }
        write_json(groups_dir / f"{group_name}.json", data)
    print(f"  groups/: {len(GROUPS)} archivos generados (n_sims={n_sims})")


def export_tournament(runtime, quick: bool) -> None:
    """Generate tournament.json (list mode)."""
    poisson = runtime.poisson

    def _predictor(team_a, team_b):
        return runtime.cached_predict(team_a, team_b)

    simulator = TournamentSimulator(poisson_model=poisson, predictor=_predictor)
    n_sims = 1000 if quick else 2500
    top_n = 20

    df = simulator.simulate(n_sims=n_sims).head(top_n)
    data = {
        "n_sims": n_sims,
        "results": df.to_dict(orient="records"),
    }
    write_json(DATA_DIR / "tournament.json", data)
    print(f"  tournament.json: n_sims={n_sims}, top_n={top_n}")


def export_tournament_bracket(runtime, quick: bool) -> None:
    """Generate tournament_bracket.json (bracket tree)."""
    poisson = runtime.poisson

    def _predictor(team_a, team_b):
        return runtime.cached_predict(team_a, team_b)

    simulator = TournamentSimulator(poisson_model=poisson, predictor=_predictor)
    n_sims = 1000 if quick else 5000

    rounds = simulator.simulate_bracket(n_sims=n_sims)
    data = {
        "n_sims": n_sims,
        "rounds": rounds,
    }
    write_json(DATA_DIR / "tournament_bracket.json", data)
    print(f"  tournament_bracket.json: n_sims={n_sims}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Export static data for GitHub Pages")
    parser.add_argument("--quick", action="store_true", help="Reduced n_sims for fast iteration")
    parser.add_argument("--skip-predictions", action="store_true", help="Skip the 2256 prediction pairs")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=== prode-ML: Export estatico para GitHub Pages ===")
    print(f"Output: {DATA_DIR}")
    print(f"Modo: {'rapido' if args.quick else 'completo'}")
    print()

    # 1. Load runtime once for non-prediction exports
    print("[1/6] Cargando modelos...")
    t0 = time.perf_counter()
    runtime = load_prediction_runtime()
    print(f"  Modelos cargados en {time.perf_counter() - t0:.1f}s")
    print()

    # 2. Teams & Groups (instant)
    print("[2/6] Exportando teams + groups...")
    write_json(DATA_DIR / "teams.json", export_teams(runtime))
    write_json(DATA_DIR / "groups.json", export_groups())
    print()

    # 3. Fixtures
    print("[3/6] Exportando fixtures...")
    write_json(DATA_DIR / "fixtures.json", export_fixtures(runtime))
    print()

    # 4. Predictions (the expensive one)
    if args.skip_predictions:
        print("[4/6] Predictions: SKIPPED (--skip-predictions)")
    else:
        print("[4/6] Exportando predictions/...")
        export_predictions()
    print()

    # 5. Group simulations
    print("[5/6] Exportando simulaciones de grupos...")
    export_groups_sim(runtime, args.quick)
    print()

    # 6. Tournament
    print("[6/6] Exportando torneo...")
    export_tournament(runtime, args.quick)
    export_tournament_bracket(runtime, args.quick)
    print()

    total = time.perf_counter() - t0
    print(f"=== Export completado en {total:.1f}s ({total/60:.1f} min) ===")


if __name__ == "__main__":
    main()
