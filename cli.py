"""
CLI interactivo de predicciones FIFA World Cup 2026.

Uso:
    python cli.py                              # Modo interactivo
    python cli.py predict "Argentina" "France"
    python cli.py simulate-group H
    python cli.py simulate-tournament
    python cli.py list-teams
    python cli.py update-data [--fast] [--force]
    python cli.py train
"""
import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

# Agregar raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import MODELS_DIR, DATA_PROCESSED, LOG_LEVEL
from config.team_aliases import resolve_team_name, TEAM_ALIASES
from config.wc2026_groups import GROUPS

console = Console()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, "INFO"), format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ─── Carga global de modelos (una sola vez) ─────────────────────────────────

_models_loaded = False
_ensemble = None
_feature_builder = None
_poisson = None
_elo_df = None
_predict_cache: dict[tuple[str, str], dict] = {}


def _predict_for_simulation(team_a: str, team_b: str) -> dict:
    key = (team_a, team_b)
    if key in _predict_cache:
        return _predict_cache[key]
    x = _feature_builder.build_match_features(team_a, team_b)
    result = _ensemble.predict(team_a, team_b, x, elo_source=_elo_dict if _elo_dict is not None else _elo_df)
    _predict_cache[key] = result
    return result


def _load_models(silent: bool = False) -> bool:
    global _models_loaded, _ensemble, _feature_builder, _poisson, _elo_df, _elo_dict

    if _models_loaded:
        return True

    if not silent:
        console.print("[dim]Cargando modelos...[/dim]", end=" ")

    try:
        import pandas as pd
        from src.models.poisson_model import PoissonGoalModel
        from src.models.xgb_model import XGBOutcomeClassifier
        from src.models.lgbm_model import LGBMOutcomeClassifier
        from src.models.elo_model import EloModel
        from src.models.ensemble import EnsemblePredictor
        from src.features.feature_builder import FeatureBuilder
        from src.data.cache_manager import CacheManager
        import joblib

        cache = CacheManager()
        _elo_df = cache.load("elo_ratings")
        _elo_dict = None
        if _elo_df is not None and not _elo_df.empty and "team" in _elo_df.columns:
            _elo_dict = dict(zip(_elo_df["team"], _elo_df["elo_rating"]))
        scaler_path = MODELS_DIR / "scaler.pkl"
        scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        poisson = PoissonGoalModel().load()
        xgb = XGBOutcomeClassifier().load()
        lgbm = LGBMOutcomeClassifier().load()
        elo_model = EloModel(elo_df=_elo_df)

        _ensemble = EnsemblePredictor(
            xgb_model=xgb, lgbm_model=lgbm,
            elo_model=elo_model, poisson_model=poisson,
            scaler=scaler,
        )
        _feature_builder = FeatureBuilder(elo_df=_elo_df)
        _poisson = poisson
        _models_loaded = True

        if not silent:
            console.print("[green]OK[/green]")
        return True

    except Exception as e:
        if not silent:
            console.print(f"[yellow]advertencia: {e}[/yellow]")
        # Fallback: modelos mínimos (solo Elo + Poisson con defaults)
        try:
            from src.models.poisson_model import PoissonGoalModel
            from src.models.elo_model import EloModel
            from src.models.ensemble import EnsemblePredictor
            from src.features.feature_builder import FeatureBuilder
            import pandas as pd

            poisson = PoissonGoalModel()
            poisson._set_default_params(pd.DataFrame())
            elo_model = EloModel()
            _ensemble = EnsemblePredictor(elo_model=elo_model, poisson_model=poisson)
            _feature_builder = FeatureBuilder()
            _poisson = poisson
            _models_loaded = True
            if not silent:
                console.print("[yellow]Usando modelos por defecto (sin entrenamiento).[/yellow]")
            return True
        except Exception as e2:
            console.print(f"[red]Error crítico al cargar modelos: {e2}[/red]")
            return False


def _resolve_or_error(name: str) -> str | None:
    canonical = resolve_team_name(name)
    if canonical is None:
        # Búsqueda parcial
        lower = name.lower()
        for canonical_name in TEAM_ALIASES:
            if lower in canonical_name.lower():
                return canonical_name
        console.print(f"[red]Equipo no encontrado: '{name}'[/red]")
        console.print("[dim]Usá 'python cli.py list-teams' para ver equipos disponibles.[/dim]")
        return None
    return canonical


def _do_predict(team_a_raw: str, team_b_raw: str) -> None:
    if not _load_models():
        return

    team_a = _resolve_or_error(team_a_raw)
    team_b = _resolve_or_error(team_b_raw)
    if team_a is None or team_b is None:
        return

    console.print(f"[dim]Calculando predicción: {team_a} vs {team_b}...[/dim]")

    try:
        x = _feature_builder.build_match_features(team_a, team_b)
        result = _ensemble.predict(team_a, team_b, x, elo_df=_elo_df)
        shap_features = _ensemble.get_shap_explanation(x)

        from src.output.match_report import print_match_report
        print_match_report(result, shap_features=shap_features if shap_features else None)

        # También correr Monte Carlo para confirmar probabilidades
        from src.simulation.match_simulator import simulate_match_from_models
        mc = simulate_match_from_models(team_a, team_b, poisson_model=_poisson)
        console.print(
            f"[dim]Monte Carlo ({mc['n_sims']:,} iter): "
            f"Win={mc['p_win_a']*100:.1f}%  Draw={mc['p_draw']*100:.1f}%  "
            f"Loss={mc['p_win_b']*100:.1f}%[/dim]"
        )
    except Exception as e:
        logger.error(f"Error en predicción: {e}", exc_info=True)
        console.print(f"[red]Error al calcular predicción: {e}[/red]")


def _do_simulate_group(group_name: str) -> None:
    if not _load_models():
        return

    gname = group_name.upper()
    if gname not in GROUPS:
        console.print(f"[red]Grupo '{gname}' no encontrado. Grupos disponibles: {list(GROUPS.keys())}[/red]")
        return

    console.print(f"[dim]Simulando Grupo {gname} (100.000 iteraciones)...[/dim]")
    try:
        from src.simulation.group_simulator import GroupSimulator
        from src.output.tournament_table import print_group_fixtures, print_group_table
        from config.settings import MC_ITERATIONS

        sim = GroupSimulator(poisson_model=_poisson, predictor=_predict_for_simulation)
        result = sim.simulate_group(gname, n_sims=MC_ITERATIONS)
        fixtures = sim.simulate_group_fixtures(gname, n_sims=MC_ITERATIONS)
        print_group_table(result, gname, MC_ITERATIONS)
        print_group_fixtures(fixtures, gname)
    except Exception as e:
        logger.error(f"Error simulando grupo: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


def _do_simulate_tournament(top_n: int = 20) -> None:
    if not _load_models():
        return

    console.print("[dim]Simulando torneo completo (100.000 iteraciones, puede tomar 30-60 segundos)...[/dim]")
    try:
        from src.simulation.tournament_simulator import TournamentSimulator
        from src.output.tournament_table import print_tournament_forecast
        from config.settings import MC_ITERATIONS

        sim = TournamentSimulator(poisson_model=_poisson, predictor=_predict_for_simulation)
        df = sim.simulate(n_sims=MC_ITERATIONS)
        print_tournament_forecast(df, MC_ITERATIONS, top_n=top_n)
    except Exception as e:
        logger.error(f"Error simulando torneo: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")


def _interactive_mode() -> None:
    console.print()
    console.print("[bold cyan]╔══════════════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   prode-ML  |  FIFA World Cup 2026 Predictor    ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════════════╝[/bold cyan]")
    console.print()
    console.print("Comandos disponibles:")
    console.print("  [cyan]predict <equipo_a> <equipo_b>[/cyan]  — predicción de partido")
    console.print("  [cyan]group <A-L>[/cyan]                    — simular fase de grupos")
    console.print("  [cyan]tournament[/cyan]                     — simular torneo completo")
    console.print("  [cyan]top <N>[/cyan]                        — top N candidatos al título")
    console.print("  [cyan]list[/cyan]                           — ver todos los equipos")
    console.print("  [cyan]help[/cyan]                           — mostrar esta ayuda")
    console.print("  [cyan]quit / exit[/cyan]                    — salir")
    console.print()

    _load_models()

    while True:
        try:
            raw = console.input("[bold green]>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Hasta luego.[/dim]")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd = parts[0].lower()

        if cmd in ("quit", "exit", "q", "salir"):
            console.print("[dim]Hasta luego.[/dim]")
            break

        elif cmd == "predict" and len(parts) >= 3:
            _do_predict(parts[1], parts[2])

        elif cmd == "predict" and len(parts) == 2:
            console.print("[yellow]Uso: predict <equipo_a> <equipo_b>[/yellow]")

        elif cmd in ("group", "grupo") and len(parts) >= 2:
            _do_simulate_group(parts[1])

        elif cmd in ("tournament", "torneo"):
            _do_simulate_tournament()

        elif cmd == "top":
            n = int(parts[1]) if len(parts) >= 2 and parts[1].isdigit() else 10
            _do_simulate_tournament(top_n=n)

        elif cmd in ("list", "teams", "equipos"):
            console.print()
            for conf, teams in {
                "CONMEBOL": ["Argentina", "Brazil", "Colombia", "Venezuela", "Chile", "Ecuador", "Peru", "Paraguay", "Uruguay"],
                "UEFA": ["France", "England", "Germany", "Spain", "Portugal", "Netherlands", "Belgium", "Croatia", "Poland", "Ukraine", "Austria", "Czechia"],
                "CAF": ["Morocco", "Nigeria", "Senegal", "Cameroon", "Egypt", "Tunisia", "Mali", "Algeria", "Cote d'Ivoire", "DR Congo"],
                "AFC": ["Japan", "South Korea", "Saudi Arabia", "Iran", "Uzbekistan", "Bahrain"],
                "CONCACAF": ["United States", "Mexico", "Canada", "Panama", "Honduras", "Jamaica", "Trinidad and Tobago", "Costa Rica"],
            }.items():
                console.print(f"[bold yellow]{conf}:[/bold yellow] {', '.join(teams)}")
            console.print()

        elif cmd == "help":
            _interactive_mode.__doc__ and console.print(_interactive_mode.__doc__)
            console.print("Comandos: predict, group, tournament, top, list, quit")

        else:
            # Intentar parsear "Argentina vs Francia" directamente
            if "vs" in raw.lower():
                idx = raw.lower().index("vs")
                ta = raw[:idx].strip()
                tb = raw[idx+2:].strip()
                if ta and tb:
                    _do_predict(ta, tb)
                    continue
            console.print(f"[yellow]Comando no reconocido: '{raw}'. Escribí 'help' para ayuda.[/yellow]")


# ─── CLI con Click ───────────────────────────────────────────────────────────

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """prode-ML — Predicciones FIFA World Cup 2026."""
    if ctx.invoked_subcommand is None:
        _interactive_mode()


@cli.command()
@click.argument("team_a")
@click.argument("team_b")
def predict(team_a, team_b):
    """Predicción de un partido específico."""
    _do_predict(team_a, team_b)


@cli.command("simulate-group")
@click.argument("group")
def simulate_group(group):
    """Simula la fase de grupos para un grupo (A-L)."""
    _do_simulate_group(group)


@cli.command("simulate-tournament")
@click.option("--top", default=20, help="Mostrar top N equipos")
def simulate_tournament(top):
    """Simula el torneo completo (100.000 iteraciones)."""
    _do_simulate_tournament(top_n=top)


@cli.command("list-teams")
def list_teams():
    """Lista todos los equipos disponibles."""
    console.print("\n[bold]Equipos del Mundial 2026:[/bold]\n")
    for team in sorted(TEAM_ALIASES.keys()):
        console.print(f"  {team}")


@cli.command("update-data")
@click.option("--fast", is_flag=True, help="Solo Elo + Kaggle (más rápido)")
@click.option("--force", is_flag=True, help="Ignorar cache y re-descargar")
def update_data(fast, force):
    """Ejecuta el pipeline de recolección de datos."""
    from src.data.pipeline import DataPipeline
    pipeline = DataPipeline()
    if fast:
        pipeline.run_fast(force_refresh=force)
    else:
        pipeline.run(force_refresh=force)


@cli.command("train")
def train():
    """Entrena los modelos ML con los datos disponibles."""
    from src.data.cache_manager import CacheManager
    from src.models.trainer import ModelTrainer
    from src.data.collectors.enriched_training_collectors import (
        KaggleInternationalEloCollector,
        KaggleMatchFeaturesCollector,
        StatsBombShotsCollector,
    )
    from src.data.collectors.international_results_collector import InternationalResultsCollector
    from src.data.collectors.kaggle_collector import KaggleCollector

    console.print("[dim]Cargando datos para entrenamiento...[/dim]")
    international_df = InternationalResultsCollector().collect_match_history()
    kaggle_elo_df = KaggleInternationalEloCollector().collect()
    kaggle_features_df = KaggleMatchFeaturesCollector().collect()
    statsbomb_xg_df = StatsBombShotsCollector().collect_team_xg_profiles()
    kaggle_df = KaggleCollector().collect_match_history()
    cache = CacheManager()
    elo_df = cache.load("elo_ratings")

    trainer = ModelTrainer(elo_df=elo_df)
    metadata = trainer.train_all(
        kaggle_df=kaggle_df,
        international_df=international_df,
        enriched_match_df=kaggle_features_df,
        elo_history_df=kaggle_elo_df,
        statsbomb_xg_df=statsbomb_xg_df,
    )

    console.print("[green]Entrenamiento completado.[/green]")
    if "accuracy_xgb_global" in metadata:
        console.print(
            f"  XGBoost accuracy global: [bold]{metadata['accuracy_xgb_global']:.2%}[/bold]"
        )
        console.print(
            f"  Alta confianza (P>65%):  [bold]{metadata['accuracy_high_confidence']:.2%}[/bold]"
        )


if __name__ == "__main__":
    cli()
