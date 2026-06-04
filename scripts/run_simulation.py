"""Script de simulación del torneo completo."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.models.poisson_model import PoissonGoalModel
from src.simulation.tournament_simulator import TournamentSimulator
from src.output.tournament_table import print_tournament_forecast
from config.settings import MC_ITERATIONS

print("Cargando modelo Poisson...")
poisson = PoissonGoalModel().load()

print(f"Iniciando simulación del torneo ({MC_ITERATIONS:,} iteraciones)...")
sim = TournamentSimulator(poisson_model=poisson)
df = sim.simulate(n_sims=MC_ITERATIONS)

print_tournament_forecast(df, MC_ITERATIONS, top_n=48)
