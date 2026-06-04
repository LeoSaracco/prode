"""Script de recolección de datos."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from src.data.pipeline import DataPipeline

parser = argparse.ArgumentParser(description="Pipeline de recolección de datos prode-ML")
parser.add_argument("--fast", action="store_true", help="Solo Elo + Kaggle")
parser.add_argument("--force", action="store_true", help="Ignorar cache")
args = parser.parse_args()

pipeline = DataPipeline()
if args.fast:
    pipeline.run_fast(force_refresh=args.force)
else:
    pipeline.run(force_refresh=args.force)
