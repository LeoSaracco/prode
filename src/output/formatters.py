"""Utilidades de formateo para terminal con rich."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

console = Console()


def probability_bar(prob: float, width: int = 20) -> str:
    filled = round(prob * width)
    empty = width - filled
    return "█" * filled + "░" * empty


def confidence_color(level: str) -> str:
    return {"ALTO": "green", "MEDIO": "yellow", "BAJO": "red"}.get(level, "white")


def format_prob_pct(prob: float) -> str:
    return f"{prob * 100:.1f}%"
