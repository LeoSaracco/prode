"""Formateador de tablas de simulación de grupos y torneo."""
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()


def print_group_table(group_df: pd.DataFrame, group_name: str, n_sims: int) -> None:
    console.print()
    console.rule(f"[bold cyan]GRUPO {group_name}  |  {n_sims:,} simulaciones[/bold cyan]")
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Equipo", style="cyan", width=24)
    t.add_column("1°", justify="right", width=8)
    t.add_column("2°", justify="right", width=8)
    t.add_column("3°", justify="right", width=8)
    t.add_column("4°", justify="right", width=8)
    t.add_column("Clasifica", justify="right", width=10)
    t.add_column("Pts prom", justify="right", width=10)
    t.add_column("DG prom", justify="right", width=10)

    for _, row in group_df.iterrows():
        qualify_pct = f"{row['qualify_direct_prob']*100:.1f}%"
        qualify_color = "green" if row['qualify_direct_prob'] > 0.70 else (
            "yellow" if row['qualify_direct_prob'] > 0.40 else "red"
        )
        t.add_row(
            row["team"],
            f"{row['prob_1st']*100:.1f}%",
            f"{row['prob_2nd']*100:.1f}%",
            f"{row['prob_3rd']*100:.1f}%",
            f"{row['prob_4th']*100:.1f}%",
            f"[{qualify_color}]{qualify_pct}[/{qualify_color}]",
            f"{row['avg_pts']:.2f}",
            f"{row['avg_gd']:+.2f}",
        )
    console.print(t)
    console.print()


def print_group_fixtures(fixtures: list[dict], group_name: str) -> None:
    if not fixtures:
        return

    console.rule(f"[bold cyan]RESULTADOS MAS PROBABLES - GRUPO {group_name}[/bold cyan]")
    t = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    t.add_column("Partido", style="cyan", width=42)
    t.add_column("Resultado", justify="center", width=12)
    t.add_column("Prob.", justify="right", width=10)
    t.add_column("xG", justify="right", width=18)

    for fixture in fixtures:
        scoreline = fixture.get("most_likely_scoreline")
        if scoreline is None:
            continue
        team_a = fixture["team_a"]
        team_b = fixture["team_b"]
        xg = fixture.get("expected_goals", {})
        t.add_row(
            f"{team_a} vs {team_b}",
            f"{scoreline['goals_a']}-{scoreline['goals_b']}",
            f"{scoreline['probability'] * 100:.1f}%",
            f"{xg.get('team_a', 0):.2f} - {xg.get('team_b', 0):.2f}",
        )

    console.print(t)
    console.print()


def print_tournament_forecast(df: pd.DataFrame, n_sims: int, top_n: int = 20) -> None:
    console.print()
    console.rule(f"[bold cyan]PRONÓSTICO MUNDIAL 2026  |  {n_sims:,} simulaciones[/bold cyan]")
    t = Table(box=box.ROUNDED, header_style="bold magenta")
    t.add_column("#", justify="right", width=4)
    t.add_column("Selección", style="cyan", width=24)
    t.add_column("Campeón", justify="right", width=10)
    t.add_column("Finalista", justify="right", width=10)
    t.add_column("Semifinal", justify="right", width=10)
    t.add_column("Cuartos", justify="right", width=10)
    t.add_column("Clasifica GS", justify="right", width=14)

    for _, row in df.head(top_n).iterrows():
        champ_pct = row["p_champion"] * 100
        champ_color = "gold1" if champ_pct > 15 else ("green" if champ_pct > 5 else "white")
        t.add_row(
            str(int(row["rank"])),
            row["team"],
            f"[{champ_color}]{champ_pct:.1f}%[/{champ_color}]",
            f"{row['p_finalist']*100:.1f}%",
            f"{row['p_semifinal']*100:.1f}%",
            f"{row['p_quarterfinal']*100:.1f}%",
            f"{row['p_group_stage']*100:.1f}%",
        )
    console.print(t)

    # Dark horses: equipos fuera del top 8 con p_champion > 3%
    dark_horses = df[(df["rank"] > 8) & (df["p_champion"] > 0.03)]
    if not dark_horses.empty:
        console.print("[bold yellow]🐴 DARK HORSES[/bold yellow]")
        for _, row in dark_horses.iterrows():
            console.print(
                f"  [cyan]{row['team']}[/cyan]  →  "
                f"[green]{row['p_champion']*100:.1f}%[/green] de probabilidad de campeonato"
            )

    console.print()
