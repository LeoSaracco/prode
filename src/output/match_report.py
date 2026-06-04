"""Generador del reporte completo de predicción de partido."""
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from src.output.formatters import probability_bar, confidence_color, format_prob_pct
from src.features.elo_features import get_elo_rating
from src.data.national_team_proxy import FALLBACK_STATS

console = Console()


def _build_executive_summary(result: dict) -> str:
    team_a = result["team_a"]
    team_b = result["team_b"]
    p_win = result["p_win_a"]
    p_loss = result["p_win_b"]
    elo_a = result.get("elo_a", 1800)
    elo_b = result.get("elo_b", 1800)
    elo_diff = result.get("elo_diff", 0)

    stats_a = FALLBACK_STATS.get(team_a, {"xg_pg": 1.2, "xga_pg": 1.1, "form_5": 0.55, "ppda": 12})
    stats_b = FALLBACK_STATS.get(team_b, {"xg_pg": 1.2, "xga_pg": 1.1, "form_5": 0.55, "ppda": 12})

    if p_win > p_loss + 0.10:
        favorite = team_a
        underdog = team_b
        fav_adv = "favorito"
    elif p_loss > p_win + 0.10:
        favorite = team_b
        underdog = team_a
        fav_adv = "favorito"
    else:
        favorite = None
        fav_adv = "parejo"

    summary_parts = []

    if favorite:
        summary_parts.append(
            f"{favorite} entra como {fav_adv} (Elo: {elo_a:.0f} vs {elo_b:.0f}, diferencia: {elo_diff:.0f} puntos)."
        )
    else:
        summary_parts.append(
            f"Partido muy parejo. Elo {team_a}: {elo_a:.0f} vs {team_b}: {elo_b:.0f} (diferencia: {elo_diff:.0f} puntos)."
        )

    form_adv = team_a if stats_a["form_5"] > stats_b["form_5"] else team_b
    form_diff = abs(stats_a["form_5"] - stats_b["form_5"])
    if form_diff > 0.05:
        summary_parts.append(
            f"{form_adv} llega con mejor forma reciente ({stats_a['form_5']:.0%} vs {stats_b['form_5']:.0%})."
        )

    if stats_a["ppda"] < stats_b["ppda"] - 1:
        summary_parts.append(
            f"{team_a} presiona más alto (PPDA {stats_a['ppda']:.1f} vs {stats_b['ppda']:.1f})."
        )
    elif stats_b["ppda"] < stats_a["ppda"] - 1:
        summary_parts.append(
            f"{team_b} presiona más alto (PPDA {stats_b['ppda']:.1f} vs {stats_a['ppda']:.1f})."
        )

    xg_a = result.get("xg_a", 1.2)
    xg_b = result.get("xg_b", 1.0)
    summary_parts.append(
        f"xG esperados: {team_a} {xg_a:.2f} — {team_b} {xg_b:.2f}."
    )

    return " ".join(summary_parts)


def print_match_report(result: dict, shap_features: list | None = None) -> None:
    team_a = result["team_a"]
    team_b = result["team_b"]
    p_win = result["p_win_a"]
    p_draw = result["p_draw"]
    p_loss = result["p_win_b"]
    xg_a = result.get("xg_a", 1.2)
    xg_b = result.get("xg_b", 1.0)
    confidence = result.get("confidence", "BAJO")
    top_scorelines = result.get("top_scorelines", [])
    upset_risk = result.get("upset_risk", 0.0)
    elo_a = result.get("elo_a", 1800)
    elo_b = result.get("elo_b", 1800)

    conf_color = confidence_color(confidence)
    title = f"  {team_a.upper()} vs {team_b.upper()}  |  FIFA World Cup 2026"

    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]")
    console.print()

    # Resumen ejecutivo
    summary = _build_executive_summary(result)
    console.print(Panel(summary, title="[bold]RESUMEN EJECUTIVO[/bold]", border_style="blue"))
    console.print()

    # Top features influyentes
    if shap_features:
        console.print("[bold yellow]TOP 5 VARIABLES MÁS INFLUYENTES[/bold yellow]")
        feat_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
        feat_table.add_column("Rank", style="dim", width=4)
        feat_table.add_column("Variable", style="cyan", width=28)
        feat_table.add_column("Valor SHAP", justify="right", width=10)
        feat_table.add_column("Dirección", width=22)
        for i, (feat, val) in enumerate(shap_features[:5], 1):
            direction = f"[green]↑ {team_a}[/green]" if val > 0 else f"[red]↑ {team_b}[/red]"
            feat_table.add_row(str(i), feat, f"{val:+.3f}", direction)
        console.print(feat_table)
        console.print()

    # Probabilidades
    console.print("[bold yellow]PROBABILIDADES[/bold yellow]")
    prob_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    prob_table.add_column("Resultado", width=28)
    prob_table.add_column("Barra", width=22)
    prob_table.add_column("Porcentaje", justify="right", width=8)

    win_color = "green" if p_win > p_loss else "white"
    loss_color = "green" if p_loss > p_win else "white"

    prob_table.add_row(
        f"[{win_color}]Victoria {team_a}[/{win_color}]",
        probability_bar(p_win), f"[{win_color}]{format_prob_pct(p_win)}[/{win_color}]"
    )
    prob_table.add_row("Empate", probability_bar(p_draw), format_prob_pct(p_draw))
    prob_table.add_row(
        f"[{loss_color}]Victoria {team_b}[/{loss_color}]",
        probability_bar(p_loss), f"[{loss_color}]{format_prob_pct(p_loss)}[/{loss_color}]"
    )
    console.print(prob_table)
    console.print()

    # xG y confianza
    console.print(
        f"[bold]GOLES ESPERADOS:[/bold]  "
        f"[cyan]{team_a}[/cyan]: [bold]{xg_a:.2f} xG[/bold]    "
        f"[magenta]{team_b}[/magenta]: [bold]{xg_b:.2f} xG[/bold]"
    )
    console.print(
        f"[bold]NIVEL DE CONFIANZA:[/bold]  [{conf_color}][bold]{confidence}[/bold][/{conf_color}]"
        f"  [dim](ΔElo: {result.get('elo_diff', 0):.0f})[/dim]"
    )
    console.print()

    # Marcadores
    if top_scorelines:
        best = top_scorelines[0]
        console.print(
            f"[bold]RESULTADO MÁS PROBABLE:[/bold]  "
            f"[bold green]{team_a} {best[0]} — {best[1]} {team_b}[/bold green]  "
            f"[dim]({best[2]*100:.1f}%)[/dim]"
        )
        alts = "  |  ".join(
            f"{s[0]}-{s[1]} ({s[2]*100:.1f}%)" for s in top_scorelines[1:]
        )
        console.print(f"[bold]ALTERNATIVOS:[/bold]  [dim]{alts}[/dim]")
        console.print()

    # Riesgo de sorpresa
    upset_color = "red" if upset_risk > 0.25 else ("yellow" if upset_risk > 0.15 else "green")
    console.print(
        f"[bold]RIESGO DE SORPRESA:[/bold]  [{upset_color}]{format_prob_pct(upset_risk)}[/{upset_color}]"
    )
    console.rule(style="dim")
    console.print()
