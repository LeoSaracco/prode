"""
Reporte PDF oficial prode-ML — FIFA World Cup 2026.
Paleta: CDA (#002b54 / #0056a7 / #0096d6 / #d4a017) con estetica FIFA.
Estructura dual: resumen ejecutivo (no tecnico) + informe de datos (tecnico).
"""

import json
import logging
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.WARNING)

from config.wc2026_groups import GROUPS
from src.runtime import load_prediction_runtime, predict_match
from src.simulation.group_simulator import GroupSimulator
from src.simulation.tournament_simulator import TournamentSimulator, R32_SLOTS
from src.models.poisson_model import PoissonGoalModel
from src.prediction_policy import outcome_scoreline
from src.features.feature_builder import MATCH_FEATURE_COLUMNS

OUTPUT_DIR = Path(__file__).parent.parent / "reports"
MODELS_DIR = Path(__file__).parent.parent / "models"
CHARTS_DIR = OUTPUT_DIR / "_charts"
REPORT_SIMS = 10_000
PW = 210  # ancho A4 mm
REPORT_AUTHOR = "Lic. Leandro Saracco"
REPORT_LINKEDIN = "https://www.linkedin.com/in/leandro-saracco/"

# ── Nombres de equipos en castellano ─────────────────────────────────────────
TEAMS_ES: dict[str, str] = {
    "Mexico": "Mexico", "South Africa": "Sudafrica", "South Korea": "Corea del Sur",
    "Czechia": "Republica Checa", "Canada": "Canada", "Switzerland": "Suiza",
    "Qatar": "Qatar", "Bosnia and Herzegovina": "Bosnia y Herzegovina",
    "Brazil": "Brasil", "Morocco": "Marruecos", "Scotland": "Escocia", "Haiti": "Haiti",
    "United States": "Estados Unidos", "Paraguay": "Paraguay", "Australia": "Australia",
    "Turkiye": "Turquia", "Germany": "Alemania", "Ecuador": "Ecuador",
    "Cote d'Ivoire": "Costa de Marfil", "Curacao": "Curazao",
    "Netherlands": "Paises Bajos", "Sweden": "Suecia", "Japan": "Japon", "Tunisia": "Tunez",
    "Belgium": "Belgica", "Egypt": "Egipto", "Iran": "Iran", "New Zealand": "Nueva Zelanda",
    "Spain": "Espana", "Uruguay": "Uruguay", "Saudi Arabia": "Arabia Saudita",
    "Cape Verde": "Cabo Verde", "France": "Francia", "Senegal": "Senegal",
    "Norway": "Noruega", "Iraq": "Irak", "Argentina": "Argentina", "Austria": "Austria",
    "Algeria": "Argelia", "Jordan": "Jordania", "Portugal": "Portugal",
    "Colombia": "Colombia", "Uzbekistan": "Uzbekistan", "DR Congo": "R. D. del Congo",
    "England": "Inglaterra", "Croatia": "Croacia", "Ghana": "Ghana", "Panama": "Panama",
}

def _es(name: str) -> str:
    """Traduce nombre de equipo al castellano."""
    return TEAMS_ES.get(name, name)

# ── Paleta CDA con estetica FIFA ──────────────────────────────────────────────
DARK      = (0,   43,  84)   # #002b54 — fondo oscuro principal
PRIMARY   = (0,   86, 167)   # #0056a7 — azul corporativo
SECONDARY = (0,  150, 214)   # #0096d6 — azul acento
ACCENT    = (0,  168, 224)   # #00a8e0 — highlight
LIGHT     = (240, 244, 248)  # #f0f4f8 — fila alternada clara
TEXT_COL  = (26,  35,  50)   # #1a2332 — texto principal
GOLD      = (212, 160,  23)  # #d4a017 — campeones, KPIs
GREEN     = (22,  163,  74)  # ALTO confianza
AMBER     = (210, 120,   0)  # MEDIO confianza
LOW_GRAY  = (94,  111, 141)  # BAJO confianza / texto secundario
WHITE     = (255, 255, 255)
OFF_WHITE = (247, 249, 252)
L_GRAY    = (220, 227, 236)


# ── Helpers de texto y dibujo ─────────────────────────────────────────────────
def _t(value: str) -> str:
    for src, dst in {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u",
        "Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U",
        "ñ":"n","Ñ":"N","ü":"u","Ü":"U",
        "–":"-","—":"-","'":"'","'":"'",
        "“":'"',"”":'"',"•":"-"," ":" ",
        "→":"->","▶":">","°":"","™":"(TM)",
    }.items():
        value = value.replace(src, dst)
    value = unicodedata.normalize("NFKD", str(value))
    return value.encode("latin-1", "ignore").decode("latin-1")


def _c(pdf, w, h=0, txt="", **kw):
    return pdf.cell(w, h, _t(txt), **kw)


def _rgb(pdf, rgb):  pdf.set_fill_color(*rgb)
def _txt(pdf, rgb):  pdf.set_text_color(*rgb)
def _draw(pdf, rgb): pdf.set_draw_color(*rgb)


def _fpdf():
    try:
        from fpdf import FPDF
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])
        from fpdf import FPDF
    return FPDF


def _rule(pdf, color=GOLD, y_offset=0):
    y = pdf.get_y() + y_offset
    _rgb(pdf, color)
    pdf.rect(10, y, PW - 20, 0.7, "F")
    pdf.set_y(y + 1.5)


def _section_header(pdf, title, subtitle="", color=PRIMARY):
    """Barra de seccion con fondo de color, acento dorado a la izquierda."""
    y = pdf.get_y()
    _rgb(pdf, color)
    pdf.rect(0, y, PW, 13, "F")
    _rgb(pdf, GOLD)
    pdf.rect(0, y, 3, 13, "F")
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_xy(7, y + 1.5)
    _c(pdf, PW - 14, 6, title.upper(), ln=True)
    if subtitle:
        _txt(pdf, ACCENT)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_x(7)
        _c(pdf, PW - 14, 5, subtitle, ln=True)
    _txt(pdf, TEXT_COL)
    _rgb(pdf, WHITE)
    pdf.ln(2)


def _page_footer(pdf, page_label=""):
    """Pie de pagina al final del contenido de la pagina actual."""
    previous_auto_page_break = getattr(pdf, "auto_page_break", True)
    previous_margin = getattr(pdf, "b_margin", 0)
    pdf.set_auto_page_break(False)
    pdf.set_y(-16)
    _rgb(pdf, L_GRAY)
    pdf.rect(10, pdf.get_y(), PW - 20, 0.7, "F")
    pdf.set_y(pdf.get_y() + 1.5)
    _txt(pdf, LOW_GRAY)
    pdf.set_font("Helvetica", "", 6)
    _c(pdf, PW - 20, 4,
       f"prode-ML  |  FIFA World Cup 2026  |  {page_label}  |  "
       f"{datetime.now().strftime('%d/%m/%Y')}",
       ln=True, align="C")
    _c(pdf, PW - 20, 4, f"{REPORT_AUTHOR}  |  LinkedIn: {REPORT_LINKEDIN}", ln=True, align="C")
    _txt(pdf, TEXT_COL)
    pdf.set_auto_page_break(previous_auto_page_break, margin=previous_margin)


# ── Helpers de prediccion ─────────────────────────────────────────────────────
def _outcome(r):
    pa, pd_, pb = r.get("p_win_a", .38), r.get("p_draw", .24), r.get("p_win_b", .38)
    if pa >= pd_ and pa >= pb: return "win_a"
    if pb >= pd_ and pb >= pa: return "win_b"
    return "draw"


def _scoreline(r, outcome):
    scoreline = outcome_scoreline(r, outcome)
    if scoreline is None:
        return 0, 0
    return int(scoreline[0]), int(scoreline[1])


def _match_summary(r, team_a, team_b):
    out = _outcome(r)
    ga, gb = _scoreline(r, out)
    conf = r.get("confidence", "BAJO")
    xa, xb = r.get("xg_a", 0), r.get("xg_b", 0)
    col_conf = GREEN if conf == "ALTO" else (AMBER if conf == "MEDIO" else LOW_GRAY)
    a16, b16 = _es(team_a)[:16], _es(team_b)[:16]
    if   out == "win_a": titular = f"{a16.upper()} GANA"; score = f"{ga}-{gb}"
    elif out == "win_b": titular = f"{b16.upper()} GANA"; score = f"{gb}-{ga}"
    else:                titular = "EMPATE";              score = f"{ga}-{gb}"
    return titular, score, f"xG {xa:.1f}-{xb:.1f}", conf, col_conf


def _pdf_text(value: str) -> str:
    """Backward-compatible test helper for PDF-safe text."""
    return _t(value)


def _match_row(pdf, team_a: str, team_b: str, result: dict) -> None:
    """Backward-compatible compact row helper used by report tests."""
    titular, score, xg_txt, conf, _ = _match_summary(result, team_a, team_b)
    _c(pdf, 0, 5, f"{team_a} vs {team_b} | {titular} | {score} | {xg_txt} | {conf}", ln=True)


def _metadata_feature_count(meta: dict) -> int:
    return int(meta.get("n_features") or len(meta.get("features") or MATCH_FEATURE_COLUMNS))


def _chart_theme():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.facecolor": "#f7f9fc",
        "figure.facecolor": "white",
        "axes.edgecolor": "#dce3ec",
        "axes.labelcolor": "#1a2332",
        "xtick.color": "#5e6f8d",
        "ytick.color": "#5e6f8d",
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "font.size": 8,
        "grid.color": "#dce3ec",
        "grid.linewidth": 0.7,
    })
    return plt


def _save_bar_chart(
    path: Path,
    labels: list[str],
    values: list[float],
    title: str,
    x_label: str = "",
    value_format: str = "{:.1%}",
) -> Path | None:
    if not labels or not values:
        return None
    plt = _chart_theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=180)
    colors = ["#0056a7" if i else "#d4a017" for i in range(len(labels))]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.58)
    ax.set_title(title, loc="left", color="#002b54", pad=8)
    ax.set_xlabel(x_label)
    ax.grid(axis="x", alpha=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for idx, value in enumerate(values[::-1]):
        ax.text(value + max(values) * 0.015, idx, value_format.format(value), va="center", color="#1a2332", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_count_chart(path: Path, counts: Counter, title: str) -> Path | None:
    if not counts:
        return None
    labels, values = zip(*counts.most_common(10))
    plt = _chart_theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=180)
    ax.bar(labels, values, color="#0096d6", width=0.62)
    ax.set_title(title, loc="left", color="#002b54", pad=8)
    ax.set_ylabel("partidos")
    ax.grid(axis="y", alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.3, str(value), ha="center", color="#1a2332", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_probability_scatter(path: Path, all_partidos: dict) -> Path | None:
    rows = []
    for gname, partidos in all_partidos.items():
        for ea, eb, res in partidos:
            if res is None:
                continue
            s = res.get("outcome_scoreline") or outcome_scoreline(res)
            total_goals = (int(s[0]) + int(s[1])) if s else 0
            rows.append((
                float(res.get("xg_a", 0)) + float(res.get("xg_b", 0)),
                max(float(res.get("p_win_a", 0)), float(res.get("p_draw", 0)), float(res.get("p_win_b", 0))),
                total_goals,
                gname,
            ))
    if not rows:
        return None
    plt = _chart_theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=180)
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    sizes = [45 + r[2] * 28 for r in rows]
    ax.scatter(xs, ys, s=sizes, c="#0056a7", alpha=0.68, edgecolors="white", linewidths=0.8)
    ax.axvline(2.5, color="#d4a017", linewidth=1.2, linestyle="--")
    ax.set_title("Volumen de goles esperados vs confianza del pronostico", loc="left", color="#002b54", pad=8)
    ax.set_xlabel("xG total del partido")
    ax.set_ylabel("probabilidad del resultado predicho")
    ax.grid(alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_line_chart(path: Path, rows: list[dict], title: str) -> Path | None:
    if not rows:
        return None
    xs = [float(r.get("threshold", 0)) for r in rows]
    ys = [float(r.get("accuracy", 0)) for r in rows]
    ns = [int(r.get("n", 0)) for r in rows]
    plt = _chart_theme()
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=180)
    ax.plot(xs, ys, color="#0056a7", marker="o", linewidth=2)
    ax.axhline(1 / 3, color="#d4a017", linewidth=1.2, linestyle="--", label="azar 33%")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y + 0.018, f"n={n}", ha="center", fontsize=7, color="#5e6f8d")
    ax.set_title(title, loc="left", color="#002b54", pad=8)
    ax.set_xlabel("umbral de confianza")
    ax.set_ylabel("accuracy validacion")
    ax.set_ylim(0.25, max(0.9, max(ys) + 0.08))
    ax.grid(alpha=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _embed_chart(pdf, path: Path | None, title: str, caption: str, width: float = 185) -> None:
    if path is None or not Path(path).exists():
        return
    if pdf.get_y() > 210:
        pdf.add_page()
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 8.5)
    _c(pdf, 0, 6, f"  {title}", ln=True)
    x = (PW - width) / 2
    pdf.image(str(path), x=x, w=width)
    _txt(pdf, LOW_GRAY)
    pdf.set_font("Helvetica", "I", 6.5)
    pdf.set_x(12)
    pdf.multi_cell(PW - 24, 4, _t(caption))
    _txt(pdf, TEXT_COL)
    pdf.ln(2)


def _build_report_charts(meta: dict, tourn_df, all_group_dfs: dict, all_partidos: dict) -> dict[str, Path | None]:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    charts: dict[str, Path | None] = {}

    if tourn_df is not None and not tourn_df.empty:
        top = tourn_df.sort_values("p_champion", ascending=False).head(8)
        charts["champions"] = _save_bar_chart(
            CHARTS_DIR / "top_champions.png",
            [_es(str(t)) for t in top["team"].tolist()],
            [float(v) for v in top["p_champion"].tolist()],
            "Top candidatos al titulo por simulacion Monte Carlo",
            "probabilidad de campeon",
        )

    group_labels = []
    group_values = []
    for gname, gdf in all_group_dfs.items():
        if gdf is None or gdf.empty:
            continue
        leader = gdf.sort_values("qualify_direct_prob", ascending=False).iloc[0]
        group_labels.append(f"Grupo {gname}: {_es(str(leader.get('team', '')))}")
        group_values.append(float(leader.get("qualify_direct_prob", 0)))
    charts["group_leaders"] = _save_bar_chart(
        CHARTS_DIR / "group_leaders.png",
        group_labels,
        group_values,
        "Fortaleza del favorito de cada grupo",
        "probabilidad de clasificar directo",
    )

    score_counts = Counter()
    for partidos in all_partidos.values():
        for _, _, res in partidos:
            if res is None:
                continue
            s = res.get("outcome_scoreline") or outcome_scoreline(res)
            if s:
                score_counts[f"{int(s[0])}-{int(s[1])}"] += 1
    charts["scorelines"] = _save_count_chart(
        CHARTS_DIR / "scoreline_distribution.png",
        score_counts,
        "Distribucion de marcadores recomendados en fase de grupos",
    )
    charts["xg_scatter"] = _save_probability_scatter(CHARTS_DIR / "xg_confidence_scatter.png", all_partidos)

    team_stats: dict[str, dict[str, float]] = {}
    for partidos in all_partidos.values():
        for team_a, team_b, res in partidos:
            if res is None:
                continue
            xg_a = float(res.get("xg_a", 0))
            xg_b = float(res.get("xg_b", 0))
            for team, xgf, xga in [(team_a, xg_a, xg_b), (team_b, xg_b, xg_a)]:
                row = team_stats.setdefault(team, {"xgf": 0.0, "xga": 0.0, "matches": 0.0})
                row["xgf"] += xgf
                row["xga"] += xga
                row["matches"] += 1
    team_rows = [
        (
            team,
            values["xgf"] / max(values["matches"], 1.0),
            values["xga"] / max(values["matches"], 1.0),
        )
        for team, values in team_stats.items()
    ]
    top_attacks = sorted(team_rows, key=lambda row: row[1], reverse=True)[:10]
    best_defenses = sorted(team_rows, key=lambda row: row[2])[:10]
    worst_defenses = sorted(team_rows, key=lambda row: row[2], reverse=True)[:10]
    charts["top_attacks"] = _save_bar_chart(
        CHARTS_DIR / "top_attacks_xg.png",
        [_es(team) for team, _, _ in top_attacks],
        [xgf for _, xgf, _ in top_attacks],
        "Ataques con mayor xG promedio en fase de grupos",
        "xG a favor por partido",
        "{:.2f}",
    )
    charts["best_defenses"] = _save_bar_chart(
        CHARTS_DIR / "best_defenses_xga.png",
        [_es(team) for team, _, _ in best_defenses],
        [xga for _, _, xga in best_defenses],
        "Defensas con menor xGA promedio permitido",
        "xGA permitido por partido",
        "{:.2f}",
    )
    charts["worst_defenses"] = _save_bar_chart(
        CHARTS_DIR / "worst_defenses_xga.png",
        [_es(team) for team, _, _ in worst_defenses],
        [xga for _, _, xga in worst_defenses],
        "Defensas mas vulnerables por xGA permitido",
        "xGA permitido por partido",
        "{:.2f}",
    )

    weights = meta.get("voting_weights") or {}
    if weights:
        labels = [k.upper() for k in weights.keys()]
        values = [float(v) for v in weights.values()]
        charts["weights"] = _save_bar_chart(
            CHARTS_DIR / "ensemble_weights.png",
            labels,
            values,
            "Pesos normalizados del ensemble",
            "peso en voting",
        )

    bins = (meta.get("confidence_thresholds") or {}).get("validation_bins") or []
    charts["calibration"] = _save_line_chart(
        CHARTS_DIR / "confidence_accuracy.png",
        bins,
        "Accuracy por umbral de confianza en validacion",
    )

    return charts


def _tournament_section(pdf, tourn_df) -> None:
    """Backward-compatible compact tournament helper used by report tests."""
    if tourn_df is None or tourn_df.empty:
        return
    for _, row in tourn_df.iterrows():
        _c(pdf, 0, 5, str(row.get("team", "")), ln=False)
        _c(pdf, 0, 5, f"{row.get('p_champion', 0):.1%}", ln=False)
        _c(pdf, 0, 5, f"{row.get('p_finalist', 0):.1%}", ln=True)


def _1_in_x(prob: float) -> str:
    if prob <= 0: return "raramente"
    n = round(1.0 / prob)
    return f"1 de cada {n} simulaciones"


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 1 — PORTADA
# ══════════════════════════════════════════════════════════════════════════════
def _portada(pdf, meta: dict):
    # Fondo oscuro superior
    _rgb(pdf, DARK)
    pdf.rect(0, 0, PW, 68, "F")
    # Banda de color primario
    _rgb(pdf, PRIMARY)
    pdf.rect(0, 68, PW, 4, "F")
    # Acento dorado
    _rgb(pdf, GOLD)
    pdf.rect(0, 72, PW, 1.5, "F")

    # Eyebrow
    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_xy(0, 10)
    _c(pdf, PW, 6, "FIFA WORLD CUP 2026 (TM)  |  Canada  Mexico  Estados Unidos", ln=True, align="C")

    # Titulo principal
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 30)
    pdf.set_xy(0, 20)
    _c(pdf, PW, 16, "prode-ML", ln=True, align="C")

    pdf.set_font("Helvetica", "", 13)
    _c(pdf, PW, 8, "Informe de Predicciones", ln=True, align="C")

    _txt(pdf, ACCENT)
    pdf.set_font("Helvetica", "", 9)
    _c(pdf, PW, 6, "Machine Learning aplicado al futbol internacional", ln=True, align="C")

    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "I", 7.5)
    pdf.set_xy(0, 58)
    _c(pdf, PW, 5, f"Generado: {datetime.now().strftime('%d de %B de %Y  |  %H:%M')}", ln=True, align="C")

    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 7.5)
    pdf.set_xy(0, 63)
    _c(pdf, PW, 5, f"{REPORT_AUTHOR}  |  LinkedIn: {REPORT_LINKEDIN}", ln=True, align="C")

    pdf.set_y(80)
    _txt(pdf, TEXT_COL)

    # ── KPIs ────────────────────────────────────────────────────────────────
    acc   = meta.get("accuracy_voting", 0)
    hi    = meta.get("accuracy_high_confidence", 0)
    n_hi  = meta.get("n_high_confidence_matches", 0)
    elo   = meta.get("accuracy_high_elo_diff_200", 0)
    n_elo = meta.get("n_high_elo_matches", 0)
    n_src = meta.get("n_source_matches", 0)

    def _kpi(x, y, color, valor, label, sub):
        _rgb(pdf, color)
        pdf.rect(x, y, 58, 28, "F")
        _rgb(pdf, DARK); _draw(pdf, GOLD)
        pdf.rect(x, y + 24, 58, 4, "F")
        _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_xy(x, y + 2)
        _c(pdf, 58, 12, valor, ln=False, align="C")
        pdf.set_font("Helvetica", "B", 6)
        pdf.set_xy(x, y + 15)
        _c(pdf, 58, 4, label.upper(), ln=False, align="C")
        _txt(pdf, GOLD)
        pdf.set_font("Helvetica", "", 5.5)
        pdf.set_xy(x, y + 25)
        _c(pdf, 58, 3, sub, ln=False, align="C")
        _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    y0 = 82
    _kpi(11,  y0, PRIMARY,   f"{acc:.0%}",  "Precision global",     "vs 33% al azar")
    _kpi(76,  y0, GREEN,     f"{hi:.0%}",   "Confianza ALTO",        f"sobre {n_hi} partidos")
    _kpi(141, y0, SECONDARY, f"{n_src:,}",  "Partidos analizados",   "desde el ano 2000")
    pdf.set_y(y0 + 34)
    pdf.ln(4)

    # ── Indice de secciones ─────────────────────────────────────────────────
    _rule(pdf, GOLD)
    pdf.ln(2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 10)
    _c(pdf, 0, 7, "  Contenido de este informe", ln=True)

    secciones = [
        ("1", "Resumen Ejecutivo",       "Que hace el modelo y hallazgos clave — para cualquier lector"),
        ("2", "Fase de Grupos (A-L)",    "Predicciones de clasificacion y partidos, grupo por grupo"),
        ("3", "Fase Eliminatoria",       "Round of 32, R16, cuartos, semis y final predichos"),
        ("4", "Informe Tecnico",         "Arquitectura del modelo, features y metodologia — para data scientists"),
        ("5", "Resultados del Modelo",   "Metricas de entrenamiento, calibracion y limitaciones"),
    ]
    _txt(pdf, TEXT_COL)
    for num, titulo, desc in secciones:
        pdf.set_font("Helvetica", "B", 8)
        _rgb(pdf, LIGHT)
        pdf.rect(10, pdf.get_y(), PW - 20, 8, "F")
        _rgb(pdf, PRIMARY)
        pdf.rect(10, pdf.get_y(), 8, 8, "F")
        _txt(pdf, WHITE)
        pdf.set_xy(11, pdf.get_y() + 1)
        pdf.set_font("Helvetica", "B", 7)
        _c(pdf, 7, 6, num, ln=False, align="C")
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_x(21)
        _c(pdf, 60, 8, titulo, ln=False)
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 7)
        _c(pdf, PW - 85, 8, desc, ln=True)
        _rgb(pdf, WHITE)
    pdf.ln(4)

    _txt(pdf, LOW_GRAY)
    pdf.set_font("Helvetica", "I", 6.5)
    _c(pdf, 0, 5,
       "Las predicciones son probabilidades estadisticas basadas en datos historicos. "
       "El futbol siempre puede sorprender.", ln=True, align="C")
    _txt(pdf, TEXT_COL)

    _page_footer(pdf, "Portada")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 2A — RESUMEN EJECUTIVO: COMO FUNCIONA
# ══════════════════════════════════════════════════════════════════════════════
def _resumen_ejecutivo_a(pdf, meta: dict):
    _section_header(pdf, "Resumen Ejecutivo — Como Funciona el Modelo",
                    "Para cualquier lector, sin conocimientos tecnicos previos")

    # Introduccion coloquial
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9.5)
    _c(pdf, 0, 7, "  Imaginate que contratas a 6 analistas de futbol especializados...", ln=True)

    intro = (
        "Cada analista tiene una forma distinta de ver el juego. Antes de cada partido, "
        "los seis estudian los datos historicos de ambas selecciones y hacen su pronostico. "
        "Despues votan, y el voto de cada uno vale segun que tan bien le vino yendo ultimamente. "
        "La prediccion final es el resultado de esa votacion ponderada."
    )
    _txt(pdf, TEXT_COL)
    pdf.set_font("Helvetica", "", 8.5)
    pdf.set_x(12)
    pdf.multi_cell(PW - 24, 5.5, _t(intro))
    pdf.ln(3)

    # Los 6 analistas
    analistas = [
        ("Random Forest",
         "El historiador",
         f"Peso: {meta.get('voting_weights',{}).get('rf',0):.0%}",
         "Estudiou 8.000+ partidos internacionales buscando patrones. "
         "Si Argentina gano el 80% de las veces que tuvo un Elo 200 puntos mayor, el lo recuerda. "
         "Es muy bueno con reglas solidas que se repiten en el tiempo."),
        ("XGBoost",
         "El estratega adaptativo",
         f"Peso: {meta.get('voting_weights',{}).get('xgb',0):.0%}",
         "Aprende de sus propios errores de forma iterativa. "
         "Cada vez que se equivoca en un pronostico, ajusta su modelo para mejorar. "
         "Especialista en captar tendencias recientes y diferencias de nivel entre equipos."),
        ("LightGBM",
         "El analista rapido",
         f"Peso: {meta.get('voting_weights',{}).get('lgbm',0):.0%}",
         "Similar al anterior pero mas eficiente con grandes volumenes de datos. "
         "Procesa miles de variables rapidamente y encuentra senales sutiles "
         "que los otros pueden pasar por alto."),
        ("CatBoost",
         "El especialista en contexto",
         f"Peso: {meta.get('voting_weights',{}).get('catboost',0):.0%}",
         "Muy bueno distinguiendo el contexto del partido: "
         "no es lo mismo un amistoso que un partido mundialista. "
         "Tambien considera si el equipo juega en casa o en cancha neutral."),
        ("Elo",
         "El matematico puro",
         f"Peso: {meta.get('voting_weights',{}).get('elo',0):.0%}",
         "Usa una formula matematica probada (la misma que usa el ajedrez) "
         "basada en el historial de victorias y derrotas de cada seleccion. "
         "Simple pero sorprendentemente efectivo para identificar a los favoritos claros."),
        ("Poisson",
         "El contador de goles",
         f"Peso: {meta.get('voting_weights',{}).get('poisson',0):.0%}",
         "En lugar de preguntar quien gana, pregunta cuantos goles espera "
         "que marque cada equipo y de ahi deduce el resultado. "
         "Aporta una perspectiva distinta a los demas modelos."),
    ]

    pdf.set_y(pdf.get_y())
    for i, (nombre, rol, peso, desc) in enumerate(analistas):
        bg = LIGHT if i % 2 == 0 else OFF_WHITE
        y_start = pdf.get_y()
        _rgb(pdf, bg)
        pdf.rect(10, y_start, PW - 20, 16, "F")
        _rgb(pdf, PRIMARY)
        pdf.rect(10, y_start, 3, 16, "F")
        _rgb(pdf, WHITE)

        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_xy(16, y_start + 1.5)
        _c(pdf, 60, 5, nombre, ln=False)
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "I", 7)
        _c(pdf, 50, 5, f"({rol})", ln=False)
        _txt(pdf, GOLD)
        pdf.set_font("Helvetica", "B", 7)
        _c(pdf, 35, 5, peso, ln=True)

        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_x(16)
        pdf.multi_cell(PW - 28, 4.5, _t(desc))
        if pdf.get_y() < y_start + 16:
            pdf.set_y(y_start + 16)
        pdf.ln(1)

    _rgb(pdf, WHITE)
    pdf.ln(2)

    # Confianza
    _rule(pdf, PRIMARY)
    pdf.ln(1)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9.5)
    _c(pdf, 0, 7, "  Como saber si el modelo esta seguro o no", ln=True)

    _txt(pdf, TEXT_COL)
    pdf.set_font("Helvetica", "", 8.5)
    hi_acc = meta.get("accuracy_high_confidence", 0)
    n_hi   = meta.get("n_high_confidence_matches", 0)
    pdf.set_x(12)
    txt = (
        f"No todas las predicciones valen lo mismo. Cuando el modelo identifica una "
        f"ventaja clara, lo marca como ALTO. En esos casos, historicamente acerto "
        f"{hi_acc:.0%} de las veces (sobre {n_hi} partidos de prueba). "
        f"Es como un pronosticador que solo habla cuando esta muy seguro."
    )
    pdf.multi_cell(PW - 24, 5.5, _t(txt))
    pdf.ln(3)

    niveles = [
        (GREEN,    "ALTO",  "> 70% de probabilidad en un resultado.",
         "El modelo ve una ventaja clara. Confiabilidad historica: ~82%."),
        (AMBER,    "MEDIO", "60-70% de probabilidad.",
         "Hay senal, pero el partido tiene incertidumbre. Puede ir a cualquier lado."),
        (LOW_GRAY, "BAJO",  "< 60% de probabilidad.",
         "El modelo lo ve como un partido parejo. Una sorpresa es posible."),
    ]
    for col, label, definition, note in niveles:
        y = pdf.get_y()
        _rgb(pdf, col)
        pdf.rect(10, y, 22, 11, "F")
        _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_xy(11, y + 1.5)
        _c(pdf, 20, 5, label, ln=False, align="C")
        pdf.set_font("Helvetica", "", 5.5)
        _c(pdf, 20, 5, "confianza", ln=True, align="C")
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 7.5)
        pdf.set_xy(36, y + 1)
        _c(pdf, PW - 48, 5, definition, ln=True)
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_x(36)
        _c(pdf, PW - 48, 5, note, ln=True)
        _rgb(pdf, WHITE)
        pdf.ln(2)

    _page_footer(pdf, "Seccion 1A")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 2B — RESUMEN EJECUTIVO: HALLAZGOS CLAVE
# ══════════════════════════════════════════════════════════════════════════════
def _resumen_ejecutivo_b(pdf, meta: dict, tourn_df, all_group_dfs: dict, all_partidos: dict, charts: dict | None = None):
    _section_header(pdf, "Resumen Ejecutivo — Hallazgos Clave del Mundial",
                    "Los pronosticos mas importantes en lenguaje directo")

    # Candidatos al titulo
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9.5)
    _c(pdf, 0, 7, "  Principales candidatos al titulo", ln=True)

    txt_intro = (
        f"El modelo simulo el torneo completo {REPORT_SIMS:,} veces con el bracket oficial FIFA. "
        "En cada simulacion, los 48 equipos compiten desde la fase de grupos hasta la final. "
        "Ningun equipo esta garantizado de ganar. En este informe, candidato significa "
        "tener al menos 8% de probabilidad estadistica de campeonar:"
    )
    _txt(pdf, TEXT_COL)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_x(12)
    pdf.multi_cell(PW - 24, 5, _t(txt_intro))
    pdf.ln(3)

    # Tabla de candidatos estadisticos: no todo top 10 es candidato.
    top10 = None
    if tourn_df is not None:
        candidates = tourn_df[tourn_df["p_champion"] >= 0.08]
        top10 = candidates.head(10) if not candidates.empty else tourn_df.head(5)
    if top10 is not None:
        _rgb(pdf, DARK); _txt(pdf, GOLD)
        pdf.set_font("Helvetica", "B", 6.5)
        ths = ["#", "Seleccion", "Campeon", "Final", "Semifinal", "Lectura coloquial"]
        tws = [8, 42, 18, 18, 18, 86]
        for h, w in zip(ths, tws):
            pdf.set_x(10 + sum(tws[:ths.index(h)]))
            _c(pdf, w, 5.5, h, ln=False, align="C")
        pdf.ln()
        _rgb(pdf, WHITE); _txt(pdf, TEXT_COL)

        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            bg = LIGHT if rank % 2 == 0 else OFF_WHITE
            _rgb(pdf, bg)
            pdf.rect(10, pdf.get_y(), PW - 20, 7, "F")
            if rank == 1: _rgb(pdf, GOLD); pdf.rect(10, pdf.get_y(), 3, 7, "F")
            _rgb(pdf, WHITE)

            pchamp = row.get("p_champion", 0)
            pfin   = row.get("p_finalist", 0)
            psemi  = row.get("p_semifinal", 0)

            lectura = _1_in_x(pchamp) + " la da campeon" if pchamp > 0 else "-"
            bold = rank <= 3
            pdf.set_font("Helvetica", "B" if bold else "", 6.5)
            if rank == 1: _txt(pdf, (140, 90, 0))
            else:         _txt(pdf, TEXT_COL)
            pdf.set_x(10)
            _c(pdf, tws[0], 7, str(rank), ln=False, align="C")
            _c(pdf, tws[1], 7, f"  {_es(str(row.get('team','')))[:22]}", ln=False)
            _txt(pdf, TEXT_COL)
            pdf.set_font("Helvetica", "", 6.5)
            _c(pdf, tws[2], 7, f"{pchamp:.1%}", ln=False, align="C")
            _c(pdf, tws[3], 7, f"{pfin:.1%}",   ln=False, align="C")
            _c(pdf, tws[4], 7, f"{psemi:.1%}",  ln=False, align="C")
            _txt(pdf, LOW_GRAY)
            pdf.set_font("Helvetica", "", 6)
            _c(pdf, tws[5], 7, lectura, ln=True, align="L")
        _rgb(pdf, WHITE)
    pdf.ln(4)

    charts = charts or {}
    _rule(pdf, SECONDARY)
    pdf.ln(2)
    _embed_chart(
        pdf,
        charts.get("champions"),
        "Lectura visual: carrera por el titulo",
        "El grafico concentra la simulacion completa del bracket. No mide solo fuerza de equipo: tambien captura camino probable, cruces y penales simulados.",
    )
    _embed_chart(
        pdf,
        charts.get("scorelines"),
        "Lectura visual: marcadores comunicados",
        "La distribucion resume los marcadores recomendados para comunicar el resultado. El modo exacto puro se conserva aparte, pero aqui se prioriza coherencia con outcome y volumen xG.",
    )
    _embed_chart(
        pdf,
        charts.get("xg_scatter"),
        "Lectura visual: xG total y confianza",
        "Cada punto es un partido de fase de grupos. Puntos mas grandes indican marcadores recomendados con mayor total de goles.",
    )
    _embed_chart(
        pdf,
        charts.get("top_attacks"),
        "Lectura visual: ataques mas productivos",
        "Este ranking usa xG promedio de los tres partidos de grupo. Sirve para detectar equipos con mayor volumen ofensivo esperado, mas alla del marcador modal.",
    )
    _embed_chart(
        pdf,
        charts.get("best_defenses"),
        "Lectura visual: defensas mas solidas",
        "Menor xGA promedio implica menos goles esperados concedidos. Este grafico ayuda a distinguir candidatos que ganan por control defensivo.",
    )
    _embed_chart(
        pdf,
        charts.get("worst_defenses"),
        "Lectura visual: defensas mas vulnerables",
        "Mayor xGA promedio senala rivales que permiten volumen ofensivo. Esta evidencia habilita marcadores como 2-1, 2-0 o 3-0 cuando el favorito tambien tiene xG alto.",
    )

    # Grupos destacados
    _rule(pdf, PRIMARY)
    pdf.ln(2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9.5)
    _c(pdf, 0, 7, "  Grupos para prestar atencion", ln=True)

    if all_group_dfs:
        # Grupo mas competido (menor prob del lider)
        min_leader_prob = 1.0; min_leader_group = "?"
        max_leader_prob = 0.0; max_leader_group = "?"
        max_upset_group = "?"; max_upset_team = "?"; max_upset_val = 0.0

        for gname, gdf in all_group_dfs.items():
            if gdf is None or gdf.empty: continue
            leader_prob = gdf.iloc[0].get("qualify_direct_prob", 0)
            if leader_prob < min_leader_prob:
                min_leader_prob = leader_prob
                min_leader_group = gname
            if leader_prob > max_leader_prob:
                max_leader_prob = leader_prob
                max_leader_group = gname

        if all_partidos:
            for gname, partidos in all_partidos.items():
                for ea, eb, res in partidos:
                    if res is None: continue
                    ur = res.get("upset_risk", 0)
                    if ur > max_upset_val:
                        max_upset_val = ur
                        max_upset_team = f"{ea} vs {eb}"
                        max_upset_group = gname

        grupos_info = [
            (PRIMARY, "Grupo mas competido",
             f"Grupo {min_leader_group}: el lider solo clasifica en el {min_leader_prob:.0%} "
             f"de las simulaciones. Cualquiera de los equipos puede pasar."),
            (SECONDARY, "Grupo mas predecible",
             f"Grupo {max_leader_group}: el lider clasifica en el {max_leader_prob:.0%} "
             f"de las simulaciones. El favorito tiene una ventaja muy clara."),
            (AMBER, "Partido con mayor riesgo de sorpresa",
             f"Grupo {max_upset_group}: {max_upset_team} "
             f"(indice de sorpresa: {max_upset_val:.0%}). El modelo ve al favorito vulnerable."),
        ]
        for col, label, texto in grupos_info:
            y = pdf.get_y()
            _rgb(pdf, col)
            pdf.rect(10, y, 3, 12, "F")
            _rgb(pdf, LIGHT)
            pdf.rect(13, y, PW - 23, 12, "F")
            _txt(pdf, DARK)
            pdf.set_font("Helvetica", "B", 7.5)
            pdf.set_xy(16, y + 1.5)
            _c(pdf, PW - 28, 5, label, ln=True)
            _txt(pdf, TEXT_COL)
            pdf.set_font("Helvetica", "", 7)
            pdf.set_x(16)
            pdf.multi_cell(PW - 28, 4.5, _t(texto))
            if pdf.get_y() < y + 12: pdf.set_y(y + 12)
            _rgb(pdf, WHITE)
            pdf.ln(2)

    pdf.ln(3)

    # Dato del modelo
    _rule(pdf, SECONDARY)
    pdf.ln(2)
    n_src  = meta.get("n_source_matches", 0)
    n_trn  = meta.get("n_train_samples", 0)
    n_features = _metadata_feature_count(meta)
    acc    = meta.get("accuracy_voting", 0)
    hi_acc = meta.get("accuracy_high_confidence", 0)

    _rgb(pdf, LIGHT)
    pdf.rect(10, pdf.get_y(), PW - 20, 26, "F")
    _rgb(pdf, WHITE)
    pdf.set_xy(14, pdf.get_y() + 2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 8.5)
    _c(pdf, 0, 6, "En numeros: lo que hay detras de estas predicciones", ln=True)
    _txt(pdf, TEXT_COL)
    pdf.set_font("Helvetica", "", 8)
    bullets = [
        f"Se analizaron {n_src:,} partidos internacionales desde el ano 2000.",
        f"El modelo usa {n_trn:,} ejemplos de entrenamiento con {n_features} variables por partido.",
        f"Precision general: {acc:.1%} (el azar seria 33%). Supera al azar en mas del doble.",
        f"Cuando dice ALTO: {hi_acc:.1%} de acierto historico. Mas confiable que muchos expertos.",
        f"El torneo completo se simulo {REPORT_SIMS:,} veces con el bracket oficial FIFA.",
    ]
    for b in bullets:
        pdf.set_x(16)
        _c(pdf, 4, 4.5, "-", ln=False)
        pdf.set_x(20)
        _c(pdf, PW - 32, 4.5, b, ln=True)
    _txt(pdf, TEXT_COL)
    _rgb(pdf, WHITE)

    _page_footer(pdf, "Seccion 1B")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 3 — FASE DE GRUPOS
# ══════════════════════════════════════════════════════════════════════════════
def _pagina_grupo(pdf, nombre, equipos, partidos, group_df):
    # Banner del grupo
    _rgb(pdf, DARK)
    pdf.rect(0, pdf.get_y() - 1, PW, 14, "F")
    _rgb(pdf, GOLD)
    pdf.rect(0, pdf.get_y() - 1, 3.5, 14, "F")
    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_xy(6, pdf.get_y() + 0.5)
    _c(pdf, 18, 13, nombre, ln=False)
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(26, pdf.get_y() + 1.5)
    _c(pdf, PW - 34, 5, "GRUPO " + nombre, ln=True)
    _txt(pdf, ACCENT)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_x(26)
    _c(pdf, PW - 34, 5, "  |  ".join(_es(e).upper() for e in equipos), ln=True)
    _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)
    pdf.ln(2)

    # 1. CLASIFICACION ESTIMADA
    _rgb(pdf, PRIMARY)
    pdf.rect(10, pdf.get_y(), PW - 20, 6, "F")
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(12, pdf.get_y())
    _c(pdf, PW - 24, 6, f"CLASIFICACION ESTIMADA  ({REPORT_SIMS:,} simulaciones)", ln=True, align="L")
    _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)
    pdf.ln(0.5)

    if group_df is not None and not group_df.empty:
        group_chart = _save_bar_chart(
            CHARTS_DIR / f"group_{nombre}_qualify.png",
            [_es(str(t)) for t in group_df.sort_values("qualify_direct_prob", ascending=True)["team"].tolist()],
            [float(v) for v in group_df.sort_values("qualify_direct_prob", ascending=True)["qualify_direct_prob"].tolist()],
            f"Grupo {nombre}: probabilidad de clasificar directo",
            "probabilidad",
        )
        _embed_chart(
            pdf,
            group_chart,
            "Grafico de clasificacion",
            "La barra muestra la probabilidad simulada de terminar entre los dos primeros del grupo.",
            width=175,
        )

        _rgb(pdf, DARK); _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 6.5)
        ws = [8, 46, 20, 20, 20, 20, 26, 20]
        hs = ["#", "Equipo", "1ro %", "2do %", "3ro %", "4to %", "Clasifica", "Pts prom"]
        for h, w in zip(hs, ws):
            _c(pdf, w, 5.5, h, ln=False, align="C")
        pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

        for pos, (_, row) in enumerate(group_df.iterrows(), 1):
            bg = LIGHT if pos % 2 == 0 else OFF_WHITE
            _rgb(pdf, bg)
            pdf.rect(10, pdf.get_y(), PW - 20, 5.5, "F")
            _rgb(pdf, WHITE)
            pdf.set_font("Helvetica", "B" if pos <= 2 else "", 6.5)
            if pos == 1:   _txt(pdf, (140, 90, 0))
            elif pos == 2: _txt(pdf, PRIMARY)
            else:          _txt(pdf, LOW_GRAY)
            _c(pdf, ws[0], 5.5, str(pos), ln=False, align="C")
            _txt(pdf, TEXT_COL)
            _c(pdf, ws[1], 5.5, "  " + _es(str(row.get("team","")))[:22], ln=False, align="L")
            vals = [
                f"{row.get('prob_1st', 0):.0%}",
                f"{row.get('prob_2nd', 0):.0%}",
                f"{row.get('prob_3rd', 0):.0%}",
                f"{row.get('prob_4th', 0):.0%}",
                f"{row.get('qualify_direct_prob', 0):.0%}",
                f"{row.get('avg_pts', 0):.1f}",
            ]
            for v, w in zip(vals, ws[2:]):
                _c(pdf, w, 5.5, v, ln=False, align="C")
            pdf.ln()
    pdf.ln(3)

    # 2. PREDICCION DE PARTIDOS
    _rgb(pdf, SECONDARY)
    pdf.rect(10, pdf.get_y(), PW - 20, 6, "F")
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(12, pdf.get_y())
    _c(pdf, PW - 24, 6, "PREDICCION DE PARTIDOS", ln=True, align="L")
    _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)
    pdf.ln(0.5)

    for idx, (ea, eb, res) in enumerate(partidos):
        if res is None:
            pdf.set_font("Helvetica", "I", 7)
            _c(pdf, 0, 6, f"  {ea} vs {eb}  --  sin datos", ln=True)
            continue
        titular, score, xg_txt, conf, col_conf = _match_summary(res, ea, eb)
        bg = LIGHT if idx % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 8, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 7)
        _c(pdf, 54, 8, f"  {_es(ea)[:19]} vs {_es(eb)[:19]}", ln=False)
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 9)
        _c(pdf, 48, 8, titular, ln=False)
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "B", 13)
        _c(pdf, 18, 8, score, ln=False, align="C")
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 6.5)
        _c(pdf, 24, 8, xg_txt, ln=False, align="C")
        bx = pdf.get_x() + 1; by = pdf.get_y() + 1.5
        _rgb(pdf, col_conf)
        pdf.rect(bx, by, 22, 5, "F")
        _rgb(pdf, WHITE); _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 6)
        _c(pdf, 24, 8, conf, ln=True, align="C")
        _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)
    pdf.ln(3)

    # 3. DETALLE ESTADISTICO
    _rgb(pdf, DARK); _txt(pdf, GOLD)
    pdf.rect(10, pdf.get_y(), PW - 20, 5.5, "F")
    pdf.set_font("Helvetica", "B", 6.5)
    pdf.set_xy(12, pdf.get_y())
    _c(pdf, PW - 24, 5.5, "DETALLE ESTADISTICO", ln=True, align="L")
    _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)
    pdf.ln(0.5)

    _rgb(pdf, L_GRAY); _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 5.5)
    col_dw = [52, 14, 14, 14, 20, 28, 14, 14]
    col_dh = ["Partido", "Gana A", "Empate", "Gana B", "xG A-B", "Marcador prob.", "Conf.", "Sorpresa"]
    for h, w in zip(col_dh, col_dw):
        _c(pdf, w, 5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    for idx, (ea, eb, res) in enumerate(partidos):
        if res is None: continue
        bg = LIGHT if idx % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 4.5, "F")
        _rgb(pdf, WHITE)
        s = res.get("outcome_scoreline") or res.get("most_likely_scoreline") or outcome_scoreline(res)
        if s:
            ga = int(s.get("goals_a", 0) if isinstance(s, dict) else s[0])
            gb = int(s.get("goals_b", 0) if isinstance(s, dict) else s[1])
            pr = float(s.get("probability", 0) if isinstance(s, dict) else s[2])
            score_t = f"{ga}-{gb} ({pr:.0%})"
        else:
            score_t = "-"
        pdf.set_font("Helvetica", "", 5.5)
        row_vals = [
            f"  {_es(ea)[:19]} v {_es(eb)[:17]}",
            f"{res.get('p_win_a', 0):.0%}",
            f"{res.get('p_draw', 0):.0%}",
            f"{res.get('p_win_b', 0):.0%}",
            f"{res.get('xg_a', 0):.1f} - {res.get('xg_b', 0):.1f}",
            score_t,
            res.get("confidence", "?"),
            f"{res.get('upset_risk', 0):.0%}",
        ]
        aligns = ["L", "C", "C", "C", "C", "C", "C", "C"]
        for v, w, al in zip(row_vals, col_dw, aligns):
            _c(pdf, w, 4.5, v, ln=False, align=al)
        pdf.ln()

    # Resumen coloquial del grupo
    pdf.ln(3)
    if group_df is not None and not group_df.empty:
        _rgb(pdf, LIGHT)
        pdf.rect(10, pdf.get_y(), PW - 20, 14, "F")
        _rgb(pdf, GOLD); pdf.rect(10, pdf.get_y(), 2.5, 14, "F")
        _rgb(pdf, WHITE)
        pdf.set_xy(16, pdf.get_y() + 2)
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 7)
        _c(pdf, 0, 5, "Resumen del grupo en palabras", ln=True)

        sorted_gdf = group_df.sort_values("qualify_direct_prob", ascending=False)
        t1 = _es(str(sorted_gdf.iloc[0].get("team",""))) if len(sorted_gdf) > 0 else "?"
        p1 = sorted_gdf.iloc[0].get("qualify_direct_prob", 0)
        t2 = _es(str(sorted_gdf.iloc[1].get("team",""))) if len(sorted_gdf) > 1 else "?"
        p2 = sorted_gdf.iloc[1].get("qualify_direct_prob", 0)
        t3 = _es(str(sorted_gdf.iloc[2].get("team",""))) if len(sorted_gdf) > 2 else "?"
        p3 = sorted_gdf.iloc[2].get("qualify_direct_prob", 0)

        resumen = (
            f"{t1} clasifica en el {p1:.0%} de las simulaciones como favorito del grupo. "
            f"{t2} acompana con {p2:.0%} de probabilidad. "
            f"{t3} necesita una actuacion solida para alcanzarlos ({p3:.0%})."
        )
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_x(16)
        pdf.multi_cell(PW - 28, 4.5, _t(resumen))
        _rgb(pdf, WHITE)

    pdf.ln(4)
    _page_footer(pdf, f"Grupo {nombre}")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 4 — FASE ELIMINATORIA
# ══════════════════════════════════════════════════════════════════════════════
def _resolve_bracket_slot(slot: str, likely_winners: dict, likely_runners: dict) -> str:
    if slot.startswith("w_"): return likely_winners.get(slot[2:], f"1ro Grupo {slot[2:]}")
    if slot.startswith("r_"): return likely_runners.get(slot[2:], f"2do Grupo {slot[2:]}")
    return f"Mejor 3ro (clasificado)"


def _seccion_eliminatorias(pdf, all_group_dfs: dict, runtime, tourn_df):
    _section_header(pdf, "Fase Eliminatoria — Round of 32",
                    "Cruces del bracket oficial FIFA basados en los clasificados mas probables de cada grupo",
                    DARK)

    # Derivar clasificados mas probables por grupo
    likely_winners = {}
    likely_runners = {}
    for gname, gdf in all_group_dfs.items():
        if gdf is None or gdf.empty: continue
        sdf = gdf.sort_values("qualify_direct_prob", ascending=False)
        if len(sdf) >= 1: likely_winners[gname] = str(sdf.iloc[0]["team"])
        if len(sdf) >= 2: likely_runners[gname] = str(sdf.iloc[1]["team"])

    # Cabecera de la tabla R32
    _rgb(pdf, PRIMARY); _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 6.5)
    ths_r32 = ["Partido", "Equipo A", "vs", "Equipo B", "Resultado predicho", "Marcador", "Conf."]
    tws_r32 = [10, 48, 6, 48, 40, 18, 14]
    for h, w in zip(ths_r32, tws_r32):
        _c(pdf, w, 5.5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    pod_labels = ["A-B", "A-B", "C-D", "C-D", "E-F", "E-F",
                  "G-H", "G-H", "I-J", "I-J", "K-L", "K-L",
                  "3ros", "3ros", "3ros", "3ros"]

    for idx, (slot_a, slot_b) in enumerate(R32_SLOTS):
        team_a = _resolve_bracket_slot(slot_a, likely_winners, likely_runners)
        team_b = _resolve_bracket_slot(slot_b, likely_winners, likely_runners)

        bg = LIGHT if idx % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 7, "F")
        _rgb(pdf, WHITE)

        try:
            if "3ro" not in team_a and "3ro" not in team_b:
                res, _ = predict_match(runtime, team_a, team_b)
                titular, score, _, conf, col_conf = _match_summary(res, team_a, team_b)
            else:
                res = None
                titular = "Por determinar"; score = "?-?"; conf = "-"; col_conf = LOW_GRAY
        except Exception:
            res = None
            titular = "Por determinar"; score = "?-?"; conf = "-"; col_conf = LOW_GRAY

        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 6)
        _c(pdf, tws_r32[0], 7, pod_labels[idx], ln=False, align="C")
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 6.5)
        _c(pdf, tws_r32[1], 7, _es(team_a)[:22], ln=False, align="R")
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 6)
        _c(pdf, tws_r32[2], 7, "vs", ln=False, align="C")
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B" if res else "", 6.5)
        _c(pdf, tws_r32[3], 7, _es(team_b)[:22], ln=False, align="L")
        _txt(pdf, PRIMARY)
        pdf.set_font("Helvetica", "B", 6)
        _c(pdf, tws_r32[4], 7, titular[:28], ln=False, align="C")
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 6)
        _c(pdf, tws_r32[5], 7, score, ln=False, align="C")
        _rgb(pdf, col_conf); _txt(pdf, WHITE)
        bx = pdf.get_x(); by = pdf.get_y() + 1
        pdf.rect(bx, by, tws_r32[6], 5, "F")
        pdf.set_font("Helvetica", "B", 5.5)
        _c(pdf, tws_r32[6], 7, conf, ln=True, align="C")
        _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    pdf.ln(3)
    _txt(pdf, LOW_GRAY)
    pdf.set_font("Helvetica", "I", 6.5)
    _c(pdf, 0, 5,
       "Nota: Los partidos de 'Mejor 3ro' no tienen prediccion hasta conocer los equipos "
       "clasificados. Los cruces respetan el bracket oficial WC2026 con pods (A-B), (C-D), etc.",
       ln=True)
    _txt(pdf, TEXT_COL)
    _page_footer(pdf, "Seccion 3 — Eliminatoria")


def _seccion_campeones(pdf, tourn_df, charts: dict | None = None):
    """Tabla completa de probabilidades por ronda + analisis coloquial."""
    _section_header(pdf, "Probabilidades por Ronda — Todos los Equipos",
                    f"Basado en {REPORT_SIMS:,} simulaciones del torneo completo con el bracket oficial",
                    SECONDARY)

    if tourn_df is None or tourn_df.empty:
        _c(pdf, 0, 6, "Simulacion no disponible.", ln=True)
        return

    charts = charts or {}
    _embed_chart(
        pdf,
        charts.get("champions"),
        "Top candidatos al titulo",
        "Vista compacta de las probabilidades de campeon. La tabla siguiente conserva el detalle por ronda.",
    )

    top = tourn_df.head(24)

    # Cabecera
    _rgb(pdf, DARK); _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "B", 6.5)
    ths = ["#", "Seleccion", "Campeon", "Final", "Semifinal", "Cuartos", "Octavos", "Sale grupos"]
    tws = [8, 40, 18, 18, 18, 18, 18, 22]
    for h, w in zip(ths, tws):
        _c(pdf, w, 5.5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
        pchamp = row.get("p_champion", 0)
        bg = LIGHT if rank % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 5, "F")
        if rank == 1:
            _rgb(pdf, GOLD); pdf.rect(10, pdf.get_y(), 3, 5, "F")
        _rgb(pdf, WHITE)

        pdf.set_font("Helvetica", "B" if rank <= 3 else "", 6.5)
        _txt(pdf, (140, 90, 0) if rank == 1 else (PRIMARY if rank <= 3 else TEXT_COL))
        _c(pdf, tws[0], 5, str(rank), ln=False, align="C")
        _c(pdf, tws[1], 5, f"  {_es(str(row.get('team','')))[:22]}", ln=False)
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "B" if pchamp > 0.08 else "", 6.5)
        _c(pdf, tws[2], 5, f"{pchamp:.1%}", ln=False, align="C")
        pdf.set_font("Helvetica", "", 6.5)
        for col, w in [("p_finalist", tws[3]), ("p_semifinal", tws[4]),
                       ("p_quarterfinal", tws[5]), ("p_round_16", tws[6]),
                       ("p_group_stage", tws[7])]:
            _c(pdf, w, 5, f"{row.get(col, 0):.1%}", ln=False, align="C")
        pdf.ln()

    _rgb(pdf, WHITE)
    pdf.ln(3)
    _txt(pdf, LOW_GRAY)
    pdf.set_font("Helvetica", "I", 6.5)
    _c(pdf, 0, 5,
       "Empates en eliminatoria resueltos por penales ponderados por diferencia de Elo. "
       "Terceros clasificados seleccionados por rendimiento (puntos > diferencia de goles > goles a favor).",
       ln=True)
    _txt(pdf, TEXT_COL)
    _page_footer(pdf, "Seccion 3 — Tabla de Campeones")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 5 — INFORME TECNICO
# ══════════════════════════════════════════════════════════════════════════════
def _informe_tecnico_a(pdf, meta: dict, charts: dict | None = None):
    _section_header(pdf, "Informe Tecnico — Arquitectura del Modelo",
                    "Para lectores con conocimiento en Machine Learning y Data Science", DARK)

    # Arquitectura del ensemble
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, "  Ensemble de 6 Modelos con Voting Ponderado", ln=True)

    desc_ensemble = (
        "El ensemble combina 6 modelos base mediante votacion ponderada. "
        "Los pesos de RF, XGB, LGBM y CatBoost se derivan de su accuracy en el conjunto de validacion: "
        "peso = max(0.05, accuracy_val - 0.33). "
        "Elo y Poisson tienen pesos fijos (0.10 y 0.07 respectivamente) por su naturaleza no-clasificatoria. "
        "Los pesos se normalizan para sumar 1.0."
    )
    _txt(pdf, TEXT_COL)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_x(12)
    pdf.multi_cell(PW - 24, 5, _t(desc_ensemble))
    pdf.ln(3)

    charts = charts or {}
    _embed_chart(
        pdf,
        charts.get("weights"),
        "Grafico de pesos del ensemble",
        "Los pesos reflejan cuanto aporta cada familia de modelo al voting final. Elo y Poisson funcionan como senales estructurales complementarias.",
    )

    # Tabla de modelos y pesos
    vw = meta.get("voting_weights", {})
    va = meta.get("val_accuracies", {})
    modelos = [
        ("RandomForest",  "rf",       "Bagging de arboles de decision. Robusto al overfitting."),
        ("XGBoost",       "xgb",      "Gradient boosting con regularizacion L1/L2. Hiperparametros tuneados con Optuna."),
        ("LightGBM",      "lgbm",     "GBDT con histogramas. Eficiente en datasets grandes."),
        ("CatBoost",      "catboost", "GBDT con manejo nativo de variables categoricas. Mejor modelo individual actual."),
        ("Elo (baseline)","elo",      "Formula Bradley-Terry clasica. Peso fijo como ancla estable."),
        ("Poisson",       "poisson",  "Distribucion bivariate de goles. W/D/L implicito. Peso fijo."),
    ]
    _rgb(pdf, DARK); _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 6.5)
    ths = ["Modelo", "Peso ensemble", "Acc. Val", "Rol"]
    tws_m = [40, 25, 22, 93]
    for h, w in zip(ths, tws_m):
        _c(pdf, w, 5.5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    for i, (nombre, key, rol) in enumerate(modelos):
        bg = LIGHT if i % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 6, "F")
        _rgb(pdf, WHITE)
        peso = vw.get(key, 0)
        vacc = va.get(key, va.get(key.replace("catboost","cb"), 0))
        pdf.set_font("Helvetica", "B", 6.5)
        _txt(pdf, DARK)
        _c(pdf, tws_m[0], 6, nombre, ln=False, align="L")
        _txt(pdf, PRIMARY)
        _c(pdf, tws_m[1], 6, f"{peso:.1%}", ln=False, align="C")
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 6.5)
        _c(pdf, tws_m[2], 6, f"{vacc:.1%}" if vacc else "-", ln=False, align="C")
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 6)
        _c(pdf, tws_m[3], 6, rol, ln=True)
    _rgb(pdf, WHITE)
    pdf.ln(4)

    # Pipeline de entrenamiento
    _rule(pdf, PRIMARY)
    pdf.ln(2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, "  Pipeline de Entrenamiento", ln=True)

    pasos = [
        ("1", "Recoleccion de datos",
         f"Se combinan {meta.get('n_source_matches',0):,} partidos internacionales desde el 2000. "
         "Fuentes: international-results, Kaggle match features, Kaggle Elo, FIFA rankings."),
        ("2", "Split cronologico",
         "70% train / 15% validation / 15% test en orden temporal estricto. "
         "No hay datos futuros visibles durante entrenamiento ni validacion."),
        ("3", "Features rolling sin leakage",
         "Para cada partido en fecha D, todas las features se calculan con datos anteriores a D. "
         "Solo en train se duplica con perspectiva inversa (A vs B y B vs A)."),
        ("4", "Sample weights por recencia",
         "Partidos recientes tienen mayor peso en entrenamiento. "
         "Decay exponencial con half-life de 3 anos."),
        ("5", "Entrenamiento paralelo",
         "RF, XGB, LGBM y CatBoost se entrenan en paralelo (ThreadPoolExecutor). "
         "Poisson se entrena con datos de train sobre goles reales."),
        ("6", "Calibracion por temperature scaling",
         "T = argmin log_loss sobre las probabilidades del ensemble en validation. "
         f"T actual = {meta.get('temperature_scaling', meta.get('temperature_scaling', 'N/A')):.4f} "
         "(cercano a 1.0: probs bien calibradas)."),
        ("7", "Evaluacion unica en test",
         "Las metricas finales se calculan una sola vez sobre el conjunto de test cronologico. "
         "Esto evita overfitting al conjunto de evaluacion."),
    ]
    for num, titulo, desc in pasos:
        y = pdf.get_y()
        _rgb(pdf, PRIMARY)
        pdf.rect(10, y, 8, 13, "F")
        _rgb(pdf, LIGHT)
        pdf.rect(18, y, PW - 28, 13, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_xy(11, y + 2.5)
        _c(pdf, 6, 6, num, ln=False, align="C")
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_xy(21, y + 1)
        _c(pdf, PW - 33, 5, titulo, ln=True)
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 6.5)
        pdf.set_x(21)
        pdf.multi_cell(PW - 33, 4, _t(desc))
        if pdf.get_y() < y + 13: pdf.set_y(y + 13)
        pdf.ln(2)

    _page_footer(pdf, "Seccion 4A — Informe Tecnico")


def _informe_tecnico_b(pdf, meta: dict):
    _section_header(pdf, "Informe Tecnico — Features y Calidad del Plantel",
                    "Variables de entrada al modelo y metodologia de squad quality", DARK)

    feature_names = list(meta.get("features") or MATCH_FEATURE_COLUMNS)
    n_features = _metadata_feature_count(meta)

    # Tabla de features
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, f"  Las {n_features} Features del Modelo", ln=True)

    feature_info = {
        "elo_diff": ("Diferencia de rating Elo entre equipo A y B.", "Elo"),
        "elo_win_prob_a": ("Probabilidad de victoria segun formula Bradley-Terry.", "Elo"),
        "xg_diff": ("Diferencia de goles esperados por partido (rolling, sin leakage).", "Ataque"),
        "xga_diff": ("Diferencia de goles en contra esperados por partido.", "Defensa"),
        "form_5_diff": ("Diferencia de forma reciente (puntos / max puntos, ultimos 5 partidos).", "Forma"),
        "offensive_power_diff": ("Diferencia de potencia ofensiva derivada de xG y conversion.", "Ataque"),
        "defensive_stability_diff": ("Diferencia de solidez defensiva (xGA, PPDA, clean sheets).", "Defensa"),
        "squad_depth_diff": ("Calidad del plantel desde ratings FIFA: avg_overall, max, depth_ratio.", "Plantel"),
        "consistency_diff": ("Diferencia de regularidad: std dev de resultados ultimos 5 partidos.", "Forma"),
        "wc_history_diff": ("Diferencia de historial mundialista (rendimiento en WCs anteriores).", "Historia"),
        "market_value_diff": ("Diferencia de valor de mercado relativo (millones EUR).", "Plantel"),
        "tactical_advantage": ("Ventaja tactica compuesta: xG vs xGA del rival, modulada por Elo.", "Tactica"),
        "h2h_advantage": ("Historial directo (rolling, min 3 partidos; 0 si insuficiente).", "H2H"),
        "is_tournament": ("1 si es partido competitivo, 0 si amistoso.", "Contexto"),
        "is_wc": ("1 si el partido es un Mundial.", "Contexto"),
        "is_qualifier": ("1 si es clasificatorio.", "Contexto"),
        "is_home": ("1 si el equipo A juega de local.", "Contexto"),
        "fifa_rank_diff": ("Diferencia de ranking FIFA disponible antes del partido.", "Ranking"),
        "elo_momentum_diff": ("Diferencia de momentum Elo: cambio en los ultimos 5 partidos.", "Momentum"),
        "days_since_last_match_diff": ("Diferencia de dias desde el ultimo partido jugado.", "Descanso"),
        "rest_days_diff": ("Inversa: diferencia de descanso relativo entre ambos equipos.", "Descanso"),
        "wc_recent_goal_balance_diff": ("Diferencia de balance goleador reciente en Mundiales.", "Historia"),
        "wc_recent_win_rate_diff": ("Diferencia de tasa de victorias reciente en Mundiales.", "Historia"),
        "wc_experience_diff": ("Diferencia de experiencia mundialista acumulada.", "Historia"),
        "wc_knockout_depth_diff": ("Diferencia de profundidad alcanzada en eliminatorias mundialistas.", "Historia"),
        "squad_market_value_diff": ("Diferencia de valor de plantel usado como senal secundaria.", "Plantel"),
        "fifa_points_pre_tournament_diff": ("Diferencia de puntos FIFA previos al torneo.", "Ranking"),
    }
    features = [
        (feat, *feature_info.get(feat, ("Feature diferencial de entrada al modelo.", "Modelo")))
        for feat in feature_names
    ]
    _rgb(pdf, DARK); _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 6)
    ths = ["Feature", "Descripcion", "Categoria"]
    tws_f = [48, 118, 24]
    for h, w in zip(ths, tws_f):
        _c(pdf, w, 5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    cat_colors = {
        "Elo": PRIMARY, "Ataque": GREEN, "Defensa": SECONDARY,
        "Forma": (160, 100, 0), "Plantel": (100, 60, 160),
        "Historia": DARK, "Tactica": ACCENT,
        "H2H": AMBER, "Contexto": LOW_GRAY, "Ranking": (80, 80, 80),
        "Momentum": SECONDARY, "Descanso": (120, 120, 120), "Modelo": TEXT_COL,
    }
    for i, (feat, desc, cat) in enumerate(features):
        bg = LIGHT if i % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 4.5, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 5.5)
        _c(pdf, tws_f[0], 4.5, feat, ln=False, align="L")
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 5.5)
        _c(pdf, tws_f[1], 4.5, desc, ln=False, align="L")
        col_cat = cat_colors.get(cat, TEXT_COL)
        _rgb(pdf, col_cat); _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 5)
        _c(pdf, tws_f[2], 4.5, cat, ln=True, align="C")
        _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    pdf.ln(4)

    # Squad Quality desde FIFA ratings
    _rule(pdf, PRIMARY)
    pdf.ln(2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, "  Squad Quality: Ratings FIFA en lugar de Valor de Mercado", ln=True)

    desc_sq = (
        "El feature 'squad_depth_diff' utiliza ratings FIFA reales de jugadores en lugar de "
        "valor de mercado como proxy. Fuente: player_aggregates.csv (FIFA 15-24, Kaggle). "
        "Se usa la version mas reciente por pais. El score integra:"
    )
    _txt(pdf, TEXT_COL)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_x(12)
    pdf.multi_cell(PW - 24, 5, _t(desc_sq))
    pdf.ln(2)

    sq_metrics = [
        ("avg_overall (35%)", "Calidad promedio del plantel. Rango tipico 65-87."),
        ("max_overall (20%)", "Rating del mejor jugador (estrella). Mbappé, Messi, etc."),
        ("depth_ratio (25%)", "avg/max: cuán robustez sin la estrella. <0.90 = dependiente de una figura."),
        ("avg_shooting (10%)", "Capacidad ofensiva promedio del plantel."),
        ("avg_defending (10%)", "Solidez defensiva promedio del plantel."),
    ]
    for metric, desc in sq_metrics:
        y = pdf.get_y()
        _rgb(pdf, LIGHT)
        pdf.rect(10, y, PW - 20, 7, "F")
        _rgb(pdf, ACCENT)
        pdf.rect(10, y, 2, 7, "F")
        _rgb(pdf, WHITE)
        pdf.set_xy(15, y + 1)
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 7)
        _c(pdf, 50, 4.5, metric, ln=False)
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 7)
        _c(pdf, PW - 70, 4.5, desc, ln=True)
        pdf.ln(1)

    _rgb(pdf, WHITE)
    _page_footer(pdf, "Seccion 4B — Informe Tecnico")


# ══════════════════════════════════════════════════════════════════════════════
# SECCION 6 — RESULTADOS DEL ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════════════
def _resultados_entrenamiento(pdf, meta: dict, charts: dict | None = None):
    _section_header(pdf, "Resultados del Modelo Entrenado",
                    "Metricas de evaluacion, calibracion y limitaciones conocidas", DARK)

    # Metricas principales
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, "  Metricas de Evaluacion (conjunto de test cronologico)", ln=True)

    charts = charts or {}
    _embed_chart(
        pdf,
        charts.get("calibration"),
        "Grafico de validacion: confianza vs accuracy",
        "La curva muestra como cambia el acierto historico cuando se exige mayor probabilidad maxima. Sirve para interpretar las etiquetas BAJO, MEDIO y ALTO.",
    )

    metricas = [
        ("accuracy_voting",            meta.get("accuracy_voting", 0),            True,  "Precision global del ensemble. Referencia aleatorio: 33.3%."),
        ("accuracy_catboost",          meta.get("accuracy_catboost", 0),           False, "Mejor modelo individual en esta corrida."),
        ("accuracy_lgbm",              meta.get("accuracy_lgbm", 0),              False, "LightGBM solo."),
        ("accuracy_rf",                meta.get("accuracy_rf", 0),                False, "Random Forest solo."),
        ("accuracy_xgb",               meta.get("accuracy_xgb", 0),               False, "XGBoost solo."),
        ("accuracy_high_confidence",   meta.get("accuracy_high_confidence", 0),    True,  f"Solo predicciones ALTO (n={meta.get('n_high_confidence_matches',0)}). Target: >80%."),
        ("accuracy_high_elo_diff_200", meta.get("accuracy_high_elo_diff_200", 0),  False, f"Partidos con delta Elo >=200 (n={meta.get('n_high_elo_matches',0)})."),
        ("log_loss_voting",            meta.get("log_loss_voting", 0),             False, "Calidad de probabilidades calibradas. Menor es mejor."),
        ("log_loss_uncalibrated",      meta.get("log_loss_uncalibrated", 0),       False, "Log-loss sin temperature scaling, para comparacion."),
    ]
    _rgb(pdf, DARK); _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 6.5)
    for h, w in zip(["Metrica", "Valor", "Descripcion"], [60, 20, 100]):
        _c(pdf, w, 5.5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, TEXT_COL); _rgb(pdf, WHITE)

    for i, (nombre, valor, destacado, desc) in enumerate(metricas):
        bg = LIGHT if i % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 6, "F")
        if destacado:
            _rgb(pdf, GOLD); pdf.rect(10, pdf.get_y(), 2.5, 6, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B" if destacado else "", 6.5)
        _c(pdf, 60, 6, f"  {nombre}", ln=False)
        pct = f"{valor:.1%}" if nombre.startswith("accuracy") else f"{valor:.4f}"
        if destacado: _txt(pdf, (140, 90, 0))
        pdf.set_font("Helvetica", "B", 6.5)
        _c(pdf, 20, 6, pct, ln=False, align="C")
        _txt(pdf, LOW_GRAY)
        pdf.set_font("Helvetica", "", 6)
        _c(pdf, 100, 6, desc, ln=True)
    _rgb(pdf, WHITE)
    pdf.ln(4)

    # Datos del entrenamiento
    _rule(pdf, PRIMARY)
    pdf.ln(2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, "  Datos de Entrenamiento", ln=True)

    datos = [
        ("Partidos fuente",         str(meta.get("n_source_matches", "?"))),
        ("Muestras de train",       str(meta.get("n_train_samples", "?"))),
        ("Muestras de validacion",  str(meta.get("n_val_samples", "?"))),
        ("Muestras de test",        str(meta.get("n_test_samples", "?"))),
        ("Arquitectura",            meta.get("ensemble_architecture", "?")),
        ("Hiperparametros tuneados",str(meta.get("tuned_hyperparameters", False))),
        ("Recency weights",         "Si (half-life 3 anos)"),
        ("Temperature scaling T",   f"{meta.get('temperature_scaling', 'N/A'):.4f}"),
        ("Fecha de entrenamiento",  str(meta.get("train_date", "?"))[:19]),
    ]
    for i, (label, valor) in enumerate(datos):
        bg = LIGHT if i % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 5, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, DARK)
        pdf.set_font("Helvetica", "B", 7)
        _c(pdf, 55, 5, f"  {label}", ln=False)
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 7)
        _c(pdf, PW - 65, 5, str(valor), ln=True)
    _rgb(pdf, WHITE)
    pdf.ln(4)

    # Limitaciones
    _rule(pdf, SECONDARY)
    pdf.ln(2)
    _txt(pdf, DARK)
    pdf.set_font("Helvetica", "B", 9)
    _c(pdf, 0, 7, "  Limitaciones y Caveats", ln=True)

    limitaciones = [
        "El target de >55% accuracy global no esta alcanzado (actual: 51.25%). "
        "Predecir futbol es intrisecamente dificil; el azar da 33.3%.",

        "Los xG rolling se ajustan por calidad Elo del rival, pero siguen dependiendo de datos "
        "historicos agregados y pueden sobrepremiar rachas contra calendarios faciles.",

        "No hay datos de lesiones, suspension de jugadores o convocatoria final WC2026. "
        "Eso puede alterar significativamente el rendimiento de un equipo en el torneo real.",

        "Las metricas se calculan sobre datos historicos hasta 2026. El Mundial real tiene "
        "condiciones unicas: presion, formato de eliminacion directa y factores psicologicos.",

        "El bracket de mejores terceros es una aproximacion. La tabla exacta de cruce de "
        "terceros segun su grupo de origen tiene 495 combinaciones posibles (FIFA).",
    ]
    for lim in limitaciones:
        y = pdf.get_y()
        _rgb(pdf, AMBER); pdf.rect(10, y, 2.5, 9, "F")
        _rgb(pdf, LIGHT); pdf.rect(12.5, y, PW - 22.5, 9, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, TEXT_COL)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_xy(16, y + 1)
        pdf.multi_cell(PW - 28, 4.5, _t(lim))
        if pdf.get_y() < y + 9: pdf.set_y(y + 9)
        pdf.ln(2)

    _txt(pdf, LOW_GRAY)
    pdf.set_font("Helvetica", "I", 6.5)
    pdf.ln(2)
    _c(pdf, 0, 5,
       "Codigo fuente: github.com/prode-ml  |  Generado con fpdf2  |  "
       "Modelos: RF + XGB + LGBM + CatBoost + Elo + Poisson",
       ln=True, align="C")
    _txt(pdf, TEXT_COL)
    _page_footer(pdf, "Seccion 5 — Resultados del Modelo")


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def build_report():
    print("Cargando modelos y runtime...")
    runtime   = load_prediction_runtime()
    poisson   = PoissonGoalModel().load()
    prediction_cache: dict[tuple[str, str], dict] = {}

    def cached_predict(a: str, b: str) -> dict:
        key = (a, b)
        if key not in prediction_cache:
            prediction_cache[key] = predict_match(runtime, a, b)[0]
        return prediction_cache[key]

    group_sim = GroupSimulator(
        poisson_model=poisson,
        predictor=cached_predict,
    )

    meta_path = MODELS_DIR / "model_metadata.json"
    meta: dict = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

    FPDF = _fpdf()
    pdf  = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Pre-calcular datos comunes
    print("Calculando simulaciones de grupos...")
    fixtures_idx = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    all_group_dfs: dict = {}
    all_partidos: dict = {}
    for gname, teams in GROUPS.items():
        partidos = []
        for i, j in fixtures_idx:
            ea, eb = teams[i], teams[j]
            try:
                res = cached_predict(ea, eb)
                partidos.append((ea, eb, res))
            except Exception:
                partidos.append((ea, eb, None))
        try:
            gdf = group_sim.simulate_group(gname, n_sims=REPORT_SIMS)
        except Exception:
            gdf = None
        all_group_dfs[gname] = gdf
        all_partidos[gname]   = partidos

    print("Simulando torneo completo...")
    tourn_df = None
    try:
        ts       = TournamentSimulator(poisson_model=poisson, group_simulator=group_sim, predictor=cached_predict)
        tourn_df = ts.simulate(n_sims=REPORT_SIMS)
    except Exception as e:
        print(f"  Advertencia: simulacion del torneo fallo: {e}")

    print("Generando graficos analiticos...")
    charts = _build_report_charts(meta, tourn_df, all_group_dfs, all_partidos)

    # ── Seccion 1: Portada ────────────────────────────────────────────────────
    print("Generando PDF...")
    pdf.add_page()
    _portada(pdf, meta)

    # ── Seccion 2: Resumen Ejecutivo ──────────────────────────────────────────
    pdf.add_page()
    _resumen_ejecutivo_a(pdf, meta)

    pdf.add_page()
    _resumen_ejecutivo_b(pdf, meta, tourn_df, all_group_dfs, all_partidos, charts)

    # ── Seccion 3: Grupos ─────────────────────────────────────────────────────
    for gname, teams in GROUPS.items():
        print(f"  Grupo {gname}...")
        pdf.add_page()
        _pagina_grupo(pdf, gname, teams, all_partidos[gname], all_group_dfs[gname])

    # ── Seccion 4: Eliminatorias ──────────────────────────────────────────────
    print("  Eliminatorias...")
    pdf.add_page()
    _seccion_eliminatorias(pdf, all_group_dfs, runtime, tourn_df)

    pdf.add_page()
    _seccion_campeones(pdf, tourn_df, charts)

    # ── Seccion 5: Informe Tecnico ────────────────────────────────────────────
    print("  Informe tecnico...")
    pdf.add_page()
    _informe_tecnico_a(pdf, meta, charts)

    pdf.add_page()
    _informe_tecnico_b(pdf, meta)

    # ── Seccion 6: Resultados del Modelo ─────────────────────────────────────
    pdf.add_page()
    _resultados_entrenamiento(pdf, meta, charts)

    # ── Guardar ───────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M")
    path = OUTPUT_DIR / f"prode_ML_WC2026_Report_{ts_str}.pdf"
    pdf.output(str(path))
    print(f"\nReporte guardado: {path}")
    return path


if __name__ == "__main__":
    path = build_report()
    print(f"\nAbrir: {path}")
    import os
    os.startfile(str(path))
