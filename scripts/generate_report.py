"""Generate PDF report with all predictions for FIFA World Cup 2026."""

import logging
import sys
import unicodedata
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)

from config.wc2026_groups import GROUPS
from src.runtime import load_prediction_runtime, predict_match
from src.simulation.group_simulator import GroupSimulator
from src.simulation.tournament_simulator import TournamentSimulator
from src.models.poisson_model import PoissonGoalModel

OUTPUT_DIR = Path(__file__).parent.parent / "reports"
MODELS_DIR = Path(__file__).parent.parent / "models"
REPORT_SIMULATIONS = 10_000

# Colores
BLUE_DARK = (20, 50, 100)
BLUE_MED  = (52, 100, 180)
GREEN     = (30, 130, 70)
AMBER     = (200, 130, 0)
GRAY      = (120, 120, 120)
GRAY_LIGHT= (230, 230, 230)
WHITE     = (255, 255, 255)
BLACK     = (0, 0, 0)


def _pdf_text(value) -> str:
    text = str(value)
    for src, dst in {
        "–": "-", "—": "-", "‘": "'", "’": "'",
        "“": '"', "”": '"', "•": "-", " ": " ",
        "→": "->", "▶": ">",
    }.items():
        text = text.replace(src, dst)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("latin-1", "ignore").decode("latin-1")


def _cell(pdf, w, h=0, txt="", **kwargs):
    return pdf.cell(w, h, _pdf_text(txt), **kwargs)


def _set_rgb(pdf, rgb):
    pdf.set_fill_color(*rgb)


def _set_text(pdf, rgb):
    pdf.set_text_color(*rgb)


def ensure_fpdf():
    try:
        from fpdf import FPDF
        return FPDF
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "--quiet"])
        from fpdf import FPDF
        return FPDF


# ─── Scoreline helper ────────────────────────────────────────────────────────

def _outcome(result):
    """Return 'win_a', 'draw', or 'win_b'."""
    pa = result.get("p_win_a", 0.38)
    pd_ = result.get("p_draw", 0.24)
    pb = result.get("p_win_b", 0.38)
    if pa >= pd_ and pa >= pb:
        return "win_a"
    if pb >= pd_ and pb >= pa:
        return "win_b"
    return "draw"


def _consistent_scoreline(result, outcome):
    """Most likely scoreline matching the predicted outcome."""
    for s in result.get("top_scorelines", []):
        ga = int(s.get("goals_a", 0) if isinstance(s, dict) else s[0])
        gb = int(s.get("goals_b", 0) if isinstance(s, dict) else s[1])
        if outcome == "win_a" and ga > gb:
            return ga, gb
        if outcome == "win_b" and gb > ga:
            return ga, gb
        if outcome == "draw" and ga == gb:
            return ga, gb
    # Fallback: derive from xG
    xa, xb = float(result.get("xg_a", 1.3)), float(result.get("xg_b", 1.1))
    if outcome == "win_a":
        ga = max(1, round(xa));  gb = max(0, min(ga - 1, round(xb)))
    elif outcome == "win_b":
        gb = max(1, round(xb));  ga = max(0, min(gb - 1, round(xa)))
    else:
        s = max(0, round((xa + xb) / 2));  ga = gb = s
    return ga, gb


def _outcome_label(result, team_a, team_b):
    """Returns (headline, scoreline_str, conf_label, conf_color)."""
    outcome = _outcome(result)
    ga, gb = _consistent_scoreline(result, outcome)
    conf = result.get("confidence", "BAJO")
    xga = result.get("xg_a", 0)
    xgb = result.get("xg_b", 0)

    conf_color = GREEN if conf == "ALTO" else (AMBER if conf == "MEDIO" else GRAY)

    short_a = team_a[:16]
    short_b = team_b[:16]

    if outcome == "win_a":
        headline = f"{short_a} WINS"
        score = f"{ga}-{gb}"
    elif outcome == "win_b":
        headline = f"{short_b} WINS"
        score = f"{gb}-{ga}"  # flip so bigger number first
    else:
        headline = "DRAW"
        score = f"{ga}-{gb}"

    xg_str = f"xG {xga:.1f}-{xgb:.1f}"
    return headline, score, xg_str, conf, conf_color


# ─── Cover page ──────────────────────────────────────────────────────────────

def _cover_page(pdf):
    import json
    pdf.set_auto_page_break(auto=True, margin=12)

    # Title band
    _set_rgb(pdf, BLUE_DARK)
    pdf.rect(0, 0, 210, 50, "F")
    _set_text(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_xy(0, 10)
    _cell(pdf, 210, 12, "prode-ML", ln=True, align="C")
    pdf.set_font("Helvetica", "", 13)
    _cell(pdf, 210, 8, "FIFA World Cup 2026 - Predictions Report", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 9)
    _cell(pdf, 210, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    _set_text(pdf, BLACK)
    pdf.ln(6)

    # Model accuracy box
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    pdf.set_font("Helvetica", "B", 13)
    _cell(pdf, 0, 8, "How accurate is this model?", ln=True)
    pdf.set_font("Helvetica", "", 9)
    _cell(pdf, 0, 5,
          "The model was trained on 8,273 international matches (2000-2026) and tested on games from 2017 onward.",
          ln=True)
    pdf.ln(3)

    # Metric boxes
    def _metric_box(x, y, label, value, sublabel, color):
        pdf.set_xy(x, y)
        _set_rgb(pdf, color)
        pdf.rect(x, y, 58, 22, "F")
        _set_text(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_xy(x, y + 2)
        _cell(pdf, 58, 10, value, ln=False, align="C")
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_xy(x, y + 12)
        _cell(pdf, 58, 4, label, ln=False, align="C")
        pdf.set_font("Helvetica", "", 6)
        pdf.set_xy(x, y + 16)
        _cell(pdf, 58, 4, sublabel, ln=False, align="C")
        _set_text(pdf, BLACK)

    acc = meta.get("accuracy_voting", 0)
    hi = meta.get("accuracy_high_confidence", 0)
    n_hi = meta.get("n_high_confidence_matches", 0)
    elo = meta.get("accuracy_high_elo_diff_200", 0)
    n_elo = meta.get("n_high_elo_matches", 0)

    y0 = pdf.get_y()
    _metric_box(12, y0, "Overall Accuracy", f"{acc:.0%}", "vs 33% random baseline", BLUE_MED)
    _metric_box(76, y0, "High-Confidence Accuracy", f"{hi:.0%}",
                f"on {n_hi} matches marked [HIGH]", GREEN)
    _metric_box(140, y0, "Large Elo-Diff Accuracy", f"{elo:.0%}",
                f"on {n_elo} one-sided matches", BLUE_DARK)
    pdf.set_xy(12, y0 + 26)
    pdf.ln(4)

    # Plain-language explanation
    pdf.set_font("Helvetica", "B", 10)
    _cell(pdf, 0, 7, "How to read this report", ln=True)
    pdf.set_font("Helvetica", "", 9)
    lines = [
        "For each group you will find three sections:",
        "  1. PREDICTED STANDINGS - which teams are most likely to qualify, based on 10,000 simulated group stages.",
        "  2. MATCH PREDICTIONS - each match shows the predicted result in plain language:",
        "       WINS / DRAW, the most likely exact scoreline, expected goals (xG), and a confidence label.",
        "  3. STATISTICAL DETAIL - full probability breakdown for those who want the numbers.",
        "",
        "Confidence labels:",
        "  [HIGH]  >70% probability on the predicted outcome. The model is most reliable here.",
        "  [MED]   60-70% probability. Good signal but some uncertainty.",
        "  [LOW]   <60% probability. The model sees the match as competitive or hard to call.",
        "",
        "Important: predictions are probabilities, not certainties. Football always surprises.",
    ]
    for line in lines:
        _cell(pdf, 0, 5, line, ln=True)

    # Model detail (small, at bottom)
    pdf.ln(4)
    _set_rgb(pdf, GRAY_LIGHT)
    pdf.rect(10, pdf.get_y(), 190, 28, "F")
    _set_text(pdf, BLACK)
    pdf.set_xy(12, pdf.get_y() + 2)
    pdf.set_font("Helvetica", "B", 8)
    _cell(pdf, 0, 5, "Technical summary", ln=True)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_x(12)
    _cell(pdf, 0, 4,
          f"Training: {meta.get('n_source_matches','?')} matches | "
          f"Models: RF + XGB + LGBM + CatBoost + Elo (weighted voting) | "
          f"Calibration: temperature scaling T={meta.get('confidence_thresholds',{}).get('high_prob','?')}",
          ln=True)
    pdf.set_x(12)
    _cell(pdf, 0, 4,
          f"Tuned: {'Yes' if meta.get('tuned_hyperparameters') else 'No'} | "
          f"Recency weights: {'Yes' if meta.get('recency_weighted_training') else 'No'} | "
          f"Log-loss: {meta.get('log_loss_voting', 0):.4f} | "
          f"Test period: {meta.get('test_date_min','?')[:7]} to {meta.get('date_max','?')[:7]}",
          ln=True)


# ─── Group page ──────────────────────────────────────────────────────────────

def _group_page(pdf, group_name, teams, match_results, group_df):
    """Full group section: standings + plain predictions + detail table."""

    # ── Header band ──────────────────────────────────────────────────────────
    _set_rgb(pdf, BLUE_DARK)
    pdf.rect(0, pdf.get_y() - 1, 210, 10, "F")
    _set_text(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 13)
    _cell(pdf, 0, 9,
          f" GROUP {group_name}   {' | '.join(teams)}",
          ln=True, fill=False)
    _set_text(pdf, BLACK)
    pdf.ln(2)

    # ── Predicted standings ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 9)
    _cell(pdf, 0, 5, "PREDICTED STANDINGS  (10,000 simulations)", ln=True)
    pdf.ln(1)

    if group_df is not None and not group_df.empty:
        # header row
        _set_rgb(pdf, BLUE_MED)
        _set_text(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 7)
        widths_s = [8, 42, 22, 22, 22, 22, 22, 26]
        headers_s = ["#", "Team", "1st", "2nd", "3rd", "4th", "Qualify", "Avg Pts"]
        for h, w in zip(headers_s, widths_s):
            _cell(pdf, w, 5, h, ln=False, align="C", border=0)
        pdf.ln()
        _set_text(pdf, BLACK)

        for rank, (_, row) in enumerate(group_df.iterrows(), 1):
            # Alternate row shading
            if rank % 2 == 0:
                _set_rgb(pdf, GRAY_LIGHT)
                pdf.rect(10, pdf.get_y(), 176, 5.5, "F")
            pdf.set_font("Helvetica", "B" if rank <= 2 else "", 7)
            vals = [
                str(rank),
                str(row.get("team", ""))[:20],
                f"{row.get('prob_1st', 0):.0%}",
                f"{row.get('prob_2nd', 0):.0%}",
                f"{row.get('prob_3rd', 0):.0%}",
                f"{row.get('prob_4th', 0):.0%}",
                f"{row.get('qualify_direct_prob', 0):.0%}",
                f"{row.get('avg_pts', 0):.1f}",
            ]
            for v, w in zip(vals, widths_s):
                align = "C" if v != vals[1] else "L"
                _cell(pdf, w, 5.5, v if v != vals[1] else f"  {v}", ln=False, align=align)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 8)
        _cell(pdf, 0, 5, "  Simulation not available.", ln=True)

    pdf.ln(4)

    # ── Plain language match predictions ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 9)
    _cell(pdf, 0, 5, "MATCH PREDICTIONS", ln=True)
    pdf.ln(1)

    for team_a, team_b, result in match_results:
        if result is None:
            pdf.set_font("Helvetica", "I", 7)
            _cell(pdf, 0, 6, f"  {team_a} vs {team_b}  --  prediction unavailable", ln=True)
            continue

        headline, score, xg_str, conf, conf_color = _outcome_label(result, team_a, team_b)

        # Match name (left)
        pdf.set_font("Helvetica", "", 8)
        _cell(pdf, 50, 7, f"  {team_a[:18]} vs {team_b[:18]}", ln=False)

        # Outcome headline (bold, dark blue)
        _set_text(pdf, BLUE_DARK)
        pdf.set_font("Helvetica", "B", 9)
        _cell(pdf, 42, 7, headline, ln=False)

        # Score (large, dark)
        _set_text(pdf, BLACK)
        pdf.set_font("Helvetica", "B", 11)
        _cell(pdf, 16, 7, score, ln=False, align="C")

        # xG (small, gray)
        _set_text(pdf, GRAY)
        pdf.set_font("Helvetica", "", 7)
        _cell(pdf, 28, 7, xg_str, ln=False, align="C")
        _set_text(pdf, BLACK)

        # Confidence badge
        _set_rgb(pdf, conf_color)
        pdf.rect(pdf.get_x() + 1, pdf.get_y() + 1, 18, 5, "F")
        _set_text(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 6)
        _cell(pdf, 20, 7, conf, ln=True, align="C")
        _set_text(pdf, BLACK)
        _set_rgb(pdf, WHITE)

    pdf.ln(4)

    # ── Detail table ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 8)
    _cell(pdf, 0, 5, "STATISTICAL DETAIL", ln=True)
    pdf.ln(1)

    _set_rgb(pdf, BLUE_MED)
    _set_text(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 6)
    col_w = [50, 15, 15, 15, 22, 26, 17, 16]
    col_h = ["Match", "Win A", "Draw", "Win B", "xG A-B", "Best Score", "Conf", "Upset"]
    for h, w in zip(col_h, col_w):
        _cell(pdf, w, 5, h, ln=False, align="C", border=0)
    pdf.ln()
    _set_text(pdf, BLACK)

    for idx, (team_a, team_b, result) in enumerate(match_results):
        if result is None:
            continue
        if idx % 2 == 0:
            _set_rgb(pdf, GRAY_LIGHT)
            pdf.rect(10, pdf.get_y(), 176, 5, "F")
        pdf.set_font("Helvetica", "", 6)
        sl = result.get("top_scorelines", [])
        if sl:
            s = sl[0]
            ga = int(s.get("goals_a", 0) if isinstance(s, dict) else s[0])
            gb = int(s.get("goals_b", 0) if isinstance(s, dict) else s[1])
            pr = float(s.get("probability", 0) if isinstance(s, dict) else s[2])
            score_txt = f"{ga}-{gb} ({pr:.0%})"
        else:
            score_txt = "-"
        conf = result.get("confidence", "?")
        vals = [
            f"  {team_a[:20]} v {team_b[:18]}",
            f"{result.get('p_win_a', 0):.0%}",
            f"{result.get('p_draw', 0):.0%}",
            f"{result.get('p_win_b', 0):.0%}",
            f"{result.get('xg_a', 0):.1f} - {result.get('xg_b', 0):.1f}",
            score_txt,
            conf,
            f"{result.get('upset_risk', 0):.0%}",
        ]
        aligns = ["L", "C", "C", "C", "C", "C", "C", "C"]
        for v, w, al in zip(vals, col_w, aligns):
            _cell(pdf, w, 5, v, ln=False, align=al)
        pdf.ln()

    pdf.ln(6)


# ─── Tournament section ───────────────────────────────────────────────────────

def _tournament_page(pdf, tourn_df):
    _set_rgb(pdf, BLUE_DARK)
    pdf.rect(0, pdf.get_y() - 1, 210, 10, "F")
    _set_text(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 13)
    _cell(pdf, 0, 9,
          f" TOURNAMENT SIMULATION  ({REPORT_SIMULATIONS:,} simulations)",
          ln=True, fill=False)
    _set_text(pdf, BLACK)
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 8)
    _cell(pdf, 0, 5,
          "Each column shows the probability of each team reaching that stage of the tournament.",
          ln=True)
    pdf.ln(3)

    tourn_df = tourn_df.sort_values("p_champion", ascending=False).head(24)

    _set_rgb(pdf, BLUE_MED)
    _set_text(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 7)
    headers = ["Team", "Champion", "Finalist", "Semi", "Quarter", "Round 16", "Round 32", "Exit Group"]
    widths  = [34,      18,         18,         16,     17,         17,          17,          17]
    for h, w in zip(headers, widths):
        _cell(pdf, w, 5, h, ln=False, align="C")
    pdf.ln()
    _set_text(pdf, BLACK)

    for idx, (_, row) in enumerate(tourn_df.iterrows()):
        if idx % 2 == 0:
            _set_rgb(pdf, GRAY_LIGHT)
            pdf.rect(10, pdf.get_y(), 174, 5, "F")
        champ = row.get("p_champion", 0)
        pdf.set_font("Helvetica", "B" if champ > 0.05 else "", 7)
        vals = [
            f"  {str(row.get('team',''))[:18]}",
            f"{champ:.1%}",
            f"{row.get('p_finalist', 0):.1%}",
            f"{row.get('p_semifinal', 0):.1%}",
            f"{row.get('p_quarterfinal', 0):.1%}",
            f"{row.get('p_round_16', 0):.1%}",
            f"{row.get('p_round_32', 0):.1%}",
            f"{row.get('p_group_stage', 0):.1%}",
        ]
        aligns = ["L"] + ["C"] * 7
        for v, w, al in zip(vals, widths, aligns):
            _cell(pdf, w, 5, v, ln=False, align=al)
        pdf.ln()

    pdf.ln(4)
    _set_text(pdf, GRAY)
    pdf.set_font("Helvetica", "I", 7)
    _cell(pdf, 0, 5,
          "Knockout rounds use Poisson-sampled scorelines. Draws resolved by Elo-weighted penalty shootout.",
          ln=True)
    _set_text(pdf, BLACK)


# ─── Main ────────────────────────────────────────────────────────────────────

def build_report():
    print("Loading models...")
    runtime = load_prediction_runtime()
    poisson = PoissonGoalModel().load()
    group_sim = GroupSimulator(
        poisson_model=poisson,
        predictor=lambda team_a, team_b: predict_match(runtime, team_a, team_b)[0],
    )

    FPDF = ensure_fpdf()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Cover page
    pdf.add_page()
    _cover_page(pdf)

    # One page per group (predictions + standings + detail)
    fixtures_order = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for group_name, teams in GROUPS.items():
        print(f"  Group {group_name}...")
        pdf.add_page()

        # Gather match predictions
        match_results = []
        for i, j in fixtures_order:
            team_a, team_b = teams[i], teams[j]
            try:
                result, _ = predict_match(runtime, team_a, team_b)
                match_results.append((team_a, team_b, result))
            except Exception as e:
                match_results.append((team_a, team_b, None))

        # Run group simulation
        try:
            group_df = group_sim.simulate_group(group_name, n_sims=REPORT_SIMULATIONS)
        except Exception:
            group_df = None

        _group_page(pdf, group_name, teams, match_results, group_df)

    # Tournament simulation page
    print("  Tournament simulation...")
    pdf.add_page()
    try:
        tourn_sim = TournamentSimulator(poisson_model=poisson, group_simulator=group_sim)
        tourn_df = tourn_sim.simulate(n_sims=REPORT_SIMULATIONS)
        _tournament_page(pdf, tourn_df)
    except Exception as e:
        pdf.set_font("Helvetica", "", 9)
        _cell(pdf, 0, 6, f"Tournament simulation error: {e}", ln=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"prode_ML_WC2026_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(str(path))
    print(f"\nReport saved: {path}")
    return path


if __name__ == "__main__":
    path = build_report()
    print(f"\nOpen: {path}")
    import os
    os.startfile(str(path))
