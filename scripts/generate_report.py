"""Generate PDF report with all predictions for FIFA World Cup 2026."""

import logging
import sys
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

ALTO = "🟢"
MEDIO = "🟡"
BAJO = "🔴"

def ensure_fpdf():
    try:
        from fpdf import FPDF
        return FPDF
    except ImportError:
        print("Installing fpdf2...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "--quiet"])
        from fpdf import FPDF
        return FPDF


def build_report():
    print("Loading models...")
    runtime = load_prediction_runtime()
    poisson = PoissonGoalModel().load()
    group_sim = GroupSimulator(poisson_model=poisson)

    FPDF = ensure_fpdf()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ── Title ──
    pdf.set_font("Helvetica", "B", 22)
    pdf.cell(0, 15, "prode-ML", ln=True, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "FIFA World Cup 2026 - Predictions Report", ln=True, align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
    pdf.ln(5)

    _load_metadata(pdf)

    # ── Group Predictions ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "1. Group Stage Predictions", ln=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "All 72 group matches with predicted probabilities and expected goals.", ln=True)
    pdf.ln(3)

    for group_name, teams in GROUPS.items():
        _group_header(pdf, group_name, teams)

        fixtures = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
        ]
        for i, j in fixtures:
            team_a, team_b = teams[i], teams[j]
            try:
                result, _ = predict_match(runtime, team_a, team_b)
                _match_row(pdf, team_a, team_b, result)
            except Exception as e:
                pdf.set_font("Helvetica", "", 8)
                pdf.cell(0, 5, f"  {team_a} vs {team_b} -- error: {e}", ln=True)

        pdf.ln(3)

    # ── Group Simulation ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "2. Group Stage Simulation (100,000 runs)", ln=True)
    pdf.ln(4)

    for group_name in GROUPS:
        _group_sim_section(pdf, group_name, group_sim)

    # ── Tournament Simulation ──
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, "3. Tournament Simulation (100,000 runs)", ln=True)
    pdf.ln(4)

    try:
        tourn_sim = TournamentSimulator(poisson_model=poisson, group_simulator=group_sim)
        tourn_df = tourn_sim.simulate(n_sims=100000)
        _tournament_section(pdf, tourn_df)
    except Exception as e:
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, f"Tournament simulation error: {e}", ln=True)

    # ── Save ──
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"prode_ML_WC2026_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(str(path))
    print(f"\nReport saved: {path}")
    return path


def _load_metadata(pdf):
    import json
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        pdf.set_font("Helvetica", "", 9)
        pdf.cell(0, 6, "No model metadata found. Train models first.", ln=True)
        return

    with open(meta_path) as f:
        meta = json.load(f)

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Model Metrics:", ln=True)
    pdf.set_font("Helvetica", "", 8)

    rows = [
        f"Training date: {meta.get('train_date', 'N/A')[:19]}",
        f"Samples: {meta.get('n_source_matches', 'N/A')} matches, {meta.get('n_train_samples', 'N/A')} train",
        f"Date range: {meta.get('date_min', 'N/A')} to {meta.get('date_max', 'N/A')}",
        f"Split: {meta.get('split_type', 'N/A')}, test from {meta.get('test_date_min', 'N/A')}",
    ]
    for r in rows:
        pdf.cell(0, 4, r, ln=True)
    pdf.ln(2)

    acc = [
        ("RandomForest", meta.get("accuracy_rf", 0)),
        ("XGBoost", meta.get("accuracy_xgb", 0)),
        ("LightGBM", meta.get("accuracy_lgbm", 0)),
        ("CatBoost", meta.get("accuracy_catboost", 0)),
        ("Meta-learner", meta.get("accuracy_meta", 0)),
        ("Two-Stage", meta.get("accuracy_two_stage", 0)),
        ("Confederation", meta.get("accuracy_confederation", 0)),
    ]
    pdf.cell(0, 5, "Model Accuracies (test set):", ln=True)
    for name, val in acc:
        if val > 0:
            pdf.cell(90, 4, f"  {name}", ln=False)
            pdf.cell(0, 4, f"{val:.1%}", ln=True)

    acc_elo = meta.get("accuracy_high_elo_diff_200", 0)
    n_elo = meta.get("n_high_elo_matches", 0)
    pdf.cell(0, 4, f"  High Elo diff (>=200): {acc_elo:.1%} (n={n_elo})", ln=True)
    pdf.cell(0, 4, f"  Baseline random: {meta.get('baseline_random', 0.333):.1%}", ln=True)


def _group_header(pdf, group_name, teams):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 7, f" GROUP {group_name}: {', '.join(teams)} ", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    pdf.set_font("Helvetica", "B", 7)
    pdf.cell(48, 5, "Match", ln=False, border=1)
    pdf.cell(18, 5, "Win A", ln=False, align="C", border=1)
    pdf.cell(18, 5, "Draw", ln=False, align="C", border=1)
    pdf.cell(18, 5, "Win B", ln=False, align="C", border=1)
    pdf.cell(24, 5, "xG", ln=False, align="C", border=1)
    pdf.cell(24, 5, "Most Likely", ln=False, align="C", border=1)
    pdf.cell(20, 5, "Conf", ln=False, align="C", border=1)
    pdf.cell(20, 5, "Upset", ln=True, align="C", border=1)


def _match_row(pdf, team_a, team_b, result):
    pdf.set_font("Helvetica", "", 7)

    match_text = f"{team_a[:22]} v {team_b[:22]}"
    pdf.cell(48, 5, match_text, ln=False, border=1)

    pdf.cell(18, 5, f"{result['p_win_a']:.1%}", ln=False, align="C", border=1)
    pdf.cell(18, 5, f"{result['p_draw']:.1%}", ln=False, align="C", border=1)
    pdf.cell(18, 5, f"{result['p_win_b']:.1%}", ln=False, align="C", border=1)

    xg_text = f"{result['xg_a']:.1f}-{result['xg_b']:.1f}"
    pdf.cell(24, 5, xg_text, ln=False, align="C", border=1)

    sl = result.get("top_scorelines", [])
    if sl:
        s = sl[0]
        score_text = f"{int(s['goals_a'])}-{int(s['goals_b'])} ({s['probability']:.1%})"
    else:
        score_text = "-"
    pdf.cell(24, 5, score_text, ln=False, align="C", border=1)

    conf = result.get("confidence", "?")
    conf_icon = ALTO if conf == "ALTO" else (MEDIO if conf == "MEDIO" else BAJO)
    pdf.cell(20, 5, f"{conf_icon} {conf}", ln=False, align="C", border=1)

    upset = result.get("upset_risk", 0)
    pdf.cell(20, 5, f"{upset:.1%}", ln=True, align="C", border=1)


def _group_sim_section(pdf, group_name, group_sim):
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(30, 60, 120)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 7, f" GROUP {group_name} ", ln=True, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)

    try:
        group_df = group_sim.simulate_group(group_name, n_sims=100000)
    except Exception:
        pdf.set_font("Helvetica", "", 8)
        pdf.cell(0, 5, "  Simulation not available", ln=True)
        return

    pdf.set_font("Helvetica", "B", 7)
    headers = ["Team", "1st", "2nd", "3rd", "4th", "Qualify", "Avg Pts", "Avg GD"]
    widths = [38, 14, 14, 14, 14, 20, 20, 20]
    for h, w in zip(headers, widths):
        pdf.cell(w, 5, h, ln=False, align="C", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for _, row in group_df.iterrows():
        vals = [
            str(row.get("team", ""))[:18],
            f"{row.get('prob_1st', 0):.1%}",
            f"{row.get('prob_2nd', 0):.1%}",
            f"{row.get('prob_3rd', 0):.1%}",
            f"{row.get('prob_4th', 0):.1%}",
            f"{row.get('qualify_direct_prob', 0):.1%}",
            f"{row.get('avg_pts', 0):.1f}",
            f"{row.get('avg_gd', 0):.2f}",
        ]
        for v, w in zip(vals, widths):
            pdf.cell(w, 5, v, ln=False, align="C", border=1)
        pdf.ln()

    try:
        fixtures = group_sim.simulate_group_fixtures(group_name, n_sims=100000)
        if fixtures:
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 7)
            pdf.cell(0, 5, "  Most Likely Results:", ln=True)
            pdf.set_font("Helvetica", "", 7)
            for fix in fixtures:
                sl = fix.get("most_likely_scoreline", {})
                pdf.cell(0, 4,
                    f"    {fix.get('team_a','')} vs {fix.get('team_b','')}  "
                    f"{int(sl.get('goals_a',0))}-{int(sl.get('goals_b',0))}  "
                    f"({sl.get('probability',0):.1%})",
                    ln=True)
    except Exception:
        pass

    pdf.ln(4)


def _tournament_section(pdf, tourn_df):
    tourn_df = tourn_df.sort_values("champion", ascending=False).head(20)

    pdf.set_font("Helvetica", "B", 7)
    headers = ["Team", "Champion", "Finalist", "Semi", "Quarter", "R16", "R32", "Group"]
    widths = [32, 17, 17, 17, 17, 17, 17, 17]
    for h, w in zip(headers, widths):
        pdf.cell(w, 5, h, ln=False, align="C", border=1)
    pdf.ln()

    pdf.set_font("Helvetica", "", 7)
    for _, row in tourn_df.iterrows():
        vals = [
            str(row.get("team", ""))[:18],
            f"{row.get('champion', 0):.1%}",
            f"{row.get('finalist', 0):.1%}",
            f"{row.get('sf', 0):.1%}",
            f"{row.get('qf', 0):.1%}",
            f"{row.get('r16', 0):.1%}",
            f"{row.get('r32', 0):.1%}",
            f"{row.get('group_stage', 0):.1%}",
        ]
        for v, w in zip(vals, widths):
            pdf.cell(w, 5, v, ln=False, align="C", border=1)
        pdf.ln()

    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "Note: Knockout rounds use Poisson-sampled results. Draws resolved by Elo-weighted penalty shootout.", ln=True)


if __name__ == "__main__":
    path = build_report()
    print(f"\nOpen: {path}")
    import os
    os.startfile(str(path))
