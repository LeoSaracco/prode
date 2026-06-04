"""
Genera el reporte PDF oficial estilo FIFA World Cup 2026.
Paleta: navy oscuro (#050A30), azul FIFA (#009DE0), dorado (#F0B800), purpura WC2026.
"""

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
MODELS_DIR  = Path(__file__).parent.parent / "models"
REPORT_SIMS = 10_000

# ── Paleta FIFA WC2026 ────────────────────────────────────────────────────────
NAVY      = (5,   10,  48)    # fondo oscuro principal
BLUE      = (0,  157, 224)    # azul FIFA
GOLD      = (240, 184,   0)   # dorado FIFA
PURPLE    = (108,  33, 168)   # purpura WC2026
WHITE     = (255, 255, 255)
OFF_WHITE = (247, 247, 252)
GRAY      = (155, 155, 170)
L_GRAY    = (232, 232, 242)
GREEN     = ( 16, 150,  72)   # ALTO confianza
AMBER     = (210, 120,   0)   # MEDIO confianza
LOW_GRAY  = (130, 130, 148)   # BAJO confianza
BLACK     = (  0,   0,   0)

PW = 210   # ancho pagina A4 mm


# ── Helpers de texto ─────────────────────────────────────────────────────────
def _t(value: str) -> str:
    """Convierte a texto compatible con Helvetica/latin-1."""
    for src, dst in {
        "á":"a","é":"e","í":"i","ó":"o","ú":"u",
        "Á":"A","É":"E","Í":"I","Ó":"O","Ú":"U",
        "ñ":"n","Ñ":"N","ü":"u","Ü":"U",
        "–":"-","—":"-","'":"'","'":"'",
        """:'"',""":'"',"•":"-"," ":" ",
        "→":"->","▶":">","°":"",
    }.items():
        value = value.replace(src, dst)
    value = unicodedata.normalize("NFKD", str(value))
    return value.encode("latin-1", "ignore").decode("latin-1")


def _c(pdf, w, h=0, txt="", **kw):
    """cell() con conversion de texto."""
    return pdf.cell(w, h, _t(txt), **kw)


def _rgb(pdf, rgb):
    pdf.set_fill_color(*rgb)

def _txt(pdf, rgb):
    pdf.set_text_color(*rgb)

def _draw(pdf, rgb):
    pdf.set_draw_color(*rgb)


# ── Carga fpdf ────────────────────────────────────────────────────────────────
def _fpdf():
    try:
        from fpdf import FPDF
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])
        from fpdf import FPDF
    return FPDF


# ── Helpers de resultado ──────────────────────────────────────────────────────
def _outcome(r):
    pa, pd_, pb = r.get("p_win_a", .38), r.get("p_draw", .24), r.get("p_win_b", .38)
    if pa >= pd_ and pa >= pb: return "win_a"
    if pb >= pd_ and pb >= pa: return "win_b"
    return "draw"

def _scoreline(r, outcome):
    for s in r.get("top_scorelines", []):
        ga = int(s.get("goals_a", 0) if isinstance(s, dict) else s[0])
        gb = int(s.get("goals_b", 0) if isinstance(s, dict) else s[1])
        if outcome == "win_a" and ga > gb: return ga, gb
        if outcome == "win_b" and gb > ga: return ga, gb
        if outcome == "draw" and ga == gb: return ga, gb
    xa, xb = float(r.get("xg_a", 1.3)), float(r.get("xg_b", 1.1))
    if outcome == "win_a":
        ga = max(1, round(xa)); gb = max(0, min(ga - 1, round(xb)))
    elif outcome == "win_b":
        gb = max(1, round(xb)); ga = max(0, min(gb - 1, round(xa)))
    else:
        s = max(0, round((xa + xb) / 2)); ga = gb = s
    return ga, gb

def _match_summary(r, team_a, team_b):
    """Devuelve (titular, marcador, xg_txt, conf, color_conf)."""
    out = _outcome(r)
    ga, gb = _scoreline(r, out)
    conf = r.get("confidence", "BAJO")
    xa, xb = r.get("xg_a", 0), r.get("xg_b", 0)
    color_conf = GREEN if conf == "ALTO" else (AMBER if conf == "MEDIO" else LOW_GRAY)

    a16, b16 = team_a[:16], team_b[:16]
    if   out == "win_a": titular = f"{a16.upper()} GANA"; score = f"{ga}-{gb}"
    elif out == "win_b": titular = f"{b16.upper()} GANA"; score = f"{gb}-{ga}"
    else:                titular = "EMPATE";              score = f"{ga}-{gb}"

    return titular, score, f"xG {xa:.1f}-{xb:.1f}", conf, color_conf


# ── Regla decorativa FIFA ─────────────────────────────────────────────────────
def _rule(pdf, y_offset=0):
    """Barra dorada delgada separadora al estilo FIFA."""
    y = pdf.get_y() + y_offset
    _rgb(pdf, GOLD)
    pdf.rect(10, y, PW - 20, 0.8, "F")
    _rgb(pdf, WHITE)
    pdf.set_y(y + 1.5)


# ── Portada ───────────────────────────────────────────────────────────────────
def _portada(pdf):
    import json

    # Banda superior navy + detalle dorado
    _rgb(pdf, NAVY)
    pdf.rect(0, 0, PW, 60, "F")
    _rgb(pdf, GOLD)
    pdf.rect(0, 60, PW, 2.5, "F")
    _rgb(pdf, PURPLE)
    pdf.rect(0, 62.5, PW, 1.5, "F")

    # Texto del encabezado
    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(0, 10)
    _c(pdf, PW, 6, "FIFA WORLD CUP 2026™", ln=True, align="C")

    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 28)
    _c(pdf, PW, 14, "prode-ML", ln=True, align="C")

    pdf.set_font("Helvetica", "", 12)
    _c(pdf, PW, 7, "Reporte oficial de predicciones", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    _c(pdf, PW, 6, "Canada  |  Mexico  |  Estados Unidos", ln=True, align="C")

    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_xy(0, 52)
    _c(pdf, PW, 5, f"Generado: {datetime.now().strftime('%d/%m/%Y  %H:%M')}", ln=True, align="C")

    _txt(pdf, BLACK)
    pdf.set_y(72)

    # ── Cajas de metricas ─────────────────────────────────────────────────────
    meta_path = MODELS_DIR / "model_metadata.json"
    if not meta_path.exists():
        return
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    acc  = meta.get("accuracy_voting", 0)
    hi   = meta.get("accuracy_high_confidence", 0)
    n_hi = meta.get("n_high_confidence_matches", 0)
    elo  = meta.get("accuracy_high_elo_diff_200", 0)
    n_elo= meta.get("n_high_elo_matches", 0)

    def _caja(x, y, color, valor, etiqueta, sub):
        _rgb(pdf, color)
        pdf.rect(x, y, 56, 26, "F")
        _rgb(pdf, WHITE); _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_xy(x, y + 2)
        _c(pdf, 56, 11, valor, ln=False, align="C")
        pdf.set_font("Helvetica", "B", 6.5)
        pdf.set_xy(x, y + 14)
        _c(pdf, 56, 4, etiqueta.upper(), ln=False, align="C")
        pdf.set_font("Helvetica", "", 5.5)
        pdf.set_xy(x, y + 18)
        _c(pdf, 56, 4, sub, ln=False, align="C")
        _txt(pdf, BLACK); _rgb(pdf, WHITE)

    y0 = pdf.get_y()
    _caja(12,  y0, BLUE,   f"{acc:.0%}",  "Precision global",      "vs 33% al azar")
    _caja(77,  y0, GREEN,  f"{hi:.0%}",   "Predicciones ALTO",     f"sobre {n_hi} partidos marcados")
    _caja(142, y0, NAVY,   f"{elo:.0%}",  "Diferencia Elo > 200",  f"sobre {n_elo} partidos claros")
    pdf.set_y(y0 + 30)
    pdf.ln(5)

    # ── Como leer este reporte ────────────────────────────────────────────────
    _rule(pdf)
    pdf.ln(3)
    _txt(pdf, NAVY)
    pdf.set_font("Helvetica", "B", 12)
    _c(pdf, 0, 7, "  Como leer este reporte", ln=True)
    _txt(pdf, BLACK)
    pdf.set_font("Helvetica", "", 9)

    lineas = [
        "Para cada grupo encontraras tres secciones:",
        "",
        "  1. CLASIFICACION ESTIMADA",
        "     Que equipos tienen mas probabilidad de pasar de fase, segun 10.000 simulaciones.",
        "     Los primeros 2 de cada grupo avanzan directamente; el mejor 3ro tambien clasifica.",
        "",
        "  2. PREDICCION DE PARTIDOS",
        "     Cada partido muestra el resultado esperado en lenguaje claro:",
        "     GANA / EMPATE  |  Marcador mas probable  |  xG esperado  |  Nivel de confianza",
        "",
        "  3. DETALLE ESTADISTICO",
        "     Tabla con las probabilidades completas para quien quiera profundizar.",
        "",
        "Niveles de confianza:",
        "     [ALTO]   Prob. maxima > 70%.  El modelo es mas confiable en estos partidos.",
        "     [MEDIO]  60-70%.  Hay senal clara pero con algo de incertidumbre.",
        "     [BAJO]   < 60%.  El modelo ve el partido como parejo o dificil de predecir.",
        "",
        "Importante: las predicciones son probabilidades, no certezas. El futbol siempre sorprende.",
    ]
    for l in lineas:
        _c(pdf, 0, 5, l, ln=True)

    # ── Resumen tecnico ───────────────────────────────────────────────────────
    pdf.ln(3)
    _rule(pdf)
    pdf.ln(2)
    _rgb(pdf, OFF_WHITE)
    y_box = pdf.get_y()
    pdf.rect(10, y_box, PW - 20, 18, "F")
    _rgb(pdf, WHITE)
    pdf.set_xy(13, y_box + 2)
    _txt(pdf, NAVY)
    pdf.set_font("Helvetica", "B", 7.5)
    _c(pdf, 0, 5, "Resumen tecnico del modelo", ln=True)
    pdf.set_x(13)
    pdf.set_font("Helvetica", "", 6.5)
    _txt(pdf, BLACK)
    n_src = meta.get("n_source_matches", "?")
    n_trn = meta.get("n_train_samples", "?")
    tuned = "Si" if meta.get("tuned_hyperparameters") else "No"
    rec   = "Si" if meta.get("recency_weighted_training") else "No"
    ll    = meta.get("log_loss_voting", 0)
    test  = meta.get("test_date_min", "?")[:7]
    _c(pdf, 0, 4,
       f"Partidos fuente: {n_src}  |  Muestras entrenamiento: {n_trn}  |"
       f"  Modelos: RF + XGB + LGBM + CatBoost + Elo (voting ponderado)",
       ln=True)
    pdf.set_x(13)
    _c(pdf, 0, 4,
       f"Hiperparametros tuneados: {tuned}  |  Pesos por recencia: {rec}  |"
       f"  Log-loss: {ll:.4f}  |  Periodo de test: desde {test}",
       ln=True)
    pdf.set_x(13)
    _c(pdf, 0, 4,
       "Calibracion: temperature scaling  |  Historial directo H2H activo  |"
       "  Consistencia de forma activa  |  Features rolling sin leakage",
       ln=True)


# ── Pagina de grupo ───────────────────────────────────────────────────────────
def _pagina_grupo(pdf, nombre, equipos, partidos, group_df):
    """Una pagina completa por grupo: clasificacion + predicciones + detalle."""

    # ── Banner del grupo ──────────────────────────────────────────────────────
    _rgb(pdf, NAVY)
    pdf.rect(0, pdf.get_y() - 1, PW, 14, "F")
    # Acento dorado en el lado izquierdo
    _rgb(pdf, GOLD)
    pdf.rect(0, pdf.get_y() - 1, 3, 14, "F")
    # Letra del grupo grande
    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_xy(5, pdf.get_y())
    _c(pdf, 18, 13, nombre, ln=False)
    # Nombre del grupo
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_xy(22, pdf.get_y() + 1)
    _c(pdf, PW - 30, 6, "GRUPO " + nombre, ln=True)
    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_x(22)
    _c(pdf, PW - 30, 5, "  |  ".join(e.upper() for e in equipos), ln=True)
    _txt(pdf, BLACK); _rgb(pdf, WHITE)
    pdf.ln(2)

    # ── 1. Clasificacion estimada ─────────────────────────────────────────────
    _rgb(pdf, BLUE)
    pdf.rect(10, pdf.get_y(), PW - 20, 6, "F")
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(11, pdf.get_y())
    _c(pdf, PW - 22, 6, "CLASIFICACION ESTIMADA  (10.000 simulaciones)",
       ln=True, align="L")
    _txt(pdf, BLACK); _rgb(pdf, WHITE)
    pdf.ln(0.5)

    if group_df is not None and not group_df.empty:
        # Cabecera de tabla
        _rgb(pdf, NAVY); _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 6)
        ws = [8, 46, 18, 18, 18, 18, 22, 22]
        hs = ["#", "Equipo", "1ro", "2do", "3ro", "4to", "Clasifica", "Pts prom"]
        for h, w in zip(hs, ws):
            _c(pdf, w, 5, h, ln=False, align="C")
        pdf.ln(); _txt(pdf, BLACK); _rgb(pdf, WHITE)

        for pos, (_, row) in enumerate(group_df.iterrows(), 1):
            bg = L_GRAY if pos % 2 == 0 else OFF_WHITE
            _rgb(pdf, bg)
            pdf.rect(10, pdf.get_y(), PW - 20, 5.5, "F")
            _rgb(pdf, WHITE)
            bold = pos <= 2
            pdf.set_font("Helvetica", "B" if bold else "", 6.5)
            # Numero de posicion
            if pos == 1:   _txt(pdf, GOLD)
            elif pos == 2: _txt(pdf, BLUE)
            else:          _txt(pdf, GRAY)
            _c(pdf, ws[0], 5.5, str(pos), ln=False, align="C")
            _txt(pdf, BLACK)
            _c(pdf, ws[1], 5.5, "  " + str(row.get("team",""))[:22], ln=False, align="L")
            vals_s = [
                f"{row.get('prob_1st', 0):.0%}",
                f"{row.get('prob_2nd', 0):.0%}",
                f"{row.get('prob_3rd', 0):.0%}",
                f"{row.get('prob_4th', 0):.0%}",
                f"{row.get('qualify_direct_prob', 0):.0%}",
                f"{row.get('avg_pts', 0):.1f}",
            ]
            for v, w in zip(vals_s, ws[2:]):
                _c(pdf, w, 5.5, v, ln=False, align="C")
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "I", 7)
        _c(pdf, 0, 5, "  Simulacion no disponible.", ln=True)

    pdf.ln(3)

    # ── 2. Prediccion de partidos ─────────────────────────────────────────────
    _rgb(pdf, PURPLE)
    pdf.rect(10, pdf.get_y(), PW - 20, 6, "F")
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(11, pdf.get_y())
    _c(pdf, PW - 22, 6, "PREDICCION DE PARTIDOS", ln=True, align="L")
    _txt(pdf, BLACK); _rgb(pdf, WHITE)
    pdf.ln(0.5)

    for idx, (ea, eb, res) in enumerate(partidos):
        if res is None:
            pdf.set_font("Helvetica", "I", 7)
            _c(pdf, 0, 6, f"  {ea} vs {eb}  --  sin datos", ln=True)
            continue

        titular, score, xg_txt, conf, col_conf = _match_summary(res, ea, eb)

        # Fila con fondo alternado
        row_bg = L_GRAY if idx % 2 == 0 else OFF_WHITE
        _rgb(pdf, row_bg)
        pdf.rect(10, pdf.get_y(), PW - 20, 8, "F")
        _rgb(pdf, WHITE)

        # Partido (equipos)
        _txt(pdf, GRAY)
        pdf.set_font("Helvetica", "", 7)
        _c(pdf, 52, 8, f"  {ea[:18]} vs {eb[:18]}", ln=False)

        # Titular del resultado (GANA / EMPATE)
        _txt(pdf, NAVY)
        pdf.set_font("Helvetica", "B", 9)
        _c(pdf, 50, 8, titular, ln=False)

        # Marcador
        _txt(pdf, BLACK)
        pdf.set_font("Helvetica", "B", 13)
        _c(pdf, 18, 8, score, ln=False, align="C")

        # xG
        _txt(pdf, GRAY)
        pdf.set_font("Helvetica", "", 6.5)
        _c(pdf, 26, 8, xg_txt, ln=False, align="C")
        _txt(pdf, BLACK)

        # Badge de confianza
        bx = pdf.get_x() + 1
        by = pdf.get_y() + 1.5
        _rgb(pdf, col_conf)
        pdf.rect(bx, by, 22, 5, "F")
        _rgb(pdf, WHITE)
        _txt(pdf, WHITE)
        pdf.set_font("Helvetica", "B", 6)
        _c(pdf, 24, 8, conf, ln=True, align="C")
        _txt(pdf, BLACK); _rgb(pdf, WHITE)

    pdf.ln(3)

    # ── 3. Detalle estadistico ────────────────────────────────────────────────
    _rgb(pdf, NAVY); _txt(pdf, GOLD)
    pdf.rect(10, pdf.get_y(), PW - 20, 5.5, "F")
    pdf.set_font("Helvetica", "B", 6.5)
    pdf.set_xy(11, pdf.get_y())
    _c(pdf, PW - 22, 5.5, "DETALLE ESTADISTICO", ln=True, align="L")
    _txt(pdf, BLACK); _rgb(pdf, WHITE)
    pdf.ln(0.5)

    # Cabecera del detalle
    _rgb(pdf, L_GRAY); _txt(pdf, NAVY)
    pdf.set_font("Helvetica", "B", 5.5)
    col_dw = [52, 14, 14, 14, 22, 26, 14, 14]
    col_dh = ["Partido", "G. A", "Emp", "G. B", "xG A-B", "Marcador prob.", "Conf", "Sorpresa"]
    for h, w in zip(col_dh, col_dw):
        _c(pdf, w, 5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, BLACK); _rgb(pdf, WHITE)

    for idx, (ea, eb, res) in enumerate(partidos):
        if res is None:
            continue
        bg = L_GRAY if idx % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg); pdf.rect(10, pdf.get_y(), PW - 20, 4.5, "F"); _rgb(pdf, WHITE)

        sl = res.get("top_scorelines", [])
        if sl:
            s = sl[0]
            ga = int(s.get("goals_a", 0) if isinstance(s, dict) else s[0])
            gb = int(s.get("goals_b", 0) if isinstance(s, dict) else s[1])
            pr = float(s.get("probability", 0) if isinstance(s, dict) else s[2])
            score_t = f"{ga}-{gb} ({pr:.0%})"
        else:
            score_t = "-"

        pdf.set_font("Helvetica", "", 5.5)
        row_vals = [
            f"  {ea[:18]} v {eb[:16]}",
            f"{res.get('p_win_a', 0):.0%}",
            f"{res.get('p_draw', 0):.0%}",
            f"{res.get('p_win_b', 0):.0%}",
            f"{res.get('xg_a', 0):.1f} - {res.get('xg_b', 0):.1f}",
            score_t,
            res.get("confidence", "?"),
            f"{res.get('upset_risk', 0):.0%}",
        ]
        row_al = ["L", "C", "C", "C", "C", "C", "C", "C"]
        for v, w, al in zip(row_vals, col_dw, row_al):
            _c(pdf, w, 4.5, v, ln=False, align=al)
        pdf.ln()

    pdf.ln(5)


# ── Pagina de torneo ──────────────────────────────────────────────────────────
def _pagina_torneo(pdf, tourn_df):
    # Banner
    _rgb(pdf, NAVY)
    pdf.rect(0, pdf.get_y() - 1, PW, 14, "F")
    _rgb(pdf, GOLD)
    pdf.rect(0, pdf.get_y() - 1, 3, 14, "F")
    _txt(pdf, WHITE)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_xy(8, pdf.get_y() + 1)
    _c(pdf, PW - 16, 6, f"SIMULACION DEL TORNEO  —  {REPORT_SIMS:,} simulaciones", ln=True)
    _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "", 7)
    pdf.set_x(8)
    _c(pdf, PW - 16, 5,
       "Probabilidad de cada seleccion de llegar a cada ronda del torneo.", ln=True)
    _txt(pdf, BLACK); _rgb(pdf, WHITE)
    pdf.ln(3)

    tourn_df = tourn_df.sort_values("p_champion", ascending=False).head(24)

    # Cabecera
    _rgb(pdf, NAVY); _txt(pdf, GOLD)
    pdf.set_font("Helvetica", "B", 6.5)
    ths = ["Seleccion", "Campeon", "Final", "Semifinal", "Cuartos", "Octavos", "32avos", "Sale grupos"]
    tws = [38, 17, 17, 17, 17, 17, 17, 17]
    for h, w in zip(ths, tws):
        _c(pdf, w, 5.5, h, ln=False, align="C")
    pdf.ln(); _txt(pdf, BLACK); _rgb(pdf, WHITE)

    for idx, (_, row) in enumerate(tourn_df.iterrows()):
        champ = row.get("p_champion", 0)
        bg = L_GRAY if idx % 2 == 0 else OFF_WHITE
        _rgb(pdf, bg); pdf.rect(10, pdf.get_y(), PW - 20, 5, "F"); _rgb(pdf, WHITE)
        bold = champ > 0.07
        pdf.set_font("Helvetica", "B" if bold else "", 6.5)
        if champ > 0.12:   _txt(pdf, NAVY)
        elif champ > 0.05: _txt(pdf, PURPLE)
        else:              _txt(pdf, BLACK)
        _c(pdf, tws[0], 5, f"  {str(row.get('team',''))[:20]}", ln=False, align="L")
        _txt(pdf, BLACK)
        tv = [
            f"{champ:.1%}",
            f"{row.get('p_finalist', 0):.1%}",
            f"{row.get('p_semifinal', 0):.1%}",
            f"{row.get('p_quarterfinal', 0):.1%}",
            f"{row.get('p_round_16', 0):.1%}",
            f"{row.get('p_round_32', 0):.1%}",
            f"{row.get('p_group_stage', 0):.1%}",
        ]
        for v, w in zip(tv, tws[1:]):
            _c(pdf, w, 5, v, ln=False, align="C")
        pdf.ln()

    pdf.ln(4)
    _txt(pdf, GRAY)
    pdf.set_font("Helvetica", "I", 6.5)
    _c(pdf, 0, 5,
       "Nota: rondas de eliminacion usan marcadores muestreados por Poisson. "
       "Empates resueltos por penales ponderados por Elo.", ln=True)
    _txt(pdf, BLACK)


# ── Main ──────────────────────────────────────────────────────────────────────
def build_report():
    print("Cargando modelos...")
    runtime   = load_prediction_runtime()
    poisson   = PoissonGoalModel().load()
    group_sim = GroupSimulator(
        poisson_model=poisson,
        predictor=lambda a, b: predict_match(runtime, a, b)[0],
    )

    FPDF = _fpdf()
    pdf  = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)

    # Portada
    pdf.add_page()
    _portada(pdf)

    # Grupos
    fixtures_idx = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    for gname, teams in GROUPS.items():
        print(f"  Grupo {gname}...")
        pdf.add_page()

        partidos = []
        for i, j in fixtures_idx:
            ea, eb = teams[i], teams[j]
            try:
                res, _ = predict_match(runtime, ea, eb)
                partidos.append((ea, eb, res))
            except Exception:
                partidos.append((ea, eb, None))

        try:
            group_df = group_sim.simulate_group(gname, n_sims=REPORT_SIMS)
        except Exception:
            group_df = None

        _pagina_grupo(pdf, gname, teams, partidos, group_df)

    # Torneo
    print("  Simulacion del torneo...")
    pdf.add_page()
    try:
        ts      = TournamentSimulator(poisson_model=poisson, group_simulator=group_sim)
        tdf     = ts.simulate(n_sims=REPORT_SIMS)
        _pagina_torneo(pdf, tdf)
    except Exception as e:
        pdf.set_font("Helvetica", "", 9)
        _c(pdf, 0, 6, f"Error en simulacion: {e}", ln=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"prode_ML_WC2026_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    pdf.output(str(path))
    print(f"\nReporte guardado: {path}")
    return path


if __name__ == "__main__":
    path = build_report()
    print(f"\nAbrir: {path}")
    import os
    os.startfile(str(path))
