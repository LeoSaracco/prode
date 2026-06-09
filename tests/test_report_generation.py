import pandas as pd

from scripts.generate_report import _match_row, _metadata_feature_count, _pdf_text, _tournament_section


class FakePDF:
    def __init__(self):
        self.text = []

    def set_font(self, *args, **kwargs):
        pass

    def set_fill_color(self, *args, **kwargs):
        pass

    def set_text_color(self, *args, **kwargs):
        pass

    def ln(self, *args, **kwargs):
        pass

    def cell(self, w, h=0, txt="", *args, **kwargs):
        txt.encode("latin-1")
        self.text.append(txt)


def _result(top_scorelines):
    return {
        "p_win_a": 0.55,
        "p_draw": 0.25,
        "p_win_b": 0.20,
        "xg_a": 1.6,
        "xg_b": 0.9,
        "top_scorelines": top_scorelines,
        "confidence": "ALTO",
        "upset_risk": 0.12,
    }


def test_report_scoreline_tuple_and_dict():
    _match_row(FakePDF(), "A", "B", _result([(1, 0, 0.18)]))
    _match_row(FakePDF(), "A", "B", _result([{"goals_a": 1, "goals_b": 0, "probability": 0.18}]))


def test_report_winner_summary_does_not_use_nil_nil():
    pdf = FakePDF()
    _match_row(FakePDF(), "A", "B", _result([(0, 0, 0.20), (2, 0, 0.14), (1, 0, 0.13)]))
    _match_row(pdf, "A", "B", _result([(0, 0, 0.20), (2, 0, 0.14), (1, 0, 0.13)]))

    assert "A GANA | 0-0" not in pdf.text[0]
    assert "A GANA | 2-0" in pdf.text[0]


def test_report_feature_count_uses_metadata_value():
    assert _metadata_feature_count({"n_features": 27}) == 27


def test_pdf_text_is_helvetica_compatible():
    assert _pdf_text("Côte d'Ivoire - México").encode("latin-1")
    _match_row(FakePDF(), "Côte d'Ivoire", "México", _result([(1, 0, 0.18)]))


def test_tournament_section_reads_real_probability_columns():
    pdf = FakePDF()
    df = pd.DataFrame([
        {
            "team": "Argentina",
            "p_group_stage": 0.95,
            "p_round_32": 0.85,
            "p_round_16": 0.70,
            "p_quarterfinal": 0.55,
            "p_semifinal": 0.40,
            "p_finalist": 0.25,
            "p_champion": 0.15,
        }
    ])

    _tournament_section(pdf, df)

    assert "15.0%" in pdf.text
    assert "25.0%" in pdf.text
