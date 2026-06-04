"""Canonical team names and source-specific aliases."""

from config.wc2026_groups import GROUPS


def _source_alias(name: str) -> dict[str, str | None]:
    return {
        "fbref": name,
        "understat": None,
        "clubelo": None,
        "sofifa": name,
        "eloratings": name,
        "kaggle": name,
        "international": name,
    }


TEAM_ALIASES: dict[str, dict[str, str | None]] = {
    team: _source_alias(team)
    for teams in GROUPS.values()
    for team in teams
}

TEAM_ALIASES["South Korea"].update({
    "fbref": "Korea Republic",
    "sofifa": "Korea Republic",
    "eloratings": "Korea Republic",
    "kaggle": "Korea Republic",
    "international": "South Korea",
})
TEAM_ALIASES["Cote d'Ivoire"].update({
    "fbref": "Ivory Coast",
    "sofifa": "Ivory Coast",
    "eloratings": "Ivory Coast",
    "kaggle": "Ivory Coast",
    "international": "Ivory Coast",
})
TEAM_ALIASES["DR Congo"].update({
    "fbref": "DR Congo",
    "sofifa": "DR Congo",
    "eloratings": "DR Congo",
    "kaggle": "DR Congo",
    "international": "DR Congo",
})
TEAM_ALIASES["Czechia"].update({
    "fbref": "Czech Republic",
    "sofifa": "Czech Republic",
    "eloratings": "Czech Republic",
    "kaggle": "Czech Republic",
    "international": "Czech Republic",
})
TEAM_ALIASES["Turkiye"].update({
    "fbref": "Turkey",
    "sofifa": "Turkey",
    "eloratings": "Turkey",
    "kaggle": "Turkey",
    "international": "Turkey",
})
TEAM_ALIASES["Curacao"].update({
    "fbref": "Curacao",
    "sofifa": "Curacao",
    "eloratings": "Curacao",
    "kaggle": None,
    "international": "Curacao",
})
TEAM_ALIASES["Bosnia and Herzegovina"].update({
    "fbref": "Bosnia and Herzegovina",
    "sofifa": "Bosnia and Herzegovina",
    "eloratings": "Bosnia and Herzegovina",
    "kaggle": "Bosnia and Herzegovina",
    "international": "Bosnia and Herzegovina",
})

CANONICAL_NAMES: dict[str, str] = {
    # Spanish/common variants
    "alemania": "Germany",
    "argelia": "Algeria",
    "argentina": "Argentina",
    "arabia saudita": "Saudi Arabia",
    "australia": "Australia",
    "austria": "Austria",
    "belgica": "Belgium",
    "belgium": "Belgium",
    "bosnia": "Bosnia and Herzegovina",
    "bosnia y herzegovina": "Bosnia and Herzegovina",
    "brasil": "Brazil",
    "cabo verde": "Cape Verde",
    "canada": "Canada",
    "chequia": "Czechia",
    "colombia": "Colombia",
    "corea": "South Korea",
    "corea del sur": "South Korea",
    "costa de marfil": "Cote d'Ivoire",
    "croacia": "Croatia",
    "curazao": "Curacao",
    "ecuador": "Ecuador",
    "egipto": "Egypt",
    "escocia": "Scotland",
    "espana": "Spain",
    "españa": "Spain",
    "estados unidos": "United States",
    "francia": "France",
    "ghana": "Ghana",
    "haiti": "Haiti",
    "holanda": "Netherlands",
    "inglaterra": "England",
    "irak": "Iraq",
    "iran": "Iran",
    "japon": "Japan",
    "jordania": "Jordan",
    "marruecos": "Morocco",
    "mexico": "Mexico",
    "nueva zelanda": "New Zealand",
    "noruega": "Norway",
    "panama": "Panama",
    "paraguay": "Paraguay",
    "paises bajos": "Netherlands",
    "portugal": "Portugal",
    "qatar": "Qatar",
    "rd congo": "DR Congo",
    "republica democratica del congo": "DR Congo",
    "senegal": "Senegal",
    "sudafrica": "South Africa",
    "suecia": "Sweden",
    "suiza": "Switzerland",
    "tunez": "Tunisia",
    "turquia": "Turkiye",
    "uruguay": "Uruguay",
    "usa": "United States",
    "uzbekistan": "Uzbekistan",
}

for canonical_name in TEAM_ALIASES:
    CANONICAL_NAMES.setdefault(canonical_name.lower(), canonical_name)

CANONICAL_NAMES.update({
    "ivory coast": "Cote d'Ivoire",
    "czech republic": "Czechia",
    "korea republic": "South Korea",
    "south korea": "South Korea",
    "turkey": "Turkiye",
    "netherlands": "Netherlands",
    "dr congo": "DR Congo",
})


def resolve_team_name(input_name: str) -> str | None:
    """Resolve a user/source name to the canonical WC2026 team name."""
    normalized = input_name.strip().lower()
    for canonical in TEAM_ALIASES:
        if canonical.lower() == normalized:
            return canonical
    return CANONICAL_NAMES.get(normalized)
