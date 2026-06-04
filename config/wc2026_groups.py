"""Official FIFA World Cup 2026 groups.

The draw was held on 2025-12-05. Play-off placeholders were resolved in
March 2026, leaving 48 unique teams across 12 groups of four.
"""

GROUPS: dict[str, list[str]] = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Scotland", "Haiti"],
    "D": ["United States", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Ecuador", "Cote d'Ivoire", "Curacao"],
    "F": ["Netherlands", "Sweden", "Japan", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Uruguay", "Saudi Arabia", "Cape Verde"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Austria", "Algeria", "Jordan"],
    "K": ["Portugal", "Colombia", "Uzbekistan", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

HOST_NATIONS = ["United States", "Mexico", "Canada"]

CONFEDERATION = {
    # UEFA
    "Austria": "UEFA",
    "Belgium": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "Croatia": "UEFA",
    "Czechia": "UEFA",
    "England": "UEFA",
    "France": "UEFA",
    "Germany": "UEFA",
    "Netherlands": "UEFA",
    "Norway": "UEFA",
    "Portugal": "UEFA",
    "Scotland": "UEFA",
    "Spain": "UEFA",
    "Sweden": "UEFA",
    "Switzerland": "UEFA",
    "Turkiye": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL",
    "Brazil": "CONMEBOL",
    "Colombia": "CONMEBOL",
    "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL",
    "Uruguay": "CONMEBOL",
    # CONCACAF
    "Canada": "CONCACAF",
    "Curacao": "CONCACAF",
    "Haiti": "CONCACAF",
    "Mexico": "CONCACAF",
    "Panama": "CONCACAF",
    "United States": "CONCACAF",
    # CAF
    "Algeria": "CAF",
    "Cape Verde": "CAF",
    "Cote d'Ivoire": "CAF",
    "DR Congo": "CAF",
    "Egypt": "CAF",
    "Ghana": "CAF",
    "Morocco": "CAF",
    "Senegal": "CAF",
    "South Africa": "CAF",
    "Tunisia": "CAF",
    # AFC
    "Australia": "AFC",
    "Iran": "AFC",
    "Iraq": "AFC",
    "Japan": "AFC",
    "Jordan": "AFC",
    "Qatar": "AFC",
    "Saudi Arabia": "AFC",
    "South Korea": "AFC",
    "Uzbekistan": "AFC",
    # OFC
    "New Zealand": "OFC",
}

WC_HISTORY_SCORE: dict[str, float] = {
    "Argentina": 1.00,
    "Brazil": 1.00,
    "Germany": 1.00,
    "France": 0.95,
    "Spain": 0.90,
    "Uruguay": 0.80,
    "England": 0.75,
    "Netherlands": 0.75,
    "Croatia": 0.70,
    "Morocco": 0.65,
    "Portugal": 0.65,
    "Belgium": 0.60,
    "South Korea": 0.60,
    "Ghana": 0.55,
    "Mexico": 0.55,
    "Poland": 0.55,
    "Sweden": 0.55,
    "Colombia": 0.50,
    "Costa Rica": 0.50,
    "Paraguay": 0.50,
    "Senegal": 0.50,
    "Cameroon": 0.45,
    "Cote d'Ivoire": 0.30,
    "Czechia": 0.45,
    "Japan": 0.45,
    "Scotland": 0.45,
    "Algeria": 0.45,
    "Australia": 0.40,
    "Norway": 0.40,
    "Saudi Arabia": 0.40,
    "Switzerland": 0.40,
    "Ecuador": 0.35,
    "Egypt": 0.35,
    "South Africa": 0.35,
    "DR Congo": 0.35,
    "Haiti": 0.30,
    "Iran": 0.30,
    "Qatar": 0.30,
    "Tunisia": 0.30,
    "Bosnia and Herzegovina": 0.25,
    "Canada": 0.25,
    "Panama": 0.25,
    "Cape Verde": 0.15,
    "Curacao": 0.15,
    "Iraq": 0.15,
    "Jordan": 0.15,
    "New Zealand": 0.15,
    "Turkiye": 0.45,
    "Uzbekistan": 0.10,
    "Austria": 0.40,
}


def get_group_for_team(team: str) -> str | None:
    for group, teams in GROUPS.items():
        if team in teams:
            return group
    return None


def get_group_opponents(team: str) -> list[str]:
    group = get_group_for_team(team)
    if group is None:
        return []
    return [t for t in GROUPS[group] if t != team]
