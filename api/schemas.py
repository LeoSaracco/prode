"""Pydantic schemas for the prode-ML API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    team_a: str = Field(min_length=1)
    team_b: str = Field(min_length=1)


class Scoreline(BaseModel):
    goals_a: int
    goals_b: int
    probability: float


class FeatureImpact(BaseModel):
    name: str
    value: float
    direction: str


class MatchResult(BaseModel):
    team_a: str
    team_b: str
    probabilities: dict[str, float]
    expected_goals: dict[str, float]
    confidence: str
    predicted_outcome: str
    outcome_scoreline: Scoreline | None
    exact_most_likely_scoreline: Scoreline | None
    most_likely_scoreline: Scoreline | None
    top_scorelines: list[Scoreline]
    upset_risk: float
    top_features: list[FeatureImpact]
    elo: dict[str, float]
    model_breakdown: dict[str, list[float]]


class TeamInfo(BaseModel):
    name: str
    group: str | None
    elo: float
    confederation: str | None


class TeamsResponse(BaseModel):
    teams: list[TeamInfo]


class GroupsResponse(BaseModel):
    groups: dict[str, list[str]]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    teams: int


class GroupSimulationRow(BaseModel):
    team: str
    group: str
    prob_1st: float
    prob_2nd: float
    prob_3rd: float
    prob_4th: float
    qualify_direct_prob: float
    avg_pts: float
    avg_gd: float


class GroupFixturePrediction(BaseModel):
    team_a: str
    team_b: str
    expected_goals: dict[str, float]
    most_likely_scoreline: Scoreline | None


class GroupSimulationResponse(BaseModel):
    group: str
    n_sims: int
    results: list[GroupSimulationRow]
    fixtures: list[GroupFixturePrediction]


class TournamentRow(BaseModel):
    team: str
    p_group_stage: float
    p_round_32: float
    p_round_16: float
    p_quarterfinal: float
    p_semifinal: float
    p_finalist: float
    p_champion: float
    rank: int
    title_tier: str | None = None


class TournamentSimulationResponse(BaseModel):
    n_sims: int
    results: list[TournamentRow]


# ── Fixtures / Calendar ──────────────────────────────────────────────────────


class FixturePrediction(BaseModel):
    date: str
    matchday: int
    group: str
    team_a: str
    team_b: str
    venue: str | None
    predicted_outcome: str
    outcome_scoreline: Scoreline | None
    win_a: float
    draw: float
    win_b: float
    confidence: str
    xg_a: float
    xg_b: float


class FixturesResponse(BaseModel):
    fixtures: list[FixturePrediction]


# ── Bracket / Tournament tree ────────────────────────────────────────────────


class BracketTeam(BaseModel):
    team: str
    prob: float


class BracketSlot(BaseModel):
    slot_index: int
    teams: list[BracketTeam]
    label: str | None = None


class BracketRound(BaseModel):
    round_name: str
    display: str
    slots: list[BracketSlot]
    feeds_from: list[list[int]]


class BracketResponse(BaseModel):
    n_sims: int
    rounds: list[BracketRound]
