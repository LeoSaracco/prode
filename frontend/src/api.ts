export const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000/api/v1";

export type TeamInfo = {
  name: string;
  group: string | null;
  elo: number;
  confederation: string | null;
};

export type Scoreline = {
  goals_a: number;
  goals_b: number;
  probability: number;
};

export type MatchResult = {
  team_a: string;
  team_b: string;
  probabilities: {
    win_a: number;
    draw: number;
    win_b: number;
  };
  expected_goals: {
    team_a: number;
    team_b: number;
  };
  confidence: string;
  most_likely_scoreline: Scoreline | null;
  top_scorelines: Scoreline[];
  upset_risk: number;
  top_features: Array<{ name: string; value: number; direction: string }>;
  elo: { team_a: number; team_b: number; diff: number };
};

export type GroupSimulationRow = {
  team: string;
  group: string;
  prob_1st: number;
  prob_2nd: number;
  prob_3rd: number;
  prob_4th: number;
  qualify_direct_prob: number;
  avg_pts: number;
  avg_gd: number;
};

export type GroupFixturePrediction = {
  team_a: string;
  team_b: string;
  expected_goals: {
    team_a: number;
    team_b: number;
  };
  most_likely_scoreline: Scoreline | null;
};

export type TournamentRow = {
  team: string;
  p_group_stage: number;
  p_round_32: number;
  p_round_16: number;
  p_quarterfinal: number;
  p_semifinal: number;
  p_finalist: number;
  p_champion: number;
  rank: number;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json() as Promise<T>;
}

export function fetchTeams() {
  return request<{ teams: TeamInfo[] }>("/teams");
}

export function fetchGroups() {
  return request<{ groups: Record<string, string[]> }>("/groups");
}

export function predictMatch(teamA: string, teamB: string) {
  return request<MatchResult>("/predict", {
    method: "POST",
    body: JSON.stringify({ team_a: teamA, team_b: teamB })
  });
}

export function simulateGroup(group: string, nSims = 5000) {
  return request<{
    group: string;
    n_sims: number;
    results: GroupSimulationRow[];
    fixtures: GroupFixturePrediction[];
  }>(
    `/simulate/group/${group}?n_sims=${nSims}`
  );
}

export function simulateTournament(nSims = 2500, topN = 20) {
  return request<{ n_sims: number; results: TournamentRow[] }>(
    `/simulate/tournament?n_sims=${nSims}&top_n=${topN}`
  );
}
