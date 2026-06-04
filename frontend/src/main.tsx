import React from "react";
import ReactDOM from "react-dom/client";
import { Activity, BarChart3, Play, Trophy } from "lucide-react";
import {
  fetchGroups,
  fetchTeams,
  GroupFixturePrediction,
  GroupSimulationRow,
  MatchResult,
  predictMatch,
  simulateGroup,
  simulateTournament,
  TeamInfo,
  TournamentRow
} from "./api";
import "./styles.css";

type View = "predict" | "groups" | "tournament";

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [view, setView] = React.useState<View>("predict");
  const [teams, setTeams] = React.useState<TeamInfo[]>([]);
  const [groups, setGroups] = React.useState<Record<string, string[]>>({});
  const [teamA, setTeamA] = React.useState("Argentina");
  const [teamB, setTeamB] = React.useState("Jordan");
  const [match, setMatch] = React.useState<MatchResult | null>(null);
  const [groupName, setGroupName] = React.useState("J");
  const [groupRows, setGroupRows] = React.useState<GroupSimulationRow[]>([]);
  const [groupFixtures, setGroupFixtures] = React.useState<GroupFixturePrediction[]>([]);
  const [tournamentRows, setTournamentRows] = React.useState<TournamentRow[]>([]);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    Promise.all([fetchTeams(), fetchGroups()])
      .then(([teamResponse, groupResponse]) => {
        setTeams(teamResponse.teams);
        setGroups(groupResponse.groups);
      })
      .catch((err) => setError(err.message));
  }, []);

  React.useEffect(() => {
    if (teams.length) void runPrediction();
  }, [teams.length]);

  async function runPrediction() {
    setLoading(true);
    setError(null);
    try {
      setMatch(await predictMatch(teamA, teamB));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo calcular la prediccion");
    } finally {
      setLoading(false);
    }
  }

  async function runGroup() {
    setLoading(true);
    setError(null);
    try {
      const response = await simulateGroup(groupName, 5000);
      setGroupRows(response.results);
      setGroupFixtures(response.fixtures);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo simular el grupo");
    } finally {
      setLoading(false);
    }
  }

  async function runTournament() {
    setLoading(true);
    setError(null);
    try {
      const response = await simulateTournament(2500, 20);
      setTournamentRows(response.results);
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo simular el torneo");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>prode-ML</h1>
          <p>Predicciones del Mundial 2026</p>
        </div>
        <nav className="tabs" aria-label="Vistas">
          <button className={view === "predict" ? "active" : ""} onClick={() => setView("predict")}>
            <Activity size={18} /> Prediccion
          </button>
          <button className={view === "groups" ? "active" : ""} onClick={() => setView("groups")}>
            <BarChart3 size={18} /> Grupos
          </button>
          <button className={view === "tournament" ? "active" : ""} onClick={() => setView("tournament")}>
            <Trophy size={18} /> Torneo
          </button>
        </nav>
      </header>

      {error && <div className="error">{error}</div>}

      {view === "predict" && (
        <section className="workspace">
          <div className="control-strip">
            <TeamSelect label="Equipo 1" value={teamA} teams={teams} onChange={setTeamA} />
            <span className="versus">VS</span>
            <TeamSelect label="Equipo 2" value={teamB} teams={teams} onChange={setTeamB} />
            <button className="primary" onClick={runPrediction} disabled={loading || teamA === teamB}>
              <Play size={18} /> Predecir partido
            </button>
          </div>
          {match && <PredictionPanel match={match} loading={loading} />}
        </section>
      )}

      {view === "groups" && (
        <section className="workspace">
          <div className="control-strip compact">
            <label>
              Grupo
              <select value={groupName} onChange={(event) => setGroupName(event.target.value)}>
                {Object.keys(groups).map((group) => (
                  <option key={group} value={group}>{group}</option>
                ))}
              </select>
            </label>
            <button className="primary" onClick={runGroup} disabled={loading}>
              <Play size={18} /> Simular grupo
            </button>
          </div>
          <TeamList title={`Grupo ${groupName}`} teams={groups[groupName] ?? []} />
          <DataTable rows={groupRows} />
          <FixtureResults group={groupName} fixtures={groupFixtures} />
        </section>
      )}

      {view === "tournament" && (
        <section className="workspace">
          <div className="control-strip compact">
            <button className="primary" onClick={runTournament} disabled={loading}>
              <Trophy size={18} /> Simular torneo
            </button>
          </div>
          <TournamentTable rows={tournamentRows} />
        </section>
      )}
    </main>
  );
}

function TeamSelect({ label, value, teams, onChange }: {
  label: string;
  value: string;
  teams: TeamInfo[];
  onChange: (value: string) => void;
}) {
  const confederations = Array.from(new Set(teams.map((team) => team.confederation ?? "Otros")));
  return (
    <label className="team-select">
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {confederations.map((confed) => (
          <optgroup key={confed} label={confed}>
            {teams.filter((team) => (team.confederation ?? "Otros") === confed).map((team) => (
              <option key={team.name} value={team.name}>
                {team.name} - Grupo {team.group ?? "-"} - Elo {Math.round(team.elo)}
              </option>
            ))}
          </optgroup>
        ))}
      </select>
    </label>
  );
}

function PredictionPanel({ match, loading }: { match: MatchResult; loading: boolean }) {
  const score = match.most_likely_scoreline;
  const outcomes = [
    { label: `Gana ${match.team_a}`, value: match.probabilities.win_a },
    { label: "Empate", value: match.probabilities.draw },
    { label: `Gana ${match.team_b}`, value: match.probabilities.win_b }
  ];
  const bestOutcome = outcomes.reduce((best, current) => current.value > best.value ? current : best);

  return (
    <div className={`prediction-grid ${loading ? "loading" : ""}`}>
      <section className="result-band">
        <div>
          <span className="eyebrow">Prediccion del modelo entrenado</span>
          <h2>{bestOutcome.label}</h2>
          <p className="result-copy">
            Marcador mas probable: {match.team_a} {score?.goals_a ?? "-"} - {score?.goals_b ?? "-"} {match.team_b}
          </p>
        </div>
        <div className="confidence">{percent(bestOutcome.value)}</div>
      </section>

      <section className="prob-panel">
        <ProbBar label={`Gana ${match.team_a}`} value={match.probabilities.win_a} />
        <ProbBar label="Empate" value={match.probabilities.draw} />
        <ProbBar label={`Gana ${match.team_b}`} value={match.probabilities.win_b} />
      </section>

      <section className="metrics">
        <Metric label={`${match.team_a} xG`} value={match.expected_goals.team_a.toFixed(2)} />
        <Metric label={`${match.team_b} xG`} value={match.expected_goals.team_b.toFixed(2)} />
        <Metric label="Confianza" value={match.confidence} />
        <Metric label="Diferencia Elo" value={Math.round(match.elo.diff).toString()} />
      </section>

      <section className="scorelines">
        <strong>Marcadores posibles</strong>
        {match.top_scorelines.map((item) => (
          <div key={`${item.goals_a}-${item.goals_b}`}>
            <strong>{item.goals_a}-{item.goals_b}</strong>
            <span>{percent(item.probability)}</span>
          </div>
        ))}
      </section>

      <section className="features">
        <strong>Variables que mas influyen</strong>
        {match.top_features.length ? match.top_features.map((feature) => (
          <div key={feature.name}>
            <span>{feature.name}</span>
            <strong>{feature.direction}</strong>
          </div>
        )) : <p>La explicacion SHAP no esta disponible para el modelo cargado.</p>}
      </section>
    </div>
  );
}

function ProbBar({ label, value }: { label: string; value: number }) {
  return (
    <div className="prob-row">
      <div className="prob-label">
        <span>{label}</span>
        <strong>{percent(value)}</strong>
      </div>
      <div className="bar"><span style={{ width: `${Math.max(3, value * 100)}%` }} /></div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function TeamList({ title, teams }: { title: string; teams: string[] }) {
  return <div className="team-list"><strong>{title}</strong>{teams.map((team) => <span key={team}>{team}</span>)}</div>;
}

function DataTable({ rows }: { rows: GroupSimulationRow[] }) {
  if (!rows.length) return <p className="empty">Ejecuta una simulacion para ver la tabla.</p>;
  return (
    <table>
      <thead><tr><th>Equipo</th><th>1ro</th><th>2do</th><th>3ro</th><th>Directo</th><th>Pts</th><th>DG</th></tr></thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.team}>
            <td>{row.team}</td><td>{percent(row.prob_1st)}</td><td>{percent(row.prob_2nd)}</td>
            <td>{percent(row.prob_3rd)}</td><td>{percent(row.qualify_direct_prob)}</td>
            <td>{row.avg_pts.toFixed(2)}</td><td>{row.avg_gd.toFixed(2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function FixtureResults({ group, fixtures }: { group: string; fixtures: GroupFixturePrediction[] }) {
  if (!fixtures.length) return null;
  return (
    <section className="fixture-results">
      <strong>Resultados mas probables - Grupo {group}</strong>
      <div className="fixture-grid">
        {fixtures.map((fixture) => {
          const score = fixture.most_likely_scoreline;
          return (
            <div className="fixture-card" key={`${fixture.team_a}-${fixture.team_b}`}>
              <span>{fixture.team_a} vs {fixture.team_b}</span>
              <strong>{score ? `${score.goals_a}-${score.goals_b}` : "-"}</strong>
              <small>
                {score ? percent(score.probability) : "Sin prob."} · xG {fixture.expected_goals.team_a.toFixed(2)}-{fixture.expected_goals.team_b.toFixed(2)}
              </small>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TournamentTable({ rows }: { rows: TournamentRow[] }) {
  if (!rows.length) return <p className="empty">Ejecuta una simulacion para ver candidatos al titulo.</p>;
  return (
    <table>
      <thead><tr><th>Rank</th><th>Equipo</th><th>R32</th><th>QF</th><th>SF</th><th>Final</th><th>Campeon</th></tr></thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.team}>
            <td>{row.rank}</td><td>{row.team}</td><td>{percent(row.p_round_32)}</td>
            <td>{percent(row.p_quarterfinal)}</td><td>{percent(row.p_semifinal)}</td>
            <td>{percent(row.p_finalist)}</td><td><strong>{percent(row.p_champion)}</strong></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
