import React from "react";
import ReactDOM from "react-dom/client";
import { Activity, BarChart3, CalendarDays, Play, Trophy } from "lucide-react";
import {
  BracketResponse,
  BracketRound,
  fetchFixtures,
  fetchGroups,
  fetchTeams,
  FixturePrediction,
  GroupFixturePrediction,
  GroupSimulationRow,
  MatchResult,
  predictMatch,
  simulateGroup,
  simulateTournament,
  simulateTournamentBracket,
  TeamInfo,
  TournamentRow
} from "./api";
import "./styles.css";

type View = "predict" | "calendar" | "groups" | "tournament";
type CalendarGroupBy = "matchday" | "date";

// ── Traducciones al castellano ────────────────────────────────────────────────
const TEAMS_ES: Record<string, string> = {
  "Mexico": "México", "South Africa": "Sudáfrica", "South Korea": "Corea del Sur",
  "Czechia": "República Checa", "Canada": "Canadá", "Switzerland": "Suiza",
  "Qatar": "Qatar", "Bosnia and Herzegovina": "Bosnia y Herzegovina",
  "Brazil": "Brasil", "Morocco": "Marruecos", "Scotland": "Escocia", "Haiti": "Haití",
  "United States": "Estados Unidos", "Paraguay": "Paraguay", "Australia": "Australia",
  "Turkiye": "Turquía", "Germany": "Alemania", "Ecuador": "Ecuador",
  "Cote d'Ivoire": "Costa de Marfil", "Curacao": "Curazao",
  "Netherlands": "Países Bajos", "Sweden": "Suecia", "Japan": "Japón",
  "Tunisia": "Túnez", "Belgium": "Bélgica", "Egypt": "Egipto",
  "Iran": "Irán", "New Zealand": "Nueva Zelanda", "Spain": "España",
  "Uruguay": "Uruguay", "Saudi Arabia": "Arabia Saudita", "Cape Verde": "Cabo Verde",
  "France": "Francia", "Senegal": "Senegal", "Norway": "Noruega", "Iraq": "Irak",
  "Argentina": "Argentina", "Austria": "Austria", "Algeria": "Argelia",
  "Jordan": "Jordania", "Portugal": "Portugal", "Colombia": "Colombia",
  "Uzbekistan": "Uzbekistán", "DR Congo": "R. D. del Congo",
  "England": "Inglaterra", "Croatia": "Croacia", "Ghana": "Ghana", "Panama": "Panamá",
};

const MONTHS_ES: Record<number, string> = {
  0: "ene", 1: "feb", 2: "mar", 3: "abr", 4: "may", 5: "jun",
  6: "jul", 7: "ago", 8: "sep", 9: "oct", 10: "nov", 11: "dic",
};

function es(name: string): string {
  return TEAMS_ES[name] ?? name;
}

function percent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatDate(dateStr: string): string {
  const d = new Date(dateStr + "T00:00:00");
  return `${d.getDate()} ${MONTHS_ES[d.getMonth()] ?? ""}`;
}

// ── App ───────────────────────────────────────────────────────────────────────
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
  const [loadingText, setLoadingText] = React.useState("");
  const [initialLoading, setInitialLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // ── Calendar state ──
  const [fixtures, setFixtures] = React.useState<FixturePrediction[]>([]);
  const [calendarGroupBy, setCalendarGroupBy] = React.useState<CalendarGroupBy>("matchday");

  // ── Compare mode ──
  const [compareMode, setCompareMode] = React.useState(false);
  const [teamA2, setTeamA2] = React.useState("Brazil");
  const [teamB2, setTeamB2] = React.useState("Morocco");
  const [match2, setMatch2] = React.useState<MatchResult | null>(null);

  // ── Tournament bracket ──
  const [tournamentView, setTournamentView] = React.useState<"list" | "bracket">("list");
  const [bracketRounds, setBracketRounds] = React.useState<BracketRound[] | null>(null);

  // Auto-predict debounce
  const debounceRef = React.useRef<ReturnType<typeof setTimeout>>();

  React.useEffect(() => {
    setInitialLoading(true);
    Promise.all([fetchTeams(), fetchGroups()])
      .then(([teamResponse, groupResponse]) => {
        setTeams(teamResponse.teams);
        setGroups(groupResponse.groups);
      })
      .catch((err) => setError(err.message))
      .finally(() => setInitialLoading(false));
  }, []);

  React.useEffect(() => {
    if (teams.length) void runPrediction();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teams.length]);

  function autoPredict(team1: string, team2: string, setter: (m: MatchResult) => void) {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    if (team1 === team2) return;
    debounceRef.current = setTimeout(async () => {
      try {
        setter(await predictMatch(team1, team2));
      } catch { /* silent */ }
    }, 600);
  }

  async function runPrediction() {
    setLoading(true);
    setLoadingText("Calculando predicción...");
    setError(null);
    try {
      setMatch(await predictMatch(teamA, teamB));
      if (compareMode) setMatch2(await predictMatch(teamA2, teamB2));
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo calcular la predicción");
    } finally {
      setLoading(false);
    }
  }

  async function runGroup() {
    setLoading(true);
    setLoadingText("Simulando grupo...");
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
    setLoadingText("Simulando torneo...");
    setError(null);
    try {
      if (tournamentView === "bracket") {
        const response = await simulateTournamentBracket(5000);
        setBracketRounds(response.rounds);
      } else {
        const response = await simulateTournament(2500, 20);
        setTournamentRows(response.results);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "No se pudo simular el torneo");
    } finally {
      setLoading(false);
    }
  }

  async function loadFixtures() {
    if (fixtures.length > 0) return;
    setLoading(true);
    setLoadingText("Cargando calendario...");
    try {
      const data = await fetchFixtures();
      setFixtures(data.fixtures);
    } catch { /* silent */ }
    finally { setLoading(false); }
  }

  // ── Calendar grouping ───────────────────────────────────────────────────
  const groupedFixtures = React.useMemo(() => {
    if (!fixtures.length) return [] as { label: string; items: FixturePrediction[] }[];

    if (calendarGroupBy === "matchday") {
      const labels = ["Jornada 1", "Jornada 2", "Jornada 3"];
      return labels.map((label, i) => ({
        label: `${label} · ${_matchdayDateRange(fixtures, i + 1)}`,
        items: fixtures.filter(f => f.matchday === i + 1),
      })).filter(g => g.items.length > 0);
    }

    const byDate = new Map<string, FixturePrediction[]>();
    for (const f of fixtures) {
      const key = f.date;
      if (!byDate.has(key)) byDate.set(key, []);
      byDate.get(key)!.push(f);
    }
    return Array.from(byDate.entries()).map(([date, items]) => ({
      label: formatDate(date),
      items,
    }));
  }, [fixtures, calendarGroupBy]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>prode-ML</h1>
          <p>Predicciones del Mundial 2026</p>
        </div>
        <nav className="tabs" aria-label="Vistas">
          <button className={view === "predict" ? "active" : ""} onClick={() => setView("predict")}>
            <Activity size={18} /> Predicción
          </button>
          <button className={view === "calendar" ? "active" : ""} onClick={() => { setView("calendar"); loadFixtures(); }}>
            <CalendarDays size={18} /> Calendario
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
      {initialLoading && <LoadingOverlay text="Cargando equipos y grupos..." />}
      {loading && <LoadingOverlay text={loadingText} />}

      {/* ── Predict ─────────────────────────────────────────────────────── */}
      {view === "predict" && (
        <section className="workspace">
          <div className={`control-strip ${compareMode ? "dual" : ""}`}>
            <TeamCombobox label="Equipo 1" value={teamA} teams={teams} onChange={v => { setTeamA(v); autoPredict(v, teamB, setMatch); }} />
            <span className="versus">VS</span>
            <TeamCombobox label="Equipo 2" value={teamB} teams={teams} onChange={v => { setTeamB(v); autoPredict(teamA, v, setMatch); }} />
            {compareMode && (
              <>
                <TeamCombobox label="Equipo 3" value={teamA2} teams={teams} onChange={v => { setTeamA2(v); autoPredict(v, teamB2, setMatch2); }} />
                <span className="versus">VS</span>
                <TeamCombobox label="Equipo 4" value={teamB2} teams={teams} onChange={v => { setTeamB2(v); autoPredict(teamA2, v, setMatch2); }} />
              </>
            )}
            <div className="action-group">
              <button className={`secondary ${compareMode ? "active" : ""}`} onClick={() => { setCompareMode(!compareMode); if (!compareMode) { autoPredict(teamA2, teamB2, setMatch2); } }}>
                <Activity size={16} /> Comparar
              </button>
              <button className="primary" onClick={runPrediction} disabled={loading || teamA === teamB}>
                <Play size={18} /> Predecir
              </button>
            </div>
          </div>
          <div className={`prediction-dual ${compareMode ? "active" : ""}`}>
            {match && <PredictionPanel match={match} compact={compareMode} />}
            {compareMode && match2 && <PredictionPanel match={match2} compact={true} />}
          </div>
        </section>
      )}

      {/* ── Calendar ────────────────────────────────────────────────────── */}
      {view === "calendar" && (
        <section className="workspace">
          <div className="control-strip compact calendar-controls">
            <div className="toggle-group">
              <button
                className={`secondary ${calendarGroupBy === "matchday" ? "active" : ""}`}
                onClick={() => setCalendarGroupBy("matchday")}
              >Por jornada</button>
              <button
                className={`secondary ${calendarGroupBy === "date" ? "active" : ""}`}
                onClick={() => setCalendarGroupBy("date")}
              >Por fecha</button>
            </div>
          </div>
          <CalendarTimeline groups={groupedFixtures} />
        </section>
      )}

      {/* ── Groups ──────────────────────────────────────────────────────── */}
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

      {/* ── Tournament ──────────────────────────────────────────────────── */}
      {view === "tournament" && (
        <section className="workspace">
          <div className="control-strip compact">
            <div className="toggle-group">
              <button
                className={`secondary ${tournamentView === "list" ? "active" : ""}`}
                onClick={() => setTournamentView("list")}
              >Lista</button>
              <button
                className={`secondary ${tournamentView === "bracket" ? "active" : ""}`}
                onClick={() => setTournamentView("bracket")}
              >Diagrama</button>
            </div>
            <button className="primary" onClick={runTournament} disabled={loading}>
              <Trophy size={18} /> Simular torneo
            </button>
          </div>
          {tournamentView === "list" && <TournamentTable rows={tournamentRows} />}
          {tournamentView === "bracket" && <BracketTree rounds={bracketRounds} />}
        </section>
      )}
    </main>
  );
}

function _matchdayDateRange(fixtures: FixturePrediction[], md: number): string {
  const mds = fixtures.filter(f => f.matchday === md);
  if (!mds.length) return "";
  const dates = mds.map(f => f.date).sort();
  return `${formatDate(dates[0])}–${formatDate(dates[dates.length - 1])}`;
}

// ── TeamCombobox — selector con búsqueda ──────────────────────────────────────
function TeamCombobox({ label, value, teams, onChange }: {
  label: string;
  value: string;
  teams: TeamInfo[];
  onChange: (value: string) => void;
}) {
  const [query, setQuery] = React.useState("");
  const [open, setOpen] = React.useState(false);
  const wrapRef = React.useRef<HTMLDivElement>(null);
  const inputRef = React.useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    function handler(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
        setQuery("");
      }
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  React.useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  const filtered = React.useMemo(() => {
    if (!query) return teams;
    const q = query.toLowerCase();
    return teams.filter(t =>
      (TEAMS_ES[t.name] ?? t.name).toLowerCase().includes(q) ||
      t.name.toLowerCase().includes(q)
    );
  }, [teams, query]);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") { setOpen(false); setQuery(""); }
    if (e.key === "Enter" && filtered.length > 0) {
      onChange(filtered[0].name);
      setOpen(false);
      setQuery("");
    }
  }

  return (
    <div ref={wrapRef} className="team-combobox">
      <span className="combobox-label">{label}</span>
      <button
        type="button"
        className="combobox-trigger"
        onClick={() => setOpen(o => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="combobox-selected">{es(value) || "Seleccionar equipo…"}</span>
        <span className="combobox-chevron" aria-hidden>{open ? "▴" : "▾"}</span>
      </button>

      {open && (
        <div className="combobox-dropdown" role="listbox">
          <div className="combobox-search">
            <input
              ref={inputRef}
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Buscar equipo…"
              aria-label="Buscar equipo"
            />
          </div>
          <ul>
            {filtered.length === 0 && (
              <li className="combobox-empty">Sin resultados</li>
            )}
            {filtered.map(team => (
              <li
                key={team.name}
                role="option"
                aria-selected={team.name === value}
                className={team.name === value ? "active" : ""}
                onMouseDown={() => {
                  onChange(team.name);
                  setOpen(false);
                  setQuery("");
                }}
              >
                <span className="combobox-team-name">{es(team.name)}</span>
                <span className="combobox-team-meta">
                  Grupo {team.group ?? "-"} · Elo {Math.round(team.elo)}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── PredictionPanel ───────────────────────────────────────────────────────────
function PredictionPanel({ match, compact }: { match: MatchResult; compact?: boolean }) {
  const score = match.outcome_scoreline ?? match.most_likely_scoreline;
  const exactScore = match.exact_most_likely_scoreline;
  const teamAEs = es(match.team_a);
  const teamBEs = es(match.team_b);
  const outcomes = [
    { label: `Gana ${teamAEs}`, value: match.probabilities.win_a },
    { label: "Empate", value: match.probabilities.draw },
    { label: `Gana ${teamBEs}`, value: match.probabilities.win_b }
  ];
  const bestOutcome = outcomes.reduce((best, current) => current.value > best.value ? current : best);

  return (
    <div className={`prediction-grid ${compact ? "compact" : ""}`}>
      <section className="result-band">
        <div>
          <span className="eyebrow">Predicción del modelo entrenado</span>
          <h2>{bestOutcome.label}</h2>
          <p className="result-copy">
            Marcador más probable: {teamAEs} {score?.goals_a ?? "-"} - {score?.goals_b ?? "-"} {teamBEs}
          </p>
          {exactScore && score && (exactScore.goals_a !== score.goals_a || exactScore.goals_b !== score.goals_b) && (
            <p className="result-note">Marcador exacto global: {exactScore.goals_a}-{exactScore.goals_b}</p>
          )}
        </div>
        <div className="confidence">{percent(bestOutcome.value)}</div>
      </section>

      <section className="prob-panel">
        <ProbBar label={`Gana ${teamAEs}`} value={match.probabilities.win_a} />
        <ProbBar label="Empate" value={match.probabilities.draw} />
        <ProbBar label={`Gana ${teamBEs}`} value={match.probabilities.win_b} />
      </section>

      <section className="metrics">
        <Metric label={`${teamAEs} xG`} value={match.expected_goals.team_a.toFixed(2)} />
        <Metric label={`${teamBEs} xG`} value={match.expected_goals.team_b.toFixed(2)} />
        <Metric label="Confianza" value={match.confidence} />
        <Metric label="Diferencia Elo" value={Math.round(match.elo.diff).toString()} />
      </section>

      <section className="scorelines">
        <strong>Marcadores posibles</strong>
        {(compact ? match.top_scorelines.slice(0, 3) : match.top_scorelines).map((item) => (
          <div key={`${item.goals_a}-${item.goals_b}`}>
            <strong>{item.goals_a}-{item.goals_b}</strong>
            <span>{percent(item.probability)}</span>
          </div>
        ))}
      </section>

      {!compact && (
        <section className="features">
          <strong>Variables que más influyen</strong>
          {match.top_features.length ? match.top_features.map((feature) => (
            <div key={feature.name}>
              <span>{feature.name}</span>
              <strong>{feature.direction}</strong>
            </div>
          )) : <p>La explicación SHAP no está disponible para el modelo cargado.</p>}
        </section>
      )}
    </div>
  );
}

// ── Componentes simples ───────────────────────────────────────────────────────
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
  return (
    <div className="team-list">
      <strong>{title}</strong>
      {teams.map((team) => <span key={team}>{es(team)}</span>)}
    </div>
  );
}

function DataTable({ rows }: { rows: GroupSimulationRow[] }) {
  if (!rows.length) return <p className="empty">Ejecutá una simulación para ver la tabla.</p>;
  return (
    <table>
      <thead>
        <tr>
          <th>Equipo</th><th>1ro</th><th>2do</th><th>3ro</th>
          <th>Clasifica</th><th>Pts</th><th>DG</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.team}>
            <td>{es(row.team)}</td>
            <td>{percent(row.prob_1st)}</td>
            <td>{percent(row.prob_2nd)}</td>
            <td>{percent(row.prob_3rd)}</td>
            <td>{percent(row.qualify_direct_prob)}</td>
            <td>{row.avg_pts.toFixed(2)}</td>
            <td>{row.avg_gd.toFixed(2)}</td>
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
      <strong>Resultados más probables — Grupo {group}</strong>
      <div className="fixture-grid">
        {fixtures.map((fixture) => {
          const score = fixture.most_likely_scoreline;
          return (
            <div className="fixture-card" key={`${fixture.team_a}-${fixture.team_b}`}>
              <span>{es(fixture.team_a)} vs {es(fixture.team_b)}</span>
              <strong>{score ? `${score.goals_a}-${score.goals_b}` : "-"}</strong>
              <small>
                {score ? percent(score.probability) : "Sin prob."} · xG{" "}
                {fixture.expected_goals.team_a.toFixed(2)}-{fixture.expected_goals.team_b.toFixed(2)}
              </small>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function TournamentTable({ rows }: { rows: TournamentRow[] }) {
  if (!rows.length) return <p className="empty">Ejecutá una simulación para ver candidatos al título.</p>;
  return (
    <table>
      <thead>
        <tr>
          <th>Rank</th><th>Equipo</th><th>R32</th><th>Cuartos</th>
          <th>Semis</th><th>Final</th><th>Campeón</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.team}>
            <td>{row.rank}</td>
            <td>
              {es(row.team)}
              {row.title_tier === "candidate" && <span className="tier-label">Candidato</span>}
            </td>
            <td>{percent(row.p_round_32)}</td>
            <td>{percent(row.p_quarterfinal)}</td>
            <td>{percent(row.p_semifinal)}</td>
            <td>{percent(row.p_finalist)}</td>
            <td><strong>{percent(row.p_champion)}</strong></td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Calendar Timeline ─────────────────────────────────────────────────────────
function CalendarTimeline({ groups }: { groups: { label: string; items: FixturePrediction[] }[] }) {
  if (!groups.length) return <p className="empty">Cargando calendario del Mundial…</p>;

  return (
    <div className="calendar-timeline">
      {groups.map((group) => (
        <section className="timeline-group" key={group.label}>
          <h3 className="timeline-header">{group.label}</h3>
          <div className="fixture-grid">
            {group.items.map((f, i) => (
              <FixtureCard key={`${f.team_a}-${f.team_b}-${i}`} fixture={f} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function FixtureCard({ fixture }: { fixture: FixturePrediction }) {
  const score = fixture.outcome_scoreline;
  const outcomeLabel = fixture.predicted_outcome === "win_a"
    ? `Gana ${es(fixture.team_a)}`
    : fixture.predicted_outcome === "win_b"
      ? `Gana ${es(fixture.team_b)}`
      : "Empate";

  const confClass = fixture.confidence === "ALTO" ? "conf-alto"
    : fixture.confidence === "MEDIO" ? "conf-medio" : "conf-bajo";

  return (
    <div className="fixture-calendar-card">
      <div className="fcc-header">
        <span className="fcc-date">{formatDate(fixture.date)}</span>
        <span className="fcc-group">Grupo {fixture.group}</span>
      </div>
      <div className="fcc-teams">
        <span className="fcc-team-a">{es(fixture.team_a)}</span>
        <span className="fcc-vs">vs</span>
        <span className="fcc-team-b">{es(fixture.team_b)}</span>
      </div>
      <div className="fcc-score">
        <span className="fcc-scoreline">{score ? `${score.goals_a}-${score.goals_b}` : "-"}</span>
        <span className={`fcc-confidence ${confClass}`}>{fixture.confidence}</span>
      </div>
      <div className="fcc-probs">
        <span className="fcc-prob" style={{ flex: fixture.win_a * 100 }} title={`${percent(fixture.win_a)}`} />
        <span className="fcc-prob fcc-draw" style={{ flex: fixture.draw * 100 }} title={`${percent(fixture.draw)}`} />
        <span className="fcc-prob fcc-prob-b" style={{ flex: fixture.win_b * 100 }} title={`${percent(fixture.win_b)}`} />
      </div>
      <div className="fcc-footer">
        <span className="fcc-outcome">{outcomeLabel}</span>
        <span className="fcc-xg">xG {fixture.xg_a.toFixed(1)}-{fixture.xg_b.toFixed(1)}</span>
      </div>
    </div>
  );
}

// ── Bracket Mirror ────────────────────────────────────────────────────────────
function BracketTree({ rounds }: { rounds: BracketRound[] | null }) {
  if (!rounds || !rounds.length) return <p className="empty">Ejecutá la simulación para ver el diagrama del torneo.</p>;

  const r32 = rounds[0];
  const r16 = rounds[1];
  const qf  = rounds[2];
  const sf  = rounds[3];
  const fin = rounds[4];

  const finalTop = fin?.slots?.[0]?.teams?.[0];

  return (
    <div className="bracket-scroll" role="figure" aria-label="Diagrama del torneo">
      <div className="bracket-mirror">
        {/* ── Rama izquierda ───────────────────────────────────────────── */}
        <div className="bracket-branch bracket-left">
          <BracketRoundColumn side="left" round={r32} slotStart={0} slotEnd={7} connectorTo="r16" />
          <BracketRoundColumn side="left" round={r16} slotStart={0} slotEnd={3} connectorTo="qf" />
          <BracketRoundColumn side="left" round={qf}  slotStart={0} slotEnd={1} connectorTo="sf" />
        </div>

        {/* ── Centro: SF + Campeón ─────────────────────────────────────── */}
        <div className="bracket-center">
          <BracketRoundColumn side="left" round={sf} slotStart={0} slotEnd={0} connectorTo="final" compact />
          <ChampionBadge team={finalTop} />
          <BracketRoundColumn side="right" round={sf} slotStart={1} slotEnd={1} connectorTo="final" compact />
        </div>

        {/* ── Rama derecha ──────────────────────────────────────────────── */}
        <div className="bracket-branch bracket-right">
          <BracketRoundColumn side="right" round={qf}  slotStart={2} slotEnd={3} connectorTo="sf" />
          <BracketRoundColumn side="right" round={r16} slotStart={4} slotEnd={7} connectorTo="qf" />
          <BracketRoundColumn side="right" round={r32} slotStart={8} slotEnd={15} connectorTo="r16" />
        </div>
      </div>
    </div>
  );
}

function BracketRoundColumn({ side, round, slotStart, slotEnd, connectorTo, compact }: {
  side: "left" | "right";
  round: BracketRound;
  slotStart: number;
  slotEnd: number;
  connectorTo?: string;
  compact?: boolean;
}) {
  if (!round) return null;
  const slots = round.slots.slice(slotStart, slotEnd + 1);
  const connClass = connectorTo === "final" || connectorTo === "sf" ? "conn-granate" : "conn-gray";

  return (
    <div className={`bracket-col ${compact ? "bracket-col-compact" : ""}`}>
      <h4 className="bracket-col-title">{round.display}</h4>
      <div className="bracket-col-slots">
        {slots.map((slot) => (
          <BracketMatch key={slot.slot_index} slot={slot} side={side} connClass={connClass} />
        ))}
      </div>
    </div>
  );
}

function BracketMatch({ slot, side, connClass }: {
  slot: { slot_index: number; teams: { team: string; prob: number }[] };
  side: "left" | "right";
  connClass: string;
}) {
  const top = slot.teams[0];
  const second = slot.teams[1];

  return (
    <div className={`bracket-match ${side}`}>
      <div className="bracket-connector">
        <div className={`connector-line ${connClass}`} />
        <div className={`connector-diamond ${connClass}`} />
      </div>
      <div className="bracket-match-teams">
        <BracketTeamRow team={top} isFavorite />
        <BracketTeamRow team={second} />
      </div>
    </div>
  );
}

function BracketTeamRow({ team, isFavorite }: {
  team: { team: string; prob: number } | undefined;
  isFavorite?: boolean;
}) {
  if (!team) {
    return <div className="bracket-team-row bracket-team-empty"><span>—</span></div>;
  }
  return (
    <div className={`bracket-team-row ${isFavorite ? "favorite" : ""}`}>
      <span className="bracket-team-flag">{_flagFor(team.team)}</span>
      <span className="bracket-team-name">{es(team.team)}</span>
      <span className="bracket-team-prob">{percent(team.prob)}</span>
    </div>
  );
}

function ChampionBadge({ team }: { team: { team: string; prob: number } | undefined }) {
  return (
    <div className="champion-badge">
      <div className="champion-icon"><Trophy size={22} /></div>
      <div className="champion-label">CAMPEÓN</div>
      {team ? (
        <div className="champion-team">
          <span className="champion-flag">{_flagFor(team.team)}</span>
          <span className="champion-name">{es(team.team)}</span>
        </div>
      ) : (
        <div className="champion-team champion-placeholder">—</div>
      )}
    </div>
  );
}

const _FLAGS: Record<string, string> = {
  "Argentina": "🇦🇷", "Brazil": "🇧🇷", "Uruguay": "🇺🇾", "Paraguay": "🇵🇾",
  "Colombia": "🇨🇴", "Ecuador": "🇪�", "Germany": "🇩🇪", "France": "🇫🇷",
  "Spain": "🇪🇸", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Netherlands": "🇳🇱", "Portugal": "🇵🇹",
  "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Switzerland": "🇨🇭", "Norway": "🇳🇴",
  "Sweden": "🇸🇪", "Austria": "🇦🇹", "Czechia": "🇨🇿", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿",
  "Turkiye": "🇹🇷", "Bosnia and Herzegovina": "🇧🇦", "Mexico": "🇲🇽",
  "United States": "🇺🇸", "Canada": "🇨🇦", "Panama": "🇵🇦", "Haiti": "🇭🇹",
  "Curacao": "🇨🇼", "Morocco": "🇲🇦", "Senegal": "🇸🇳", "Egypt": "🇪🇬",
  "Algeria": "🇩🇿", "Tunisia": "🇹🇳", "Ghana": "🇬", "South Africa": "🇿🇦",
  "Cote d'Ivoire": "🇨🇮", "Cape Verde": "🇨🇻", "DR Congo": "🇨🇩",
  "Japan": "🇯🇵", "South Korea": "🇰🇷", "Australia": "🇦🇺",
  "Iran": "🇮🇷", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦", "Iraq": "🇮🇶",
  "Jordan": "🇯🇴", "Uzbekistan": "🇺🇿", "New Zealand": "🇳🇿",
};

function _flagFor(team: string): string {
  return _FLAGS[team] ?? "🏳️";
}

// ── Loading Overlay ──────────────────────────────────────────────────────────
function LoadingOverlay({ text }: { text?: string }) {
  return (
    <div className="loading-overlay" role="alert" aria-busy="true">
      <div className="loading-spinner">
        <div className="spinner-ring ring-outer" />
        <div className="spinner-ring ring-inner" />
      </div>
      {text && <p className="loading-text">{text}</p>}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
