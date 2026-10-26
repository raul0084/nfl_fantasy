"""Interactive dashboard for the NFL Fantasy ML Optimizer."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from fantasy_ml.data.update import update_current_season
from fantasy_ml.decisions.lineup import DEFAULT_SLOTS, optimize_lineup
from fantasy_ml.models.projections import project_week


ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "fantasy.duckdb"
PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"

ACCENT = "#22d3a6"
ACCENT_2 = "#3b82f6"

st.set_page_config(
    page_title="NFL Fantasy ML Optimizer",
    page_icon="🏈",
    layout="wide",
)

# Fragments rerun only their own block, which is what makes interactions feel fast.
fragment = getattr(st, "fragment", lambda f=None, **_: f if f else (lambda g: g))


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"], .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1280px;
        }}

        /* Hero */
        .hero {{
            padding: 1.6rem 2rem;
            border-radius: 20px;
            margin-bottom: 1.4rem;
            background:
                radial-gradient(1200px 300px at 0% 0%, rgba(34,211,166,.28), transparent 60%),
                radial-gradient(900px 300px at 100% 100%, rgba(59,130,246,.28), transparent 60%),
                linear-gradient(135deg, #0b1220 0%, #111a2e 100%);
            border: 1px solid rgba(255,255,255,.08);
            box-shadow: 0 10px 40px rgba(0,0,0,.25);
        }}
        .hero h1 {{
            margin: 0;
            padding: 0;
            font-size: 2.1rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #f8fafc;
        }}
        .hero h1 span {{
            background: linear-gradient(90deg, {ACCENT}, {ACCENT_2});
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .hero p {{
            margin: .35rem 0 0 0;
            color: #94a3b8;
            font-size: .95rem;
        }}
        .pill {{
            display: inline-block;
            margin: .8rem .4rem 0 0;
            padding: .2rem .7rem;
            font-size: .72rem;
            font-weight: 600;
            letter-spacing: .04em;
            text-transform: uppercase;
            border-radius: 999px;
            color: {ACCENT};
            background: rgba(34,211,166,.12);
            border: 1px solid rgba(34,211,166,.3);
        }}

        /* Tabs */
        div[data-baseweb="tab-list"] {{
            gap: .4rem;
            padding: .3rem;
            border-radius: 14px;
            background: rgba(127,127,127,.10);
            width: fit-content;
        }}
        button[data-baseweb="tab"] {{
            height: 2.5rem;
            padding: 0 1.2rem;
            border-radius: 10px;
            font-weight: 600;
            background: transparent;
            transition: all .18s ease;
        }}
        button[data-baseweb="tab"]:hover {{
            background: rgba(127,127,127,.15);
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
            color: #06121f !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] p {{
            color: #06121f !important;
        }}
        div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {{
            display: none;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            padding: 1rem 1.2rem;
            border-radius: 16px;
            background: rgba(127,127,127,.08);
            border: 1px solid rgba(127,127,127,.18);
            transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            border-color: {ACCENT};
            box-shadow: 0 8px 24px rgba(34,211,166,.12);
        }}
        div[data-testid="stMetricLabel"] p {{
            font-size: .75rem;
            font-weight: 600;
            letter-spacing: .05em;
            text-transform: uppercase;
            opacity: .7;
        }}
        div[data-testid="stMetricValue"] {{
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        /* Buttons */
        .stButton > button, .stDownloadButton > button {{
            border-radius: 12px;
            font-weight: 600;
            padding: .55rem 1.2rem;
            border: 1px solid rgba(127,127,127,.3);
            transition: all .18s ease;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            border-color: {ACCENT};
            box-shadow: 0 6px 18px rgba(34,211,166,.18);
        }}
        .stButton > button[kind="primary"] {{
            color: #06121f;
            border: none;
            background: linear-gradient(135deg, {ACCENT}, {ACCENT_2});
        }}
        .stButton > button[kind="primary"]:hover {{
            filter: brightness(1.08);
            box-shadow: 0 8px 24px rgba(59,130,246,.35);
        }}

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] {{
            border-radius: 12px !important;
        }}

        /* Tables / charts */
        div[data-testid="stDataFrame"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid rgba(127,127,127,.2);
        }}
        div[data-testid="stVegaLiteChart"] {{
            padding: .6rem;
            border-radius: 14px;
            background: rgba(127,127,127,.06);
            border: 1px solid rgba(127,127,127,.15);
        }}

        /* Section headings */
        h2, h3 {{
            font-weight: 700;
            letter-spacing: -0.01em;
        }}
        hr {{
            border: none;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(127,127,127,.4), transparent);
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            border-right: 1px solid rgba(127,127,127,.15);
        }}

        /* Alerts */
        div[data-testid="stAlert"] {{
            border-radius: 14px;
        }}

        /* Hide default chrome */
        #MainMenu, footer {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1>🏈 NFL Fantasy <span>ML Optimizer</span></h1>
            <p>Historical stats from DuckDB · Projections from your existing model</p>
            <span class="pill">Full PPR</span>
            <span class="pill">Ridge projections</span>
            <span class="pill">Lineup optimizer</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_df(df: pd.DataFrame, **kwargs) -> None:
    """Render a dataframe full-width across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", hide_index=True, **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


# --------------------------------------------------------------------------- #
# Data access (cached, invalidated automatically when the DB file changes)
# --------------------------------------------------------------------------- #
def db_version() -> int:
    return DB_PATH.stat().st_mtime_ns if DB_PATH.is_file() else 0


@st.cache_data(show_spinner=False, ttl=3600, max_entries=256)
def _cached_query(query: str, params: tuple, version: int) -> pd.DataFrame:
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.execute(query, list(params)).fetchdf()


def read_query(query: str, params: list | None = None) -> pd.DataFrame:
    """Run a cached read-only query against the DuckDB database."""
    return _cached_query(query, tuple(params or ()), db_version())


def available_tables() -> set[str]:
    tables = read_query(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
        """
    )
    return set(tables["table_name"].tolist())


def load_player_options(season: int | None = None) -> pd.DataFrame:
    """Load player choices from the database, not from a static file."""
    query = """
        SELECT
            player_id,
            COALESCE(
                MAX(NULLIF(player_display_name, '')),
                MAX(NULLIF(player_name, '')),
                player_id
            ) AS player_name,
            MAX(position) AS position
        FROM player_stats
        WHERE player_id IS NOT NULL
    """
    params: list = []

    if season is not None:
        query += " AND season = ?"
        params.append(season)

    query += """
        GROUP BY player_id
        ORDER BY player_name
    """
    return read_query(query, params)


def load_seasons() -> list[int]:
    seasons = read_query(
        "SELECT DISTINCT season FROM player_stats ORDER BY season DESC"
    )
    return sorted(
        {int(year) for year in seasons["season"].dropna().tolist()},
        reverse=True,
    )


def load_weekly_stats(player_id: str, season: int) -> pd.DataFrame:
    return read_query(
        """
        SELECT
            season,
            week,
            game_id,
            team,
            opponent_team,
            position,
            completions,
            attempts,
            passing_yards,
            passing_tds,
            passing_interceptions,
            carries,
            rushing_yards,
            rushing_tds,
            targets,
            receptions,
            receiving_yards,
            receiving_tds,
            fumbles_lost_total,
            fantasy_points_ppr
        FROM player_stats
        WHERE player_id = ?
          AND season = ?
          AND season_type = 'REG'
        ORDER BY week
        """,
        [player_id, season],
    )


def load_roster_weeks(season: int, week: int) -> list[int]:
    df = read_query(
        """
        SELECT DISTINCT week
        FROM weekly_rosters
        WHERE season = ?
          AND week <= ?
          AND week BETWEEN 1 AND 18
        ORDER BY week
        """,
        [int(season), int(week)],
    )
    return df["week"].astype(int).tolist()


@st.cache_data(show_spinner=False, ttl=3600)
def load_team_catalog(version: int) -> pd.DataFrame:
    """Cleaned player catalog with display labels, built once per DB version."""
    catalog = load_player_options()
    if catalog.empty:
        return catalog

    catalog = catalog.drop_duplicates("player_id").copy()
    catalog["player_id"] = catalog["player_id"].astype(str)
    catalog["position"] = (
        catalog["position"]
        .fillna("UNK")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"DST": "DEF", "D/ST": "DEF"})
    )
    catalog["player_name"] = catalog["player_name"].fillna("Unknown player")
    catalog["label"] = (
        catalog["player_name"]
        + " ("
        + catalog["position"]
        + ") — "
        + catalog["player_id"]
    )
    return catalog


# --------------------------------------------------------------------------- #
# Player Explorer
# --------------------------------------------------------------------------- #
@fragment
def show_player_explorer(seasons: list[int]) -> None:
    st.subheader("Player Explorer")

    left, right = st.columns(2)
    with left:
        season = st.selectbox(
            "Stats season",
            seasons,
            index=0,
            key="stats_season",
        )

    players = load_player_options(season)
    if players.empty:
        st.warning(f"No player records found for {season}.")
        return

    with right:
        position_options = ["All"] + sorted(
            players["position"].dropna().unique().tolist()
        )
        selected_position = st.selectbox(
            "Position filter",
            position_options,
            key="player_position_filter",
        )

    if selected_position != "All":
        players = players[players["position"] == selected_position]

    search = st.text_input(
        "Search player",
        placeholder="e.g. Josh Allen",
        key="player_search",
    )
    if search.strip():
        mask = players["player_name"].str.contains(
            search.strip(), case=False, na=False, regex=False
        )
        players = players[mask]

    if players.empty:
        st.info("No matching players. Try a different search or position.")
        return

    labels = {
        f"{row.player_name} ({row.position or 'UNK'}) — {row.player_id}": row.player_id
        for row in players.itertuples(index=False)
    }
    selected_label = st.selectbox("Player", list(labels.keys()), key="player_pick")
    player_id = labels[selected_label]

    weekly = load_weekly_stats(player_id, season).copy()
    if weekly.empty:
        st.info("No regular-season game records found for this player.")
        return

    weekly["fantasy_points_ppr"] = pd.to_numeric(
        weekly["fantasy_points_ppr"], errors="coerce"
    ).fillna(0)

    points = weekly["fantasy_points_ppr"]
    total_points = float(points.sum())
    games = len(weekly)
    avg_points = total_points / games if games else 0.0
    last_3 = float(points.tail(3).mean()) if games else 0.0
    last_5 = float(points.tail(5).mean()) if games else 0.0

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Season PPR points", f"{total_points:.1f}")
    m2.metric("Recorded game rows", games)
    m3.metric("Average PPR / row", f"{avg_points:.1f}")
    m4.metric("Last 3 recorded rows", f"{last_3:.1f}")

    st.caption(
        f"Last 5 recorded rows average: {last_5:.1f} PPR. "
        "These are database records, not a guarantee of games played."
    )

    chart_data = weekly[["week", "fantasy_points_ppr"]].set_index("week")
    st.line_chart(chart_data, y="fantasy_points_ppr", color=ACCENT)

    display_columns = [
        "week",
        "team",
        "opponent_team",
        "passing_yards",
        "passing_tds",
        "passing_interceptions",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "fumbles_lost_total",
        "fantasy_points_ppr",
    ]
    show_df(weekly[display_columns])


# --------------------------------------------------------------------------- #
# Weekly Projections
# --------------------------------------------------------------------------- #
@fragment
def show_projections(seasons: list[int]) -> None:
    """Generate weekly projections and preserve them across Streamlit reruns."""
    st.subheader("Weekly Projections")
    st.write(
        "Generate a forecast for a target week. The app uses the latest "
        "available roster snapshot at or before that week."
    )

    season_options = sorted({int(s) for s in seasons})
    if not season_options:
        st.warning("No seasons are available in the database.")
        return

    default_season = 2026 if 2026 in season_options else max(season_options)

    col1, col2 = st.columns(2)

    with col1:
        season = st.selectbox(
            "Target season",
            options=season_options,
            index=season_options.index(default_season),
            key="projection_season",
        )
        week = st.selectbox(
            "Target week",
            options=list(range(1, 19)),
            index=7,
            key="projection_week",
        )

    try:
        available_roster_weeks = load_roster_weeks(int(season), int(week))
    except Exception as exc:
        st.error(f"Could not read weekly roster snapshots: {exc}")
        return

    if not available_roster_weeks:
        st.error(
            f"No roster snapshots exist for {season} through Week {week}. "
            "Load roster data before generating this forecast."
        )
        return

    default_roster_week = max(available_roster_weeks)

    with col2:
        # Reset the stored choice if it is no longer a valid option.
        if st.session_state.get("projection_roster_week_select") not in available_roster_weeks:
            st.session_state["projection_roster_week_select"] = default_roster_week

        roster_week = st.selectbox(
            "Roster snapshot week",
            options=available_roster_weeks,
            key="projection_roster_week_select",
            help=(
                "Selects an existing roster snapshot no later than the target "
                "week. For example, Week 7 can be used for a Week 8 forecast."
            ),
        )

        alpha = st.number_input(
            "Ridge alpha",
            min_value=0.001,
            max_value=1000.0,
            value=1.0,
            step=0.5,
            key="projection_alpha",
        )

    # A forecast must train only on seasons before the target season.
    train_max = int(season) - 1
    if train_max < min(season_options):
        st.error("No earlier training season is available for this target season.")
        return

    default_train_season = max(min(season_options), min(2025, train_max))

    train_through_season = st.number_input(
        "Train through season",
        min_value=min(season_options),
        max_value=train_max,
        value=min(default_train_season, train_max),
        step=1,
        key="projection_train_through",
    )

    if st.button("Generate projections", type="primary", key="generate_projections"):
        if int(train_through_season) >= int(season):
            st.error("The training cutoff must be earlier than the target season.")
            return

        try:
            with st.spinner(
                f"Generating projections for {season}, Week {week} "
                f"using roster snapshot Week {roster_week}..."
            ):
                result = project_week(
                    season=int(season),
                    week=int(week),
                    db_path=DB_PATH,
                    roster_week=int(roster_week),
                    alpha=float(alpha),
                    train_through_season=int(train_through_season),
                )

                # Convert Polars output to pandas for Streamlit.
                projection_df = (
                    result.to_pandas() if hasattr(result, "to_pandas") else result.copy()
                )

                if projection_df.empty:
                    st.warning(
                        "The model returned no projections. Check data "
                        "coverage and the selected roster snapshot."
                    )
                    return

                st.session_state["latest_projections"] = projection_df
                st.session_state["latest_projection_meta"] = {
                    "season": int(season),
                    "week": int(week),
                    "roster_week": int(roster_week),
                    "alpha": float(alpha),
                    "train_through_season": int(train_through_season),
                }

                # Save a durable CSV snapshot.
                PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snapshot_path = (
                    PREDICTIONS_DIR
                    / f"projections_{season}_week_{week}_{timestamp}.csv"
                )
                projection_df.to_csv(snapshot_path, index=False)
                st.session_state["latest_projection_snapshot"] = str(snapshot_path)

            st.session_state["projection_notice"] = (
                f"Generated {len(projection_df):,} projections. "
                f"Roster snapshot: Week {roster_week}."
            )
            # Full rerun so the My Team tab immediately sees the new forecast.
            st.rerun()

        except Exception as exc:
            st.error(f"Projection generation failed: {exc}")
            return

    notice = st.session_state.pop("projection_notice", None)
    if notice:
        st.success(notice)

    latest = st.session_state.get("latest_projections")
    metadata = st.session_state.get("latest_projection_meta", {})

    if latest is None or latest.empty:
        st.info("No forecast is loaded yet. Click **Generate projections** above.")
        return

    st.divider()
    st.subheader(
        f"Loaded forecast: {metadata.get('season', '?')} "
        f"Week {metadata.get('week', '?')}"
    )
    st.caption(
        f"Roster snapshot Week {metadata.get('roster_week', '?')} · "
        f"Training through {metadata.get('train_through_season', '?')}"
    )

    snapshot_path = st.session_state.get("latest_projection_snapshot")
    if snapshot_path:
        st.caption(f"Saved CSV: {snapshot_path}")

    # Filtering affects only the display, not the stored projection dataset.
    display_df = latest.copy()

    if "position" in display_df.columns:
        positions = sorted(
            display_df["position"].dropna().astype(str).unique().tolist()
        )
        selected_positions = st.multiselect(
            "Filter by position",
            options=positions,
            default=positions,
            key="projection_position_filter",
        )
        display_df = display_df[
            display_df["position"].astype(str).isin(selected_positions)
        ]

    search = st.text_input(
        "Search players",
        placeholder="Player name, ID, or team",
        key="projection_player_search",
    )

    if search.strip():
        search_mask = pd.Series(False, index=display_df.index)
        for column in ("player_name", "player_display_name", "player_id", "team"):
            if column in display_df.columns:
                search_mask |= (
                    display_df[column]
                    .fillna("")
                    .astype(str)
                    .str.contains(search.strip(), case=False, regex=False)
                )
        display_df = display_df[search_mask]

    st.metric("Players displayed", f"{len(display_df):,}")

    if "projected_points_ppr" in display_df.columns:
        display_df["projected_points_ppr"] = pd.to_numeric(
            display_df["projected_points_ppr"], errors="coerce"
        )

    show_df(display_df)

    st.download_button(
        "Download displayed projections",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"projections_{metadata.get('season', 'season')}_"
            f"week_{metadata.get('week', 'week')}.csv"
        ),
        mime="text/csv",
        key="download_projections",
    )


# --------------------------------------------------------------------------- #
# Data Status
# --------------------------------------------------------------------------- #
@fragment
def show_data_status(tables: set[str], seasons: list[int]) -> None:
    st.subheader("Data Status")

    c1, c2, c3 = st.columns(3)
    c1.metric("Database", "Connected" if DB_PATH.is_file() else "Missing")
    c2.metric("Tables available", len(tables))
    c3.metric(
        "Historical seasons",
        f"{min(seasons)}–{max(seasons)}" if seasons else "None",
    )

    st.write("Available tables:", ", ".join(sorted(tables)) or "None")

    if "player_stats" in tables:
        coverage = read_query(
            """
            SELECT
                MIN(season) AS first_season,
                MAX(season) AS latest_season,
                MAX(week) AS latest_week,
                COUNT(*) AS rows,
                MAX(CASE WHEN season = ? THEN week END) AS current_season_latest_week
            FROM player_stats
            """,
            [datetime.now(UTC).year],
        )
        show_df(coverage)

    st.caption(
        "A database refresh updates available source data; it does not itself "
        "guarantee that a new model forecast has been generated."
    )


# --------------------------------------------------------------------------- #
# My Team
# --------------------------------------------------------------------------- #
def _clear_roster() -> None:
    st.session_state["my_team_picker"] = []
    st.session_state.pop("my_team_optimization", None)


def _best_lineup(optimizer_players: list[dict]) -> tuple[dict | None, Exception | None]:
    """Largest feasible lineup, tie-broken by projected points.

    Searches from the biggest slot count downward and stops at the first size
    that yields a feasible lineup, which avoids evaluating every subset.
    """
    standard_slot_order = [
        "QB", "RB1", "RB2", "WR1", "WR2", "TE", "FLEX", "K", "DEF"
    ]
    available_slots = {
        name: DEFAULT_SLOTS[name]
        for name in standard_slot_order
        if name in DEFAULT_SLOTS
    }
    slot_pool = list(available_slots)
    max_slots = min(len(optimizer_players), len(slot_pool))

    last_error: Exception | None = None

    for slot_count in range(max_slots, 0, -1):
        best_result = None
        best_points = float("-inf")

        for slot_names in combinations(slot_pool, slot_count):
            candidate_slots = {name: available_slots[name] for name in slot_names}
            try:
                candidate = optimize_lineup(optimizer_players, slots=candidate_slots)
            except (ValueError, KeyError) as exc:
                last_error = exc
                continue

            points = float(candidate["projected_points"])
            if points > best_points:
                best_result = candidate
                best_points = points

        if best_result is not None:
            return best_result, None

    return None, last_error


@fragment
def show_my_team() -> None:
    """Select a roster and automatically optimize its projected lineup."""
    st.subheader("My Team")
    st.write(
        "Select your players. The optimizer will automatically choose the "
        "largest feasible starting lineup and maximize its projected points. "
        "Kicker and defense are optional."
    )

    catalog = load_team_catalog(db_version())

    if catalog.empty:
        st.warning("No players are available in the player_stats table.")
        return

    label_to_id = dict(zip(catalog["label"], catalog["player_id"]))
    id_to_player = catalog.set_index("player_id").to_dict("index")
    labels = catalog["label"].tolist()

    # Drop any stored picks that no longer exist before the widget is created.
    st.session_state["my_team_picker"] = [
        label
        for label in st.session_state.get("my_team_picker", [])
        if label in label_to_id
    ]

    selected_labels = st.multiselect(
        "Search and select your players",
        options=labels,
        key="my_team_picker",
        placeholder="Type a player name to add them",
    )

    selected_ids = list(dict.fromkeys(label_to_id[label] for label in selected_labels))

    if not selected_ids:
        st.info("Select players above to build your roster.")
        st.session_state.pop("my_team_optimization", None)
        return

    roster_rows = [
        {
            "player_id": pid,
            "name": str(id_to_player[pid]["player_name"]),
            "position": str(id_to_player[pid]["position"]).upper(),
        }
        for pid in selected_ids
    ]

    head_l, head_r = st.columns([4, 1])
    head_l.caption(f"Roster size: {len(roster_rows)} players")
    head_r.button("Clear roster", key="clear_my_team", on_click=_clear_roster)

    show_df(pd.DataFrame(roster_rows))

    # Read the latest forecast generated in Weekly Projections.
    latest = st.session_state.get("latest_projections")
    metadata = st.session_state.get("latest_projection_meta")

    if latest is None or metadata is None:
        st.info("Generate a forecast in Weekly Projections first, then return here.")
        return

    projection_df = latest.copy()

    if projection_df.empty or "player_id" not in projection_df.columns:
        st.warning("The saved forecast is empty or lacks player IDs.")
        return

    if "projected_points_ppr" not in projection_df.columns:
        st.error("The saved forecast has no projected_points_ppr column.")
        return

    projection_df["player_id"] = projection_df["player_id"].astype(str)
    projection_df["projected_points_ppr"] = pd.to_numeric(
        projection_df["projected_points_ppr"], errors="coerce"
    )

    projection_lookup = (
        projection_df.dropna(subset=["projected_points_ppr"])
        .drop_duplicates("player_id")
        .set_index("player_id")
    )

    # Use only players with real, valid projections. Never invent scores.
    missing_ids = [pid for pid in selected_ids if pid not in projection_lookup.index]

    if missing_ids:
        missing_names = [
            f"{id_to_player[pid]['player_name']} ({pid})" for pid in missing_ids
        ]
        st.warning(
            "These selected players have no valid projection and will be "
            "excluded from optimization: " + ", ".join(missing_names)
        )

    optimizer_players = []
    for pid in selected_ids:
        if pid not in projection_lookup.index:
            continue

        info = id_to_player[pid]
        projection = projection_lookup.loc[pid]

        # Prefer the projection's position when available.
        raw_position = projection.get("position", None)
        if raw_position is None or pd.isna(raw_position):
            raw_position = info["position"]
        position = str(raw_position).strip().upper()
        position = {"DST": "DEF", "D/ST": "DEF"}.get(position, position)

        optimizer_players.append(
            {
                "player_id": pid,
                "name": str(info["player_name"]),
                "position": position,
                "projected_points": float(projection["projected_points_ppr"]),
            }
        )

    if not optimizer_players:
        st.warning("None of the selected players have valid projections for this week.")
        return

    st.caption(
        f"Using projections for {metadata.get('season', '?')} "
        f"Week {metadata.get('week', '?')}."
    )

    # Signature so a stale lineup isn't shown after roster/week changes.
    current_signature = (
        tuple(selected_ids),
        metadata.get("season"),
        metadata.get("week"),
        tuple(
            (p["player_id"], p["projected_points"], p["position"])
            for p in optimizer_players
        ),
    )

    # The optimizer only runs on click, so ordinary reruns stay instant.
    if st.button("Optimize lineup", type="primary", key="optimize_my_team"):
        with st.spinner("Optimizing lineup..."):
            best_result, last_error = _best_lineup(optimizer_players)

        if best_result is None:
            st.error(
                "No feasible starting lineup could be built from the selected "
                "players' positions."
            )
            if last_error:
                st.caption(f"Optimizer detail: {last_error}")
            return

        st.session_state["my_team_optimization"] = {
            "result": best_result,
            "signature": current_signature,
            "season": metadata.get("season"),
            "week": metadata.get("week"),
        }

    saved = st.session_state.get("my_team_optimization")

    if not saved:
        st.info("Click **Optimize lineup** to see your recommended starters.")
        return

    if saved["signature"] != current_signature:
        st.info(
            "Your roster or forecast has changed. Click **Optimize lineup** "
            "to calculate an updated lineup."
        )
        return

    result = saved["result"]

    st.divider()
    st.subheader("Recommended Starting Lineup")

    col1, col2, col3 = st.columns(3)
    col1.metric("Projected starter points", f"{result['projected_points']:.2f}")
    col2.metric("Bench players", str(len(result["bench"])))
    col3.metric("Starting slots filled", str(len(result["starters"])))

    show_df(pd.DataFrame(result["starters"]))

    st.subheader("Bench")
    if result["bench"]:
        show_df(pd.DataFrame(result["bench"]))
    else:
        st.caption("No bench players.")

    if missing_ids:
        st.caption(
            "Players without projections are excluded from both the lineup "
            "and bench tables. They have not been assigned fabricated scores."
        )


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def main() -> None:
    inject_css()
    render_hero()

    if not DB_PATH.is_file():
        st.error(f"DuckDB database not found at `{DB_PATH}`.")
        st.info("Initialize the database using the project's existing data workflow.")
        st.stop()

    try:
        tables = available_tables()
        if "player_stats" not in tables:
            st.error("The database has no `player_stats` table.")
            st.stop()
        seasons = load_seasons()
    except Exception as exc:
        st.error(f"Could not load database metadata: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Data controls")
        st.caption("Database reads are cached and refresh automatically when the data changes.")

        if st.button("Refresh current-season source data", type="secondary"):
            try:
                with st.spinner("Downloading and validating current-season data..."):
                    report = update_current_season(db_path=DB_PATH)
                st.cache_data.clear()
                st.session_state["refresh_report"] = report
                st.toast("Current-season data refresh completed.", icon="✅")
                st.rerun()
            except Exception as exc:
                st.error(f"Data refresh failed: {exc}")
                st.exception(exc)

        report = st.session_state.get("refresh_report")
        if report is not None:
            with st.expander("Last refresh report"):
                st.json(report)

        st.caption(
            "This action accesses nflverse sources and may take time. "
            "It is separate from projection generation."
        )

    overview, projections_tab, team_tab, status_tab = st.tabs(
        ["Player Explorer", "Weekly Projections", "My Team", "Data Status"]
    )

    with overview:
        show_player_explorer(seasons)

    with projections_tab:
        show_projections(seasons)

    with team_tab:
        show_my_team()

    with status_tab:
        show_data_status(tables, seasons)


if __name__ == "__main__":
    main()


# """Interactive dashboard for the NFL Fantasy ML Optimizer."""

# from __future__ import annotations

# from datetime import UTC, datetime
# from pathlib import Path

# import duckdb
# import pandas as pd
# import streamlit as st

# from fantasy_ml.data.update import update_current_season
# from fantasy_ml.models.projections import project_week
# from fantasy_ml.decisions.lineup import optimize_lineup
# from fantasy_ml.decisions.sleeper_roster import (
#     sleeper_roster_positions_to_slots,
# )


# ROOT = Path(__file__).resolve().parents[2]
# DB_PATH = ROOT / "data" / "fantasy.duckdb"
# PREDICTIONS_DIR = ROOT / "artifacts" / "predictions"

# st.set_page_config(
#     page_title="NFL Fantasy ML Optimizer",
#     page_icon="🏈",
#     layout="wide",
# )

# st.title("🏈 NFL Fantasy ML Optimizer")
# st.caption("Full PPR · Historical stats from DuckDB · Projections from your existing model")


# def read_query(query: str, params: list | None = None) -> pd.DataFrame:
#     """Run a read-only query against the current DuckDB database."""
#     if not DB_PATH.is_file():
#         raise FileNotFoundError(f"Database not found: {DB_PATH}")

#     with duckdb.connect(str(DB_PATH), read_only=True) as con:
#         return con.execute(query, params or []).fetchdf()


# def available_tables() -> set[str]:
#     tables = read_query(
#         """
#         SELECT table_name
#         FROM information_schema.tables
#         WHERE table_schema = 'main'
#         """
#     )
#     return set(tables["table_name"].tolist())


# def load_player_options(season: int | None = None) -> pd.DataFrame:
#     """Load player choices from the database, not from a static file."""
#     query = """
#         SELECT
#             player_id,
#             COALESCE(
#                 MAX(NULLIF(player_display_name, '')),
#                 MAX(NULLIF(player_name, '')),
#                 player_id
#             ) AS player_name,
#             MAX(position) AS position
#         FROM player_stats
#         WHERE player_id IS NOT NULL
#     """
#     params: list = []

#     if season is not None:
#         query += " AND season = ?"
#         params.append(season)

#     query += """
#         GROUP BY player_id
#         ORDER BY player_name
#     """
#     return read_query(query, params)


# def load_seasons() -> list[int]:
#     seasons = read_query(
#         "SELECT DISTINCT season FROM player_stats ORDER BY season DESC"
#     )
#     return sorted(
#         {int(year) for year in seasons["season"].dropna().tolist()},
#         reverse=True,
#     )


# def load_weekly_stats(player_id: str, season: int) -> pd.DataFrame:
#     return read_query(
#         """
#         SELECT
#             season,
#             week,
#             game_id,
#             team,
#             opponent_team,
#             position,
#             completions,
#             attempts,
#             passing_yards,
#             passing_tds,
#             passing_interceptions,
#             carries,
#             rushing_yards,
#             rushing_tds,
#             targets,
#             receptions,
#             receiving_yards,
#             receiving_tds,
#             fumbles_lost_total,
#             fantasy_points_ppr
#         FROM player_stats
#         WHERE player_id = ?
#           AND season = ?
#           AND season_type = 'REG'
#         ORDER BY week
#         """,
#         [player_id, season],
#     )


# def show_player_explorer(seasons: list[int]) -> None:
#     st.subheader("Player Explorer")

#     left, right = st.columns(2)
#     with left:
#         season = st.selectbox(
#             "Stats season",
#             seasons,
#             index=0,
#             key="stats_season",
#         )

#     players = load_player_options(season)
#     if players.empty:
#         st.warning(f"No player records found for {season}.")
#         return

#     with right:
#         position_options = ["All"] + sorted(
#             players["position"].dropna().unique().tolist()
#         )
#         selected_position = st.selectbox(
#             "Position filter",
#             position_options,
#             key="player_position_filter",
#         )

#     if selected_position != "All":
#         players = players[players["position"] == selected_position]

#     search = st.text_input("Search player", placeholder="e.g. Josh Allen")
#     if search.strip():
#         mask = players["player_name"].str.contains(
#             search.strip(), case=False, na=False, regex=False
#         )
#         players = players[mask]

#     if players.empty:
#         st.info("No matching players. Try a different search or position.")
#         return

#     labels = {
#         f"{row.player_name} ({row.position or 'UNK'}) — {row.player_id}": row.player_id
#         for row in players.itertuples(index=False)
#     }
#     selected_label = st.selectbox("Player", list(labels.keys()))
#     player_id = labels[selected_label]

#     weekly = load_weekly_stats(player_id, season)
#     if weekly.empty:
#         st.info("No regular-season game records found for this player.")
#         return

#     weekly["fantasy_points_ppr"] = pd.to_numeric(
#         weekly["fantasy_points_ppr"], errors="coerce"
#     ).fillna(0)

#     points = weekly["fantasy_points_ppr"]
#     total_points = float(points.sum())
#     games = len(weekly)
#     avg_points = total_points / games if games else 0.0
#     last_3 = float(points.tail(3).mean()) if games else 0.0
#     last_5 = float(points.tail(5).mean()) if games else 0.0

#     m1, m2, m3, m4 = st.columns(4)
#     m1.metric("Season PPR points", f"{total_points:.1f}")
#     m2.metric("Recorded game rows", games)
#     m3.metric("Average PPR / row", f"{avg_points:.1f}")
#     m4.metric("Last 3 recorded rows", f"{last_3:.1f}")

#     st.caption(
#         f"Last 5 recorded rows average: {last_5:.1f} PPR. "
#         "These are database records, not a guarantee of games played."
#     )

#     chart_data = weekly[["week", "fantasy_points_ppr"]].set_index("week")
#     st.line_chart(chart_data, y="fantasy_points_ppr")

#     display_columns = [
#         "week",
#         "team",
#         "opponent_team",
#         "passing_yards",
#         "passing_tds",
#         "passing_interceptions",
#         "carries",
#         "rushing_yards",
#         "rushing_tds",
#         "targets",
#         "receptions",
#         "receiving_yards",
#         "receiving_tds",
#         "fumbles_lost_total",
#         "fantasy_points_ppr",
#     ]
#     st.dataframe(
#         weekly[display_columns],
#         use_container_width=True,
#         hide_index=True,
#     )




# def show_projections(seasons: list[int]) -> None:
#     """Generate weekly projections and preserve them across Streamlit reruns."""
#     from datetime import datetime
#     from pathlib import Path

#     import duckdb
#     import pandas as pd
#     import streamlit as st

#     from fantasy_ml.models.projections import project_week

#     st.subheader("Weekly Projections")
#     st.write(
#         "Generate a forecast for a target week. The app uses the latest "
#         "available roster snapshot at or before that week."
#     )

#     # Normalize season options.
#     season_options = sorted({int(s) for s in seasons})
#     if not season_options:
#         st.warning("No seasons are available in the database.")
#         return

#     default_season = 2026 if 2026 in season_options else max(season_options)

#     col1, col2 = st.columns(2)

#     with col1:
#         season = st.selectbox(
#             "Target season",
#             options=season_options,
#             index=season_options.index(default_season),
#             key="projection_season",
#         )

#         week = st.selectbox(
#             "Target week",
#             options=list(range(1, 19)),
#             index=7,
#             key="projection_week",
#         )

#     # Find actual roster snapshots for the selected season and target week.
#     try:
#         with duckdb.connect(str(DB_PATH), read_only=True) as con:
#             available_roster_weeks = con.execute(
#                 """
#                 SELECT DISTINCT week
#                 FROM weekly_rosters
#                 WHERE season = ?
#                   AND week <= ?
#                   AND week BETWEEN 1 AND 18
#                 ORDER BY week
#                 """,
#                 [int(season), int(week)],
#             ).fetchdf()["week"].astype(int).tolist()
#     except Exception as exc:
#         st.error(f"Could not read weekly roster snapshots: {exc}")
#         return

#     if not available_roster_weeks:
#         st.error(
#             f"No roster snapshots exist for {season} through Week {week}. "
#             "Load roster data before generating this forecast."
#         )
#         return

#     default_roster_week = max(available_roster_weeks)

#     with col2:
#         roster_week = st.selectbox(
#             "Roster snapshot week",
#             options=available_roster_weeks,
#             index=available_roster_weeks.index(default_roster_week),
#             key="projection_roster_week_select",
#             help=(
#                 "Selects an existing roster snapshot no later than the target "
#                 "week. For example, Week 7 can be used for a Week 8 forecast."
#             ),
#         )

#         alpha = st.number_input(
#             "Ridge alpha",
#             min_value=0.001,
#             max_value=1000.0,
#             value=1.0,
#             step=0.5,
#             key="projection_alpha",
#         )

#     # A forecast must train only on seasons before the target season.
#     train_max = int(season) - 1
#     if train_max < min(season_options):
#         st.error(
#             "No earlier training season is available for this target season."
#         )
#         return

#     default_train_season = min(2025, train_max)
#     default_train_season = max(min(season_options), default_train_season)

#     train_through_season = st.number_input(
#         "Train through season",
#         min_value=min(season_options),
#         max_value=train_max,
#         value=default_train_season,
#         step=1,
#         key="projection_train_through",
#     )

#     # Generate only when explicitly requested.
#     if st.button(
#         "Generate projections",
#         type="primary",
#         key="generate_projections",
#     ):
#         if int(train_through_season) >= int(season):
#             st.error(
#                 "The training cutoff must be earlier than the target season."
#             )
#             return

#         try:
#             with st.spinner(
#                 f"Generating projections for {season}, Week {week} "
#                 f"using roster snapshot Week {roster_week}..."
#             ):
#                 result = project_week(
#                     season=int(season),
#                     week=int(week),
#                     db_path=DB_PATH,
#                     roster_week=int(roster_week),
#                     alpha=float(alpha),
#                     train_through_season=int(train_through_season),
#                 )

#                 # Convert Polars output to pandas for Streamlit.
#                 if hasattr(result, "to_pandas"):
#                     projection_df = result.to_pandas()
#                 else:
#                     projection_df = result.copy()

#                 if projection_df.empty:
#                     st.warning(
#                         "The model returned no projections. Check data "
#                         "coverage and the selected roster snapshot."
#                     )
#                     return

#                 # Persist the forecast across normal Streamlit reruns/tab changes.
#                 st.session_state["latest_projections"] = projection_df
#                 st.session_state["latest_projection_meta"] = {
#                     "season": int(season),
#                     "week": int(week),
#                     "roster_week": int(roster_week),
#                     "alpha": float(alpha),
#                     "train_through_season": int(train_through_season),
#                 }

#                 # Save a durable CSV snapshot.
#                 project_root = Path(DB_PATH).resolve().parent.parent
#                 predictions_dir = project_root / "artifacts" / "predictions"
#                 predictions_dir.mkdir(parents=True, exist_ok=True)

#                 timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                 snapshot_path = (
#                     predictions_dir
#                     / f"projections_{season}_week_{week}_{timestamp}.csv"
#                 )
#                 projection_df.to_csv(snapshot_path, index=False)

#                 st.session_state["latest_projection_snapshot"] = str(
#                     snapshot_path
#                 )

#             st.success(
#                 f"Generated {len(projection_df):,} projections. "
#                 f"Roster snapshot: Week {roster_week}."
#             )

#         except Exception as exc:
#             st.error(f"Projection generation failed: {exc}")
#             return

#     # Always re-read the saved forecast from session state.
#     latest = st.session_state.get("latest_projections")
#     metadata = st.session_state.get("latest_projection_meta", {})

#     if latest is None or latest.empty:
#         st.info(
#             "No forecast is loaded yet. Click **Generate projections** above."
#         )
#         return

#     st.divider()
#     st.subheader(
#         f"Loaded forecast: {metadata.get('season', '?')} "
#         f"Week {metadata.get('week', '?')}"
#     )

#     st.caption(
#         f"Roster snapshot Week {metadata.get('roster_week', '?')} · "
#         f"Training through {metadata.get('train_through_season', '?')}"
#     )

#     snapshot_path = st.session_state.get("latest_projection_snapshot")
#     if snapshot_path:
#         st.caption(f"Saved CSV: {snapshot_path}")

#     # Filtering affects only the display, not the stored projection dataset.
#     display_df = latest.copy()

#     if "position" in display_df.columns:
#         positions = sorted(
#             display_df["position"].dropna().astype(str).unique().tolist()
#         )
#         selected_positions = st.multiselect(
#             "Filter by position",
#             options=positions,
#             default=positions,
#             key="projection_position_filter",
#         )
#         display_df = display_df[
#             display_df["position"].astype(str).isin(selected_positions)
#         ]

#     search = st.text_input(
#         "Search players",
#         placeholder="Player name, ID, or team",
#         key="projection_player_search",
#     )

#     if search.strip():
#         search_mask = pd.Series(False, index=display_df.index)

#         for column in (
#             "player_name",
#             "player_display_name",
#             "player_id",
#             "team",
#         ):
#             if column in display_df.columns:
#                 search_mask |= (
#                     display_df[column]
#                     .fillna("")
#                     .astype(str)
#                     .str.contains(search.strip(), case=False, regex=False)
#                 )

#         display_df = display_df[search_mask]

#     st.metric("Players displayed", f"{len(display_df):,}")

#     if "projected_points_ppr" in display_df.columns:
#         display_df["projected_points_ppr"] = pd.to_numeric(
#             display_df["projected_points_ppr"],
#             errors="coerce",
#         )

#     st.dataframe(
#         display_df,
#         use_container_width=True,
#         hide_index=True,
#     )

#     st.download_button(
#         "Download displayed projections",
#         data=display_df.to_csv(index=False).encode("utf-8"),
#         file_name=(
#             f"projections_{metadata.get('season', 'season')}_"
#             f"week_{metadata.get('week', 'week')}.csv"
#         ),
#         mime="text/csv",
#         key="download_projections",
#     )



# def show_data_status(tables: set[str], seasons: list[int]) -> None:
#     st.subheader("Data Status")

#     c1, c2, c3 = st.columns(3)
#     c1.metric("Database", "Connected" if DB_PATH.is_file() else "Missing")
#     c2.metric("Tables available", len(tables))
#     c3.metric(
#         "Historical seasons",
#         f"{min(seasons)}–{max(seasons)}" if seasons else "None",
#     )

#     st.write("Available tables:", ", ".join(sorted(tables)) or "None")

#     if "player_stats" in tables:
#         coverage = read_query(
#             """
#             SELECT
#                 MIN(season) AS first_season,
#                 MAX(season) AS latest_season,
#                 MAX(week) AS latest_week,
#                 COUNT(*) AS rows,
#                 MAX(CASE WHEN season = ? THEN week END) AS current_season_latest_week
#             FROM player_stats
#             """,
#             [datetime.now(UTC).year],
#         )
#         st.dataframe(coverage, use_container_width=True, hide_index=True)

#     st.caption(
#         "A database refresh updates available source data; it does not itself "
#         "guarantee that a new model forecast has been generated."
#     )



# def show_my_team() -> None:
#     """Select a roster and automatically optimize its projected lineup."""
#     from itertools import combinations

#     import pandas as pd
#     import streamlit as st

#     from fantasy_ml.decisions.lineup import DEFAULT_SLOTS, optimize_lineup

#     st.subheader("My Team")
#     st.write(
#         "Select your players. The optimizer will automatically choose the "
#         "largest feasible starting lineup and maximize its projected points. "
#         "Kicker and defense are optional."
#     )

#     # Load the player catalog.
#     catalog = load_player_options()

#     if catalog.empty:
#         st.warning("No players are available in the player_stats table.")
#         return

#     catalog = catalog.drop_duplicates("player_id").copy()
#     catalog["player_id"] = catalog["player_id"].astype(str)
#     catalog["position"] = (
#         catalog["position"].fillna("UNK").astype(str).str.strip().str.upper()
#     )
#     catalog["player_name"] = catalog["player_name"].fillna("Unknown player")

#     # Normalize common position aliases.
#     catalog["position"] = catalog["position"].replace(
#         {"DST": "DEF", "D/ST": "DEF"}
#     )

#     catalog["label"] = catalog.apply(
#         lambda row: (
#             f"{row['player_name']} ({row['position']}) — {row['player_id']}"
#         ),
#         axis=1,
#     )

#     label_to_id = dict(zip(catalog["label"], catalog["player_id"]))
#     id_to_player = catalog.set_index("player_id").to_dict("index")
#     labels = catalog["label"].tolist()

#     # Persist roster selection across ordinary Streamlit reruns.
#     if "my_team_player_ids" not in st.session_state:
#         st.session_state["my_team_player_ids"] = []

#     existing_ids = set(catalog["player_id"])
#     st.session_state["my_team_player_ids"] = [
#         str(pid)
#         for pid in st.session_state["my_team_player_ids"]
#         if str(pid) in existing_ids
#     ]

#     default_labels = [
#         label
#         for label, pid in label_to_id.items()
#         if pid in st.session_state["my_team_player_ids"]
#     ]

#     selected_labels = st.multiselect(
#         "Search and select your players",
#         options=labels,
#         default=default_labels,
#         key="my_team_picker",
#         placeholder="Type a player name to add them",
#     )

#     selected_ids = list(
#         dict.fromkeys(label_to_id[label] for label in selected_labels)
#     )
#     st.session_state["my_team_player_ids"] = selected_ids

#     if not selected_ids:
#         st.info("Select players above to build your roster.")
#         st.session_state.pop("my_team_optimization", None)
#         return

#     roster_rows = [
#         {
#             "player_id": pid,
#             "name": str(id_to_player[pid]["player_name"]),
#             "position": str(id_to_player[pid]["position"]).upper(),
#         }
#         for pid in selected_ids
#     ]

#     st.caption(f"Roster size: {len(roster_rows)} players")

#     st.dataframe(
#         pd.DataFrame(roster_rows),
#         use_container_width=True,
#         hide_index=True,
#     )

#     if st.button("Clear roster", key="clear_my_team"):
#         st.session_state["my_team_player_ids"] = []
#         st.session_state["my_team_picker"] = []
#         st.session_state.pop("my_team_optimization", None)
#         st.rerun()

#     # Read the latest forecast generated in Weekly Projections.
#     latest = st.session_state.get("latest_projections")
#     metadata = st.session_state.get("latest_projection_meta")

#     if latest is None or metadata is None:
#         st.info(
#             "Generate a forecast in Weekly Projections first, then return here."
#         )
#         return

#     projection_df = latest.copy()

#     if projection_df.empty or "player_id" not in projection_df.columns:
#         st.warning("The saved forecast is empty or lacks player IDs.")
#         return

#     if "projected_points_ppr" not in projection_df.columns:
#         st.error("The saved forecast has no projected_points_ppr column.")
#         return

#     projection_df["player_id"] = projection_df["player_id"].astype(str)
#     projection_df["projected_points_ppr"] = pd.to_numeric(
#         projection_df["projected_points_ppr"],
#         errors="coerce",
#     )

#     projection_lookup = (
#         projection_df.dropna(subset=["projected_points_ppr"])
#         .drop_duplicates("player_id")
#         .set_index("player_id")
#     )

#     # Use only players with real, valid projections. Never invent scores.
#     missing_ids = [
#         pid for pid in selected_ids if pid not in projection_lookup.index
#     ]

#     if missing_ids:
#         missing_names = [
#             f"{id_to_player[pid]['player_name']} ({pid})"
#             for pid in missing_ids
#         ]
#         st.warning(
#             "These selected players have no valid projection and will be "
#             "excluded from optimization: " + ", ".join(missing_names)
#         )

#     optimizer_players = []

#     for pid in selected_ids:
#         if pid not in projection_lookup.index:
#             continue

#         info = id_to_player[pid]
#         projection = projection_lookup.loc[pid]

#         # Prefer the projection's position when available.
#         position = str(
#             projection.get("position", info["position"])
#         ).strip().upper()

#         position = {"DST": "DEF", "D/ST": "DEF"}.get(position, position)

#         optimizer_players.append(
#             {
#                 "player_id": pid,
#                 "name": str(info["player_name"]),
#                 "position": position,
#                 "projected_points": float(
#                     projection["projected_points_ppr"]
#                 ),
#             }
#         )

#     if not optimizer_players:
#         st.warning(
#             "None of the selected players have valid projections for this week."
#         )
#         return

#     st.caption(
#         f"Using projections for {metadata.get('season', '?')} "
#         f"Week {metadata.get('week', '?')}."
#     )

#     # Standard position slots, with optional K and DEF.
#     # Try every subset to find the largest feasible lineup. Among lineups
#     # with the same number of starters, keep the one with the most points.
#     standard_slot_order = [
#         "QB", "RB1", "RB2", "WR1", "WR2", "TE", "FLEX", "K", "DEF"
#     ]

#     available_slots = {
#         name: DEFAULT_SLOTS[name]
#         for name in standard_slot_order
#         if name in DEFAULT_SLOTS
#     }

#     best_result = None
#     best_slot_names = ()
#     best_count = -1
#     best_points = float("-inf")
#     last_error = None

#     # A lineup can have no more starters than available projected players.
#     max_slots = min(len(optimizer_players), len(standard_slot_order))

#     for slot_count in range(1, max_slots + 1):
#         for slot_names in combinations(standard_slot_order, slot_count):
#             candidate_slots = {
#                 name: available_slots[name] for name in slot_names
#             }

#             try:
#                 candidate = optimize_lineup(
#                     optimizer_players,
#                     slots=candidate_slots,
#                 )
#             except (ValueError, KeyError) as exc:
#                 last_error = exc
#                 continue

#             points = float(candidate["projected_points"])

#             if (
#                 slot_count > best_count
#                 or (slot_count == best_count and points > best_points)
#             ):
#                 best_result = candidate
#                 best_slot_names = slot_names
#                 best_count = slot_count
#                 best_points = points

#     if best_result is None:
#         st.error(
#             "No feasible starting lineup could be built from the selected "
#             "players' positions."
#         )
#         if last_error:
#             st.caption(f"Optimizer detail: {last_error}")
#         return

#     # Save a signature so a stale lineup isn't shown after roster/week changes.
#     current_signature = (
#         tuple(selected_ids),
#         metadata.get("season"),
#         metadata.get("week"),
#         tuple(best_slot_names),
#         tuple(
#             (p["player_id"], p["projected_points"], p["position"])
#             for p in optimizer_players
#         ),
#     )

#     if st.button(
#         "Optimize lineup",
#         type="primary",
#         key="optimize_my_team",
#     ):
#         st.session_state["my_team_optimization"] = {
#             "result": best_result,
#             "signature": current_signature,
#             "season": metadata.get("season"),
#             "week": metadata.get("week"),
#         }

#     saved = st.session_state.get("my_team_optimization")

#     if not saved:
#         st.info("Click **Optimize lineup** to see your recommended starters.")
#         return

#     if saved["signature"] != current_signature:
#         st.info(
#             "Your roster or forecast has changed. Click **Optimize lineup** "
#             "to calculate an updated lineup."
#         )
#         return

#     result = saved["result"]

#     st.divider()
#     st.subheader("Recommended Starting Lineup")

#     col1, col2, col3 = st.columns(3)
#     col1.metric("Projected starter points", f"{result['projected_points']:.2f}")
#     col2.metric("Bench players", str(len(result["bench"])))
#     col3.metric("Starting slots filled", str(len(result["starters"])))

#     st.dataframe(
#         pd.DataFrame(result["starters"]),
#         use_container_width=True,
#         hide_index=True,
#     )

#     st.subheader("Bench")
#     if result["bench"]:
#         st.dataframe(
#             pd.DataFrame(result["bench"]),
#             use_container_width=True,
#             hide_index=True,
#         )
#     else:
#         st.caption("No bench players.")

#     if missing_ids:
#         st.caption(
#             "Players without projections are excluded from both the lineup "
#             "and bench tables. They have not been assigned fabricated scores."
#         )




# def main() -> None:
#     if not DB_PATH.is_file():
#         st.error(f"DuckDB database not found at `{DB_PATH}`.")
#         st.info("Initialize the database using the project's existing data workflow.")
#         st.stop()

#     try:
#         tables = available_tables()
#         if "player_stats" not in tables:
#             st.error("The database has no `player_stats` table.")
#             st.stop()
#         seasons = load_seasons()
#     except Exception as exc:
#         st.error(f"Could not load database metadata: {exc}")
#         st.stop()

#     with st.sidebar:
#         st.header("Data controls")
#         st.caption("Database reads are live. Refreshing the page rereads DuckDB.")

#         if st.button("Refresh current-season source data", type="secondary"):
#             try:
#                 with st.spinner("Downloading and validating current-season data..."):
#                     report = update_current_season(db_path=DB_PATH)
#                 st.success("Current-season data refresh completed.")
#                 st.json(report)
#                 st.rerun()
#             except Exception as exc:
#                 st.error(f"Data refresh failed: {exc}")
#                 st.exception(exc)

#         st.caption(
#             "This action accesses nflverse sources and may take time. "
#             "It is separate from projection generation."
#         )

#     overview, projections_tab, team_tab, status_tab = st.tabs(
#         ["Player Explorer", "Weekly Projections", "My Team", "Data Status"]
#     )

#     with overview:
#         show_player_explorer(seasons)

#     with projections_tab:
#         show_projections(seasons)

#     with team_tab:
#         show_my_team()

#     with status_tab:
#         show_data_status(tables, seasons)


# if __name__ == "__main__":
#     main()
