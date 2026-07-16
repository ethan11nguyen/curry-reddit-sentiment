"""
nba_stats_pull.py

Pulls Stephen Curry's 2014-15 Regular Season + Playoffs game logs via nba_api
and loads them into the curry_sentiment Postgres DB's `player_stats` table.

Scoped deliberately to the 2014-15 season only (not full career) to match
the Reddit dataset's May 2015 window.

Setup:
    pip install nba_api --break-system-packages
Run:
    python scripts/nba_stats_pull.py
"""

import os
import time
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

PLAYER_FULL_NAME = "Stephen Curry"
SEASON = "2014-15"
SEASON_TYPES = ["Regular Season", "Playoffs"]

# nba_api's stats.nba.com endpoints are notoriously flaky/slow — bump timeout
# and add a small retry loop rather than letting it fail on the first hiccup.
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

load_dotenv()

PG_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "curry_sentiment"),
    "user": os.getenv("POSTGRES_USER", "curry_admin"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


def get_pg_conn():
    return psycopg2.connect(**PG_CONFIG)


def find_player_id(full_name):
    matches = players.find_players_by_full_name(full_name)
    if not matches:
        raise SystemExit(f"ERROR: no player found matching '{full_name}'")
    if len(matches) > 1:
        print(f"Warning: multiple matches for '{full_name}', using first: {matches[0]}")
    return matches[0]["id"]


def fetch_game_log(player_id, season, season_type):
    """Fetch game log with basic retry, since stats.nba.com times out often."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=season,
                season_type_all_star=season_type,
                timeout=REQUEST_TIMEOUT,
            )
            df = log.get_data_frames()[0]
            return df
        except Exception as e:  # nba_api raises plain requests exceptions
            last_error = e
            print(f"  attempt {attempt}/{MAX_RETRIES} failed ({season_type}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)
    raise SystemExit(f"ERROR: could not fetch {season_type} log after {MAX_RETRIES} attempts: {last_error}")


def parse_matchup(matchup):
    """
    MATCHUP looks like 'GSW vs. NOP' (home) or 'GSW @ NOP' (away).
    Returns (opponent, home_away).
    """
    if not matchup:
        return None, None
    if "vs." in matchup:
        opponent = matchup.split("vs.")[-1].strip()
        return opponent, "home"
    if "@" in matchup:
        opponent = matchup.split("@")[-1].strip()
        return opponent, "away"
    return None, None


def rows_from_df(df, player_name):
    fetched_at = datetime.now(timezone.utc)
    rows = []
    for _, r in df.iterrows():
        opponent, home_away = parse_matchup(r.get("MATCHUP"))
        game_date = None
        raw_date = r.get("GAME_DATE")
        if raw_date:
            try:
                game_date = datetime.strptime(raw_date, "%b %d, %Y").date()
            except ValueError:
                game_date = None

        rows.append(
            (
                r.get("Game_ID"),
                player_name,
                game_date,
                r.get("MATCHUP"),
                opponent,
                home_away,
                r.get("WL"),
                r.get("MIN"),
                r.get("PTS"),
                r.get("REB"),
                r.get("AST"),
                r.get("STL"),
                r.get("BLK"),
                r.get("TOV"),
                r.get("FG_PCT"),
                r.get("FG3_PCT"),
                r.get("FT_PCT"),
                r.get("PLUS_MINUS"),
                fetched_at,
            )
        )
    return rows


def insert_rows(pg_conn, rows):
    if not rows:
        return
    with pg_conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO player_stats
                (game_id, player_name, game_date, matchup, opponent,
                 home_away, win_loss, minutes, points, rebounds, assists,
                 steals, blocks, turnovers, fg_pct, fg3_pct, ft_pct,
                 plus_minus, fetched_at)
            VALUES %s
            ON CONFLICT (game_id) DO NOTHING
            """,
            rows,
        )
    pg_conn.commit()


def main():
    print(f"Looking up player ID for: {PLAYER_FULL_NAME}")
    player_id = find_player_id(PLAYER_FULL_NAME)
    print(f"  found player_id={player_id}\n")

    pg_conn = get_pg_conn()
    total_inserted = 0

    try:
        for season_type in SEASON_TYPES:
            print(f"Fetching {SEASON} {season_type} game log...")
            df = fetch_game_log(player_id, SEASON, season_type)
            print(f"  got {len(df)} games")

            rows = rows_from_df(df, PLAYER_FULL_NAME)
            insert_rows(pg_conn, rows)
            total_inserted += len(rows)
            print(f"  inserted (or skipped as dupes): {len(rows)} rows\n")

            # be polite to stats.nba.com between calls
            time.sleep(1)
    finally:
        pg_conn.close()

    print(f"Done. Processed {total_inserted} total game rows across {SEASON_TYPES}.")


if __name__ == "__main__":
    main()
