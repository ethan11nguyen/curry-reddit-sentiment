"""
stratified_sample_comments.py

Selects a stratified sample of comments for LLM sentiment scoring, using
two tiers:
    - Baseline: BASE_PER_DAY comments randomly sampled from EVERY day in
      the season window (Oct 2024 - June 2025), preserving daily time-
      series coverage even on non-game days.
    - Game-day boost: an additional GAME_DAY_BOOST comments on days SGA
      actually played (per player_stats.game_date), since those are the
      days the correlation analysis in 03_aggregation_queries.sql
      actually depends on.

Sampling is done directly in Postgres via ROW_NUMBER() OVER (PARTITION BY
date ORDER BY random()), which is far more efficient at 7.18M rows than
pulling everything into Python first.

Output: a new table `stratified_sample` containing just the selected
comment_ids and a boolean flag for whether that row came from the game-day
boost -- keeps sampling logic decoupled from the actual scoring step,
which reads from this table separately.

Usage:
    python stratified_sample_comments.py
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")

BASE_PER_DAY = 50
GAME_DAY_BOOST = 150

CREATE_TABLE_SQL = """
    DROP TABLE IF EXISTS stratified_sample;
    CREATE TABLE stratified_sample (
        comment_id      TEXT PRIMARY KEY REFERENCES comments(comment_id),
        comment_date    DATE NOT NULL,
        is_game_day     BOOLEAN NOT NULL,
        sample_tier     TEXT NOT NULL  -- 'baseline' or 'game_day_boost'
    );
"""

# Baseline: BASE_PER_DAY random comments from every day in the season window
BASELINE_SAMPLE_SQL = """
    INSERT INTO stratified_sample (comment_id, comment_date, is_game_day, sample_tier)
    SELECT comment_id, comment_date, is_game_day, 'baseline'
    FROM (
        SELECT
            c.comment_id,
            DATE(c.created_at) AS comment_date,
            (p.game_date IS NOT NULL) AS is_game_day,
            ROW_NUMBER() OVER (
                PARTITION BY DATE(c.created_at)
                ORDER BY random()
            ) AS rn
        FROM comments c
        LEFT JOIN player_stats p ON p.game_date = DATE(c.created_at)
        WHERE DATE(c.created_at) BETWEEN '2024-10-01' AND '2025-06-30'
    ) ranked
    WHERE rn <= %s
"""

# Game-day boost: additional comments on game days only, excluding
# comment_ids already picked in the baseline pass above
GAME_DAY_BOOST_SQL = """
    INSERT INTO stratified_sample (comment_id, comment_date, is_game_day, sample_tier)
    SELECT comment_id, comment_date, is_game_day, 'game_day_boost'
    FROM (
        SELECT
            c.comment_id,
            DATE(c.created_at) AS comment_date,
            TRUE AS is_game_day,
            ROW_NUMBER() OVER (
                PARTITION BY DATE(c.created_at)
                ORDER BY random()
            ) AS rn
        FROM comments c
        JOIN player_stats p ON p.game_date = DATE(c.created_at)
        WHERE c.comment_id NOT IN (SELECT comment_id FROM stratified_sample)
    ) ranked
    WHERE rn <= %s
"""

SUMMARY_SQL = """
    SELECT
        sample_tier,
        COUNT(*) AS n_comments,
        COUNT(DISTINCT comment_date) AS n_days
    FROM stratified_sample
    GROUP BY sample_tier
    ORDER BY sample_tier;
"""

def run():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    conn.autocommit = False
    cur = conn.cursor()

    print("Creating stratified_sample table...")
    cur.execute(CREATE_TABLE_SQL)
    conn.commit()

    print(f"Selecting baseline sample ({BASE_PER_DAY}/day, all 273 days)...")
    cur.execute(BASELINE_SAMPLE_SQL, (BASE_PER_DAY,))
    conn.commit()

    print(f"Selecting game-day boost (+{GAME_DAY_BOOST}/game day)...")
    cur.execute(GAME_DAY_BOOST_SQL, (GAME_DAY_BOOST,))
    conn.commit()

    cur.execute(SUMMARY_SQL)
    rows = cur.fetchall()
    print("\nSummary:")
    total = 0
    for tier, n_comments, n_days in rows:
        print(f"  {tier}: {n_comments:,} comments across {n_days} days")
        total += n_comments
    print(f"  TOTAL: {total:,} comments selected for LLM scoring")

    cur.close()
    conn.close()

if __name__ == "__main__":
    run()
