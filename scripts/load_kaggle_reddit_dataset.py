"""
load_kaggle_reddit_dataset.py

Streams the Kaggle "Reddit Comments May 2015" dataset (database.sqlite) and
loads Curry-related comments from r/nba into the curry_sentiment Postgres DB.

Source dataset: https://www.kaggle.com/datasets/kaggle/reddit-comments-may-2015
The dataset ships as a single SQLite table (commonly named "May2015") with
columns including: id, subreddit, body, author, created_utc, score, link_id,
parent_id.

This script does NOT load the whole ~30GB file into memory or pandas. It
streams rows out of SQLite in chunks via a server-side cursor and writes to
Postgres in batches, using ON CONFLICT DO NOTHING so it's safe to re-run.

Setup:
1. Download + unzip the Kaggle dataset to get database.sqlite
2. Update SQLITE_PATH below to point at that file
3. Make sure .env has your Postgres credentials (see .env.example)
4. Run: python scripts/load_kaggle_reddit_dataset.py
"""

import os
import sqlite3
import sys
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config — update this to point at your local copy of the Kaggle file
# ---------------------------------------------------------------------------
SQLITE_PATH = "/Users/ethannguyen/Developer/active/curry-sentiment-project-data/database.sqlite"


# Table name inside the Kaggle sqlite dump. Some redistributions of this
# dataset name the table "May2015", others just "data" — check with:
#   sqlite3 database.sqlite ".tables"
# and update if needed.
SQLITE_TABLE = "May2015"

BATCH_SIZE = 5000

# Keyword filter (case-insensitive, matched against comment body)
KEYWORDS = ["curry", "steph", "chef curry", "stephen curry"]

PLACEHOLDER_TITLE = "[Post metadata unavailable -- sourced from Kaggle comment dataset]"

# ---------------------------------------------------------------------------
# DB connection
# ---------------------------------------------------------------------------
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


def get_sqlite_conn(path):
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: SQLITE_PATH does not point to a real file: {path}\n"
            "Update SQLITE_PATH at the top of this script."
        )
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def strip_t3_prefix(link_id):
    """link_id in the Kaggle dump looks like 't3_abc123'; posts.post_id
    should just be 'abc123' to stay consistent with a live-scraped schema."""
    if link_id is None:
        return None
    return link_id[3:] if link_id.startswith("t3_") else link_id


def build_keyword_where_clause():
    # Case-insensitive LIKE filter across all keywords, OR'd together.
    clauses = " OR ".join(["LOWER(body) LIKE ?"] * len(KEYWORDS))
    params = [f"%{kw.lower()}%" for kw in KEYWORDS]
    return clauses, params


def fetch_batches(sqlite_conn):
    where_clause, params = build_keyword_where_clause()
    query = f"""
        SELECT id, link_id, parent_id, body, author, created_utc, score
        FROM {SQLITE_TABLE}
        WHERE subreddit = 'nba' AND ({where_clause})
    """
    cursor = sqlite_conn.cursor()
    cursor.execute(query, params)

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break
        yield rows


def insert_batch(pg_conn, rows):
    scraped_at = datetime.now(timezone.utc)

    # Dedup post_ids within this batch for the placeholder insert
    seen_post_ids = set()
    post_rows = []
    comment_rows = []

    for row in rows:
        post_id = strip_t3_prefix(row["link_id"])
        if post_id and post_id not in seen_post_ids:
            seen_post_ids.add(post_id)
            post_rows.append((post_id, PLACEHOLDER_TITLE, scraped_at))

        # created_utc in the Kaggle dump is a unix timestamp (int/str)
        try:
            created_utc = datetime.fromtimestamp(int(row["created_utc"]), tz=timezone.utc)
        except (TypeError, ValueError):
            created_utc = None

        comment_rows.append(
            (
                row["id"],
                post_id,
                row["parent_id"],
                row["body"],
                row["author"],
                created_utc,
                row["score"],
                scraped_at,
            )
        )

    with pg_conn.cursor() as cur:
        if post_rows:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO posts (post_id, title, scraped_at)
                VALUES %s
                ON CONFLICT (post_id) DO NOTHING
                """,
                post_rows,
            )

        if comment_rows:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO comments
                    (comment_id, post_id, parent_id, body, author,
                     created_utc, score, scraped_at)
                VALUES %s
                ON CONFLICT (comment_id) DO NOTHING
                """,
                comment_rows,
            )

    pg_conn.commit()


def main():
    print(f"Reading from: {SQLITE_PATH}")
    print(f"Table: {SQLITE_TABLE}")
    print(f"Keywords: {KEYWORDS}")
    print("This can take a while — there's no index on `subreddit` in the raw table.\n")

    sqlite_conn = get_sqlite_conn(SQLITE_PATH)
    pg_conn = get_pg_conn()

    total_comments = 0
    total_batches = 0

    try:
        for batch in fetch_batches(sqlite_conn):
            insert_batch(pg_conn, batch)
            total_batches += 1
            total_comments += len(batch)
            print(f"  batch {total_batches}: +{len(batch)} comments (running total: {total_comments})")
    finally:
        sqlite_conn.close()
        pg_conn.close()

    print(f"\nDone. Loaded {total_comments} comments across {total_batches} batches.")


if __name__ == "__main__":
    main()
