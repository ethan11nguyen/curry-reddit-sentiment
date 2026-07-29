"""
load_pushshift_comments.py

Reads the filtered r/nba comments file (nba_full_season.jsonl -- one JSON
object per line, produced by filter_subreddit_streaming.py) and loads it
into the `comments` table in Postgres, as defined in sql/01_reddit_schema.sql.

Uses batched inserts (via psycopg2.extras.execute_values) rather than
one INSERT per row, since this is loading ~7.18M rows -- a naive
row-by-row INSERT loop would be impractically slow at this scale.

Usage:
    python load_pushshift_comments.py <path_to_jsonl>

Example:
    python load_pushshift_comments.py data/raw/nba_full_season.jsonl

Requires DB connection details as environment variables (matching .env):
    POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB
Assumes host=localhost, port=5432 (matching docker-compose.yml's port mapping).
"""
import sys
import os
import json
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

DB_HOST = "localhost"
DB_PORT = 5432
DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")

BATCH_SIZE = 5000

INSERT_SQL = """
    INSERT INTO comments (
        comment_id, author, body, subreddit, link_id, parent_id,
        permalink, score, controversiality, created_utc, retrieved_on,
        raw_json
    )
    VALUES %s
    ON CONFLICT (comment_id) DO NOTHING
"""

def row_from_obj(obj):
    """Extract the modeled columns from a raw pushshift comment object.
    Anything not explicitly pulled out here still gets preserved in
    raw_json, so no data is lost even if a field is missing here."""
    return (
        obj.get("id"),
        obj.get("author"),
        obj.get("body"),
        obj.get("subreddit"),
        obj.get("link_id"),
        obj.get("parent_id"),
        obj.get("permalink"),
        obj.get("score"),
        obj.get("controversiality"),
        obj.get("created_utc"),
        obj.get("retrieved_on"),
        json.dumps(obj),
    )

def load_file(jsonl_path):
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        user=DB_USER, password=DB_PASSWORD, dbname=DB_NAME
    )
    conn.autocommit = False
    cur = conn.cursor()

    batch = []
    total_read = 0
    total_inserted = 0
    total_skipped = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                total_skipped += 1
                continue

            # Skip rows missing required NOT NULL fields
            if not obj.get("id") or not obj.get("body") or not obj.get("subreddit"):
                total_skipped += 1
                continue

            batch.append(row_from_obj(obj))

            if len(batch) >= BATCH_SIZE:
                execute_values(cur, INSERT_SQL, batch)
                conn.commit()
                total_inserted += len(batch)
                batch = []
                print(f"  ...{total_read:,} lines read, {total_inserted:,} inserted so far")

        # insert any remaining rows in the final partial batch
        if batch:
            execute_values(cur, INSERT_SQL, batch)
            conn.commit()
            total_inserted += len(batch)

    cur.close()
    conn.close()
    print(f"Done: {total_inserted:,} rows inserted, {total_skipped:,} skipped, "
          f"{total_read:,} total lines read")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python load_pushshift_comments.py <path_to_jsonl>")
        sys.exit(1)
    load_file(sys.argv[1])
