"""
export_validation_sample.py

Pulls a random sample of Curry-related comments and exports them to CSV for
manual labeling. The goal: empirically measure how often a comment's overall
sentiment is genuinely about Curry vs. incidental/comparative (e.g. Curry
mentioned only as a comparison point for another player).

This is a measurement step, not part of the automated pipeline. You label
the sample by hand, then use the results to report a concrete noise-rate
figure in your findings writeup (e.g. "in a random sample of N comments,
X% were incidental/comparative mentions rather than direct Curry sentiment"),
rather than just noting the limitation vaguely.

Workflow:
    1. Run this script -> produces docs/validation_sample.csv
    2. Open the CSV (Excel, Numbers, Google Sheets, whatever) and fill in
       the `manual_subject_label` column for each row using the values:
         - "about_curry"     : sentiment is genuinely directed at Curry
         - "incidental"      : Curry mentioned in passing / as a reference
                                point, sentiment is about someone/something else
         - "comparative"     : sentence directly compares Curry to another
                                player -- sentiment is ambiguous/split
         - "unclear"         : can't tell / not really about sentiment at all
    3. Once labeled, come back and I'll help you compute agreement rates
       between VADER's sentence-filtered score and your manual labels.

Run:
    python scripts/export_validation_sample.py
"""

import csv
import os
import random

import psycopg2
from dotenv import load_dotenv

SAMPLE_SIZE = 150
RANDOM_SEED = 42  # fixed seed so the sample is reproducible
OUTPUT_PATH = "docs/validation_sample.csv"

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


def fetch_sample(pg_conn, size, seed):
    # Use Postgres's own random sampling (TABLESAMPLE isn't seedable in a
    # simple way across versions, so use ORDER BY random() with setseed()
    # for reproducibility -- fine at this table size, ~19k rows).
    with pg_conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", (seed / 1000.0,))  # setseed wants [-1, 1]
        cur.execute(
            """
            SELECT c.comment_id, c.body, s.sentiment_score, s.sentiment_label
            FROM comments c
            LEFT JOIN sentiment_scores s
                ON c.comment_id = s.comment_id
                AND s.model_version = 'vader_sentence_filtered_v1'
            WHERE c.body IS NOT NULL
            ORDER BY random()
            LIMIT %s
            """,
            (size,),
        )
        return cur.fetchall()


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    pg_conn = get_pg_conn()
    try:
        rows = fetch_sample(pg_conn, SAMPLE_SIZE, RANDOM_SEED)
    finally:
        pg_conn.close()

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "comment_id",
                "body",
                "vader_sentiment_score",
                "vader_sentiment_label",
                "manual_subject_label",  # blank -- you fill this in
                "notes",  # blank -- optional free text
            ]
        )
        for comment_id, body, score, label in rows:
            writer.writerow([comment_id, body, score, label, "", ""])

    print(f"Exported {len(rows)} comments to {OUTPUT_PATH}")
    print("Fill in the 'manual_subject_label' column, then bring it back for analysis.")


if __name__ == "__main__":
    main()
