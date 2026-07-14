"""
sga_json_scraper.py

Bridge/stopgap data collector for the SGA sentiment project.

Uses Reddit's public, read-only .json endpoints (the same pages your browser
loads, just requested as structured JSON instead of rendered HTML) rather
than PRAW/OAuth. This requires NO API key or approval, since it's just
fetching public pages. Swap this out for the PRAW-based scraper once
Data API access is approved -- the database schema and insert logic below
will carry over unchanged.

Reddit's public JSON endpoints still expect a descriptive User-Agent and
still enforce (unauthenticated, stricter) rate limits, so this script is
deliberately conservative about request pacing.
"""

import os
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# --- Config ---------------------------------------------------------------

SUBREDDIT = "nba"
SEARCH_TERMS = [
    "Shai Gilgeous-Alexander",
    "Gilgeous-Alexander",
    "Gilgeous"
]
REQUEST_DELAY_SECONDS = 2.5  # conservative pacing for unauthenticated requests
POSTS_PER_SEARCH = 100       # Reddit's max per page

HEADERS = {
    "User-Agent": os.getenv(
        "REDDIT_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://old.reddit.com/",
}

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": os.getenv("POSTGRES_DB"),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
}


# --- Reddit fetching --------------------------------------------------------

def search_subreddit(subreddit: str, query: str, after: str | None = None) -> dict:
    """Hit the public search.json endpoint for a subreddit."""
    url = f"https://old.reddit.com/r/{subreddit}/search.json"
    params = {
        "q": query,
        "restrict_sr": "on",
        "sort": "new",
        "limit": POSTS_PER_SEARCH,
        "t": "all",
    }
    if after:
        params["after"] = after

    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_post_comments(subreddit: str, post_id: str) -> list[dict]:
    """Hit the public comments.json endpoint for a single post."""
    url = f"https://old.reddit.com/r/{subreddit}/comments/{post_id}.json"
    resp = requests.get(url, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # data[0] = the post itself, data[1] = the comment tree
    return data[1]["data"]["children"] if len(data) > 1 else []


def flatten_comments(children: list[dict], post_id: str) -> list[dict]:
    """Recursively flatten Reddit's nested comment tree into flat rows."""
    flat = []
    for child in children:
        if child.get("kind") != "t1":  # skip "more comments" stubs etc.
            continue
        c = child["data"]
        flat.append({
            "comment_id": c["id"],
            "post_id": post_id,
            "parent_id": c["parent_id"].split("_", 1)[-1],
            "body": c.get("body", ""),
            "author": c.get("author", "[deleted]"),
            "created_utc": c["created_utc"],
            "score": c.get("score", 0),
            "is_submitter": c.get("is_submitter", False),
        })
        # recurse into replies, if any
        replies = c.get("replies")
        if isinstance(replies, dict):
            flat.extend(flatten_comments(replies["data"]["children"], post_id))
    return flat


# --- Database ---------------------------------------------------------------

def upsert_posts(conn, posts: list[dict]) -> None:
    if not posts:
        return
    rows = [
        (
            p["id"], p["title"], p.get("selftext", ""), p.get("author", "[deleted]"),
            p["created_utc"], p.get("score", 0), p.get("upvote_ratio"),
            p.get("num_comments", 0), p.get("url", ""), p.get("link_flair_text"),
        )
        for p in posts
    ]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO posts (post_id, title, selftext, author, created_utc,
                                score, upvote_ratio, num_comments, url, flair)
            VALUES %s
            ON CONFLICT (post_id) DO NOTHING
        """, rows, template="(%s, %s, %s, %s, to_timestamp(%s), %s, %s, %s, %s, %s)")
    conn.commit()


def upsert_comments(conn, comments: list[dict]) -> None:
    if not comments:
        return
    rows = [
        (
            c["comment_id"], c["post_id"], c["parent_id"], c["body"],
            c["author"], c["created_utc"], c["score"], c["is_submitter"],
        )
        for c in comments
    ]
    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO comments (comment_id, post_id, parent_id, body,
                                   author, created_utc, score, is_submitter)
            VALUES %s
            ON CONFLICT (comment_id) DO NOTHING
        """, rows, template="(%s, %s, %s, %s, %s, to_timestamp(%s), %s, %s)")
    conn.commit()


# --- Main workflow -----------------------------------------------------------

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    print(f"Connected to Postgres at {DB_CONFIG['host']}:{DB_CONFIG['port']}")

    seen_post_ids = set()

    for term in SEARCH_TERMS:
        print(f"\nSearching r/{SUBREDDIT} for '{term}'...")
        after = None
        page = 1

        while True:
            data = search_subreddit(SUBREDDIT, term, after=after)
            children = data["data"]["children"]
            if not children:
                break

            posts = [child["data"] for child in children]
            new_posts = [p for p in posts if p["id"] not in seen_post_ids]
            seen_post_ids.update(p["id"] for p in posts)

            upsert_posts(conn, new_posts)
            print(f"  page {page}: {len(new_posts)} new posts (of {len(posts)} total)")

            after = data["data"].get("after")
            page += 1
            time.sleep(REQUEST_DELAY_SECONDS)

            if not after:
                break

    print(f"\nTotal unique posts collected: {len(seen_post_ids)}")
    print("Fetching comments for each post...")

    for i, post_id in enumerate(seen_post_ids, 1):
        try:
            children = fetch_post_comments(SUBREDDIT, post_id)
            flat_comments = flatten_comments(children, post_id)
            upsert_comments(conn, flat_comments)
            print(f"  [{i}/{len(seen_post_ids)}] post {post_id}: {len(flat_comments)} comments")
        except requests.HTTPError as e:
            print(f"  [{i}/{len(seen_post_ids)}] post {post_id}: failed ({e})")
        time.sleep(REQUEST_DELAY_SECONDS)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
