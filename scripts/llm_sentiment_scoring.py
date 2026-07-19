"""
llm_sentiment_scoring.py

Scores Curry-related sentiment using an LLM via Hugging Face's Inference
Providers (the current replacement for the old Serverless Inference API),
rather than a lexicon-based approach like VADER.

WHY: VADER (see sentiment_scoring.py) has two known weaknesses observed in
this dataset:
  1. Subject attribution -- it can't tell whether sentiment in a comment is
     actually about Curry vs. someone else Curry is being compared to.
  2. Domain slang -- phrases like "shit on 'em" are positive in sports
     trash-talk but score strongly negative in VADER's general-English
     lexicon.

An LLM, prompted correctly, can potentially handle both by reading the
FULL comment (not a sentence-filtered fragment) and reasoning about who the
sentiment is actually directed at and what the words mean in context.

This script deliberately targets the SAME 150-comment validation sample
you're manually labeling (docs/validation_sample.csv), NOT the full 19k
comment set -- run this first, compare all three (manual / VADER / LLM),
and decide whether it's worth scaling up before spending the time/cost on
the full corpus.

Setup:
    pip install huggingface_hub --break-system-packages
    Add to .env:  HF_TOKEN=hf_your_token_here
    (create a token at https://huggingface.co/settings/tokens with
    "Make calls to Inference Providers" permission)

Run:
    python scripts/llm_sentiment_scoring.py
"""

import csv
import json
import os
import re
import time

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

INPUT_PATH = "docs/validation_sample.csv"
OUTPUT_PATH = "docs/validation_sample_with_llm.csv"

# Configurable -- swap this if the model is slow, unavailable, or you want
# to compare a second model. "auto" lets HF pick the fastest available
# backend provider for this model.
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "auto"

# Be gentle on rate limits -- free-tier Inference Providers access can be
# throttled. Adjust down if you're not hitting limits, up if you are.
REQUEST_DELAY_SECONDS = 1.5
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 5

SYSTEM_PROMPT = """You analyze Reddit comments from r/nba for sentiment specifically about \
the basketball player Stephen Curry. Many comments mention Curry only in passing \
while actually expressing sentiment about someone or something else (e.g. \
comparing him to another player, or using him as a reference point). Your job \
is to identify whether the comment expresses genuine sentiment TOWARD Curry \
himself, and if so, what that sentiment is.

Respond with ONLY a JSON object, no other text, in this exact format:
{"subject": "about_curry" | "incidental" | "comparative" | "unclear", \
"sentiment": "positive" | "negative" | "neutral", "score": <float from -1.0 to 1.0>}

Field meanings:
- "subject": "about_curry" if the sentiment is genuinely directed at Curry; \
"incidental" if Curry is mentioned but the sentiment is about someone/something \
else; "comparative" if it's a direct comparison where sentiment is split between \
Curry and another player; "unclear" if you can't tell or there's no real sentiment.
- "sentiment": the overall sentiment label. If subject is "incidental", this \
should reflect sentiment toward Curry specifically (often "neutral" if none \
is actually expressed toward him).
- "score": -1.0 (very negative) to 1.0 (very positive), 0.0 = neutral. \
This should reflect sentiment toward Curry specifically, not the whole comment.

Respond with the JSON object ONLY. No explanation, no examples, no extra text \
before or after it. Just the single JSON object.
"""

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN not found in .env. See script docstring for setup.")


def get_client():
    return InferenceClient(api_key=HF_TOKEN, provider=PROVIDER)


def score_comment(client, body):
    """Returns (subject, sentiment, score) or (None, None, None) on failure."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": body},
                ],
                temperature=0.1,  # low temp for consistency, not creativity
                max_tokens=150,
            )
            raw = completion.choices[0].message.content.strip()

            # Models sometimes wrap JSON in markdown fences, or ignore
            # instructions and add explanatory text around it, despite the
            # prompt explicitly forbidding this. Try a straight parse first;
            # if that fails, fall back to extracting the first {...} block.
            cleaned = raw.strip("`").removeprefix("json").strip()
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if not match:
                    raise
                parsed = json.loads(match.group(0))

            return (
                parsed.get("subject"),
                parsed.get("sentiment"),
                parsed.get("score"),
            )
        except json.JSONDecodeError as e:
            last_error = f"JSON parse failed on response: {raw!r} ({e})"
        except Exception as e:
            last_error = str(e)

        if attempt < MAX_RETRIES:
            print(f"    attempt {attempt} failed ({last_error}), retrying...")
            time.sleep(RETRY_DELAY_SECONDS)

    print(f"    giving up after {MAX_RETRIES} attempts: {last_error}")
    return None, None, None


def main():
    print(f"Model: {MODEL} (provider={PROVIDER})")
    print(f"Reading from: {INPUT_PATH}\n")

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    # Resume support: if OUTPUT_PATH already exists from a prior partial run,
    # load it and skip re-scoring rows that already succeeded (non-empty
    # llm_subject). This avoids burning credit re-doing successful rows
    # when a run gets interrupted partway (e.g. by a billing cap).
    already_scored = {}
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, newline="", encoding="utf-8") as f:
            prev_reader = csv.DictReader(f)
            for prev_row in prev_reader:
                if prev_row.get("llm_subject", "").strip():
                    already_scored[prev_row["comment_id"]] = prev_row
        print(f"Found existing output with {len(already_scored)} already-scored rows -- will skip those.\n")

    client = get_client()

    new_fields = ["llm_subject", "llm_sentiment", "llm_score"]
    out_fieldnames = fieldnames + [fld for fld in new_fields if fld not in fieldnames]

    scored = 0
    failed = 0
    skipped = 0

    for i, row in enumerate(rows, start=1):
        comment_id = row.get("comment_id")

        if comment_id in already_scored:
            prev = already_scored[comment_id]
            row["llm_subject"] = prev["llm_subject"]
            row["llm_sentiment"] = prev["llm_sentiment"]
            row["llm_score"] = prev["llm_score"]
            skipped += 1
            continue

        body = row.get("body", "")
        print(f"[{i}/{len(rows)}] {comment_id}: scoring...")
        subject, sentiment, score = score_comment(client, body)

        row["llm_subject"] = subject if subject is not None else ""
        row["llm_sentiment"] = sentiment if sentiment is not None else ""
        row["llm_score"] = score if score is not None else ""

        if subject is None:
            failed += 1
        else:
            scored += 1

        time.sleep(REQUEST_DELAY_SECONDS)

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Scored {scored} new, skipped {skipped} already-done, failed {failed}. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()