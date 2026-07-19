"""
test_hf_inference.py

Quick sanity check: scores just 3 comments using the hf-inference provider,
to confirm the model is available and billing actually works there, before
committing to a full 150-row (or 1,550-row) run.

Run:
    python scripts/test_hf_inference.py
"""

import csv
import json
import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient

INPUT_PATH = "docs/validation_sample.csv"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROVIDER = "hf-inference"
TEST_ROWS = 3

SYSTEM_PROMPT = """You analyze Reddit comments from r/nba for sentiment specifically about \
the basketball player Stephen Curry. Many comments mention Curry only in passing \
while actually expressing sentiment about someone or something else (e.g. \
comparing him to another player, or using him as a reference point). Your job \
is to identify whether the comment expresses genuine sentiment TOWARD Curry \
himself, and if so, what that sentiment is.

Respond with ONLY a JSON object, no other text, in this exact format:
{"subject": "about_curry" | "incidental" | "comparative" | "unclear", \
"sentiment": "positive" | "negative" | "neutral", "score": <float from -1.0 to 1.0>}
"""

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN not found in .env.")


def main():
    print(f"Testing model={MODEL} provider={PROVIDER}\n")

    with open(INPUT_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [next(reader) for _ in range(TEST_ROWS)]

    client = InferenceClient(api_key=HF_TOKEN, provider=PROVIDER)

    for i, row in enumerate(rows, start=1):
        body = row["body"]
        print(f"--- Row {i}: {row['comment_id']} ---")
        print(f"Body: {body[:100]}{'...' if len(body) > 100 else ''}")
        try:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": body},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            raw = completion.choices[0].message.content.strip()
            print(f"Raw response: {raw}")
            try:
                parsed = json.loads(raw.strip("`").removeprefix("json").strip())
                print(f"Parsed OK: {parsed}")
            except json.JSONDecodeError as e:
                print(f"JSON parse failed: {e}")
        except Exception as e:
            print(f"REQUEST FAILED: {e}")
        print()

    print("Test complete. If all 3 rows succeeded, hf-inference works -- safe to run the full script.")


if __name__ == "__main__":
    main()
