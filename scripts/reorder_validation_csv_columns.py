"""
reorder_validation_csv_columns.py

One-off utility: reorders columns in the validation CSVs so the LLM
scoring (primary sentiment series) appears before VADER (secondary),
matching the priority decision documented in the README. Does NOT change
any data values -- purely a column-order change for readability.

Run once:
    python scripts/reorder_validation_csv_columns.py
"""

import pandas as pd

FILES = [
    "docs/validation_sample_with_llm.csv",
    "docs/validation_sample_with_llm_v2.csv",
]

# Desired order: identifiers, then LLM (primary), then manual ground truth,
# then VADER (secondary), then notes.
DESIRED_ORDER = [
    "comment_id",
    "body",
    "llm_subject",
    "llm_sentiment",
    "llm_score",
    "manual_subject_label",
    "vader_sentiment_score",
    "vader_sentiment_label",
    "notes",
]

for path in FILES:
    df = pd.read_csv(path)
    # Only reorder columns that actually exist in this file, preserving any
    # extras at the end just in case the schema drifted between versions.
    existing_ordered = [c for c in DESIRED_ORDER if c in df.columns]
    remaining = [c for c in df.columns if c not in existing_ordered]
    df = df[existing_ordered + remaining]
    df.to_csv(path, index=False)
    print(f"Reordered: {path} -> {list(df.columns)}")

print("\nDone. Review with `git diff --stat` before committing --")
print("expect column-order changes only, no data value changes.")
