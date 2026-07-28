"""
filter_subreddit_streaming.py

Streams a monthly pushshift RC_*.zst comments dump and writes out only
r/nba comments to a .jsonl file, without ever holding the full decompressed
file on disk. Delete the source .zst after this completes successfully.

Usage:
    python filter_subreddit_streaming.py <input.zst> <output.jsonl>

Example:
    python filter_subreddit_streaming.py RC_2024-10.zst nba_2024_10.jsonl
"""
import zstandard as zstd
import json
import sys
import io

def filter_month(input_zst_path, output_jsonl_path, target_subreddit="nba"):
    matched = 0
    total = 0

    with open(input_zst_path, 'rb') as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        stream_reader = dctx.stream_reader(fh)
        text_stream = io.TextIOWrapper(stream_reader, encoding='utf-8', errors='replace')

        with open(output_jsonl_path, 'w', encoding='utf-8') as out:
            for line in text_stream:
                total += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if obj.get("subreddit") == target_subreddit:
                    out.write(json.dumps(obj) + "\n")
                    matched += 1

                if total % 5_000_000 == 0:
                    print(f"  ...{total:,} lines read, {matched:,} matched so far")

    print(f"Done: {matched:,} r/nba comments out of {total:,} total lines")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_subreddit_streaming.py <input.zst> <output.jsonl>")
        sys.exit(1)
    filter_month(sys.argv[1], sys.argv[2])
