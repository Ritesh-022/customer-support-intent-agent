"""Extract a clean AppleSupport working set for intent discovery.

Produces TWO non-overlapping datasets from the full AppleSupport conversation pool:
  - data/processed/apple_support_train.csv  (training pool, 9000 conversations)
  - data/processed/apple_support_golden.csv (golden pool,   1000 conversations)

The two pools share zero conversations. The golden pool is used only for
generate_golden.py and must never be used for training.

Customer-brand pairing uses explicit in_response_to_tweet_id linkage rather
than the previous chronological-adjacency heuristic.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from data_analysis import REQUIRED_COLUMNS, UnionFind, chunks, clean, inbound

BRAND = "AppleSupport"


def references(value: Any) -> list[str]:
    return [item.strip() for item in clean(value).split(",") if item.strip()]


def parse_date(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
    return None if pd.isna(parsed) else parsed


def build_graph(path: Path, chunksize: int):
    nodes: dict[str, int] = {}
    union_find = UnionFind()
    tweet_meta: dict[str, tuple[str, bool]] = {}
    apple_roots: set[int] = set()

    def node(tweet_id: str) -> int:
        if tweet_id not in nodes:
            nodes[tweet_id] = union_find.add()
        return nodes[tweet_id]

    for chunk_number, chunk in enumerate(chunks(path, chunksize), 1):
        missing = set(REQUIRED_COLUMNS).difference(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        for values in chunk.itertuples(index=False, name=None):
            row = dict(zip(chunk.columns, values))
            tweet_id = clean(row["tweet_id"])
            if not tweet_id:
                continue
            author = clean(row["author_id"])
            is_customer = inbound(row["inbound"])
            tweet_meta[tweet_id] = (author, is_customer)
            current = node(tweet_id)
            for ref in references(row["response_tweet_id"]) + references(row["in_response_to_tweet_id"]):
                union_find.union(current, node(ref))
        if chunk_number % 5 == 0:
            print(f"  graph pass: processed {chunk_number * chunksize:,}+ rows")

    for tweet_id, (author, is_customer) in tweet_meta.items():
        if author == BRAND and not is_customer:
            apple_roots.add(union_find.find(nodes[tweet_id]))
    return nodes, union_find, tweet_meta, apple_roots


def split_roots(apple_roots: set[int], train_size: int, golden_size: int,
                seed: int = 42) -> tuple[set[int], set[int]]:
    """Deterministic random split into non-overlapping train and golden pools."""
    ordered = sorted(apple_roots)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    train = set(ordered[:train_size])
    golden = set(ordered[train_size:train_size + golden_size])
    return train, golden


def collect_conversations(path: Path, chunksize: int, nodes: dict[str, int],
                          union_find: UnionFind, selected_roots: set[int]):
    conversations: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for chunk_number, chunk in enumerate(chunks(path, chunksize), 1):
        for values in chunk.itertuples(index=False, name=None):
            row = dict(zip(chunk.columns, values))
            tweet_id = clean(row["tweet_id"])
            if tweet_id not in nodes:
                continue
            root = union_find.find(nodes[tweet_id])
            if root not in selected_roots:
                continue
            conversations[root].append({
                "tweet_id": tweet_id,
                "author_id": clean(row["author_id"]),
                "inbound": inbound(row["inbound"]),
                "created_at": clean(row["created_at"]),
                "timestamp": parse_date(row["created_at"]),
                "text": clean(row["text"]),
                "response_tweet_id": references(row["response_tweet_id"]),
                "in_response_to_tweet_id": references(row["in_response_to_tweet_id"]),
            })
        if chunk_number % 5 == 0:
            print(f"  extraction pass: processed {chunk_number * chunksize:,}+ rows")
    return conversations


def conversation_examples(conversations: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Build customer→brand pairs using explicit in_response_to_tweet_id linkage.

    For each brand tweet, we look up the tweet it explicitly replies to via
    in_response_to_tweet_id. If that tweet is a customer message in the same
    conversation, we form a pair. Falls back to chronological adjacency only
    when the explicit link is absent or points outside the conversation.
    """
    examples: list[dict[str, Any]] = []
    for conversation_id, messages in conversations.items():
        messages.sort(key=lambda m: (m["timestamp"] is None, m["timestamp"], m["tweet_id"]))
        tweet_by_id: dict[str, dict[str, Any]] = {m["tweet_id"]: m for m in messages}
        history: list[str] = []

        for idx, message in enumerate(messages):
            role = "customer" if message["inbound"] else "brand"
            history.append(f"{role}: {message['text']}")

            if message["inbound"] or message["author_id"] != BRAND:
                continue

            # --- explicit linkage (preferred) ---
            customer: dict[str, Any] | None = None
            for ref_id in message["in_response_to_tweet_id"]:
                candidate = tweet_by_id.get(ref_id)
                if candidate and candidate["inbound"]:
                    customer = candidate
                    break

            # --- chronological fallback ---
            if customer is None and idx > 0 and messages[idx - 1]["inbound"]:
                customer = messages[idx - 1]

            if customer is None or not customer["text"] or not message["text"]:
                continue

            examples.append({
                "conversation_id": conversation_id,
                "customer_tweet_id": customer["tweet_id"],
                "brand_tweet_id": message["tweet_id"],
                "timestamp": message["timestamp"].isoformat() if message["timestamp"] else "",
                "customer_message": customer["text"],
                "brand_response": message["text"],
                "conversation_history": "\n".join(history[:-1]),
                "conversation_turns": len(messages),
                "is_multi_brand_candidate": int(any(
                    m["author_id"] not in {BRAND, customer["author_id"]} and not m["inbound"]
                    for m in messages
                )),
            })
    return examples


def write_pool(examples: list[dict[str, Any]], output: Path, metadata: dict) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(examples).to_csv(output, index=False)
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"  Wrote {len(examples):,} examples -> {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract AppleSupport conversations into train + golden pools")
    parser.add_argument("--input", type=Path, default=Path("dataset/twcs/twcs.csv"))
    parser.add_argument("--train-output", type=Path, default=Path("data/processed/apple_support_train.csv"))
    parser.add_argument("--golden-output", type=Path, default=Path("data/processed/apple_support_golden.csv"))
    # Keep legacy --output for backward compat (points to train output)
    parser.add_argument("--output", type=Path, default=None,
                        help="Alias for --train-output (backward compatibility)")
    parser.add_argument("--train-conversations", type=int, default=9_000)
    parser.add_argument("--golden-conversations", type=int, default=1_000)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output:
        args.train_output = args.output

    total_needed = args.train_conversations + args.golden_conversations
    if total_needed <= 0 or args.chunksize <= 0:
        raise ValueError("Conversation counts and chunksize must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    print(f"[1/4] Building conversation graph from {args.input}")
    nodes, union_find, tweet_meta, apple_roots = build_graph(args.input, args.chunksize)
    print(f"[2/4] Found {len(apple_roots):,} AppleSupport conversations; "
          f"splitting into train={args.train_conversations:,} + golden={args.golden_conversations:,}")

    if len(apple_roots) < total_needed:
        raise ValueError(
            f"Only {len(apple_roots):,} conversations available; "
            f"requested {total_needed:,}. Reduce --train-conversations or --golden-conversations."
        )

    train_roots, golden_roots = split_roots(
        apple_roots, args.train_conversations, args.golden_conversations, seed=args.seed
    )
    assert not train_roots & golden_roots, "BUG: train and golden pools overlap"

    all_selected = train_roots | golden_roots
    print(f"[3/4] Collecting conversations (train={len(train_roots):,}, golden={len(golden_roots):,})")
    all_conversations = collect_conversations(args.input, args.chunksize, nodes, union_find, all_selected)

    train_convos = {k: v for k, v in all_conversations.items() if k in train_roots}
    golden_convos = {k: v for k, v in all_conversations.items() if k in golden_roots}

    print(f"[4/4] Building examples and writing outputs")
    train_examples = conversation_examples(train_convos)
    golden_examples = conversation_examples(golden_convos)

    base_meta = {"brand": BRAND, "input": str(args.input),
                 "seed": args.seed,
                 "selection": "deterministic random sample; train and golden pools are non-overlapping"}

    write_pool(train_examples, args.train_output,
               {**base_meta, "pool": "train", "conversations": len(train_convos),
                "examples": len(train_examples), "output": str(args.train_output)})
    write_pool(golden_examples, args.golden_output,
               {**base_meta, "pool": "golden", "conversations": len(golden_convos),
                "examples": len(golden_examples), "output": str(args.golden_output)})

    print(f"\nTrain pool : {len(train_examples):,} examples from {len(train_convos):,} conversations")
    print(f"Golden pool: {len(golden_examples):,} examples from {len(golden_convos):,} conversations")
    print("Zero conversation overlap between pools.")


if __name__ == "__main__":
    main()
