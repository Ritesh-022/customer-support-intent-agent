"""Chunked full audit for the Customer Support on Twitter dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = (
    "tweet_id", "author_id", "inbound", "created_at", "text",
    "response_tweet_id", "in_response_to_tweet_id",
)
TRUE_VALUES = {"true", "1", "yes", "y", "t"}
ID_PATTERN = re.compile(r"^\d+$")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"(?<!\w)@\w+")
HASHTAG_PATTERN = re.compile(r"(?<!\w)#\w+")
EMOJI_PATTERN = re.compile("[\\U0001F300-\\U0001FAFF\\U00002600-\\U000027BF]")
LOW_INFORMATION = {"thanks", "thank you", "thx", "okay", "ok", "hi", "hello", "?", "any update?", "please help", "help"}


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []

    def add(self) -> int:
        node = len(self.parent)
        self.parent.append(node)
        self.rank.append(0)
        return node

    def find(self, node: int) -> int:
        root = node
        while root != self.parent[root]:
            root = self.parent[root]
        while node != root:
            next_node = self.parent[node]
            self.parent[node] = root
            node = next_node
        return root

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


@dataclass
class BrandStats:
    brand_tweets: int = 0
    customer_messages: int = 0
    response_pairs: int = 0
    customers: set[str] = field(default_factory=set)
    conversations: set[int] = field(default_factory=set)
    text: Counter = field(default_factory=Counter)


def clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def inbound(value: Any) -> bool:
    return clean(value).lower() in TRUE_VALUES


def chunks(path: Path, size: int):
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False, chunksize=size, on_bad_lines="warn")


def metrics(text: str) -> tuple[int, int, Counter]:
    flags = Counter({
        "urls": len(URL_PATTERN.findall(text)),
        "mentions": len(MENTION_PATTERN.findall(text)),
        "hashtags": len(HASHTAG_PATTERN.findall(text)),
        "emojis": len(EMOJI_PATTERN.findall(text)),
        "questions": int("?" in text),
        "low_information": int(re.sub(r"\s+", " ", text.lower()).strip() in LOW_INFORMATION),
    })
    length = len(text)
    flags["under_10"] = int(length < 10)
    flags["10_25"] = int(10 <= length <= 25)
    flags["26_50"] = int(26 <= length <= 50)
    flags["51_100"] = int(51 <= length <= 100)
    flags["over_100"] = int(length > 100)
    return length, len(text.split()), flags


def fingerprint(values: tuple[Any, ...]) -> str:
    raw = "\x1f".join(clean(value) for value in values)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()


def first_pass(path: Path, size: int):
    tweet_ids: set[str] = set()
    references: set[str] = set()
    nodes: dict[str, int] = {}
    meta: dict[str, tuple[bool, pd.Timestamp | None, str]] = {}
    tweet_brands: dict[str, str] = {}
    authors: set[str] = set()
    hashes: set[str] = set()
    duplicate_ids: set[str] = set()
    uf = UnionFind()
    brands: dict[str, BrandStats] = {}
    quality = Counter()
    date_min = date_max = None

    def node(tweet_id: str) -> int:
        if tweet_id not in nodes:
            nodes[tweet_id] = uf.add()
        return nodes[tweet_id]

    for chunk in chunks(path, size):
        missing = set(REQUIRED_COLUMNS).difference(chunk.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        parsed = pd.to_datetime(chunk["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
        quality["rows_read"] += len(chunk)
        for index, values in enumerate(chunk.itertuples(index=False, name=None)):
            row = dict(zip(chunk.columns, values))
            tweet_id = clean(row["tweet_id"])
            author = clean(row["author_id"])
            parent = clean(row["in_response_to_tweet_id"])
            response = clean(row["response_tweet_id"])
            text = clean(row["text"])
            is_customer = inbound(row["inbound"])
            date = parsed.iloc[index]
            date = None if pd.isna(date) else date
            if tweet_id in tweet_ids:
                duplicate_ids.add(tweet_id)
            if tweet_id:
                tweet_ids.add(tweet_id)
                nodes.setdefault(tweet_id, uf.add())
                meta[tweet_id] = (is_customer, date, author)
            for reference in (parent, response):
                if reference:
                    references.add(reference)
                    if tweet_id:
                        uf.union(node(tweet_id), node(reference))
            if author:
                authors.add(author)
            quality["invalid_or_empty_tweet_ids"] += int(not tweet_id or not ID_PATTERN.fullmatch(tweet_id))
            quality["empty_messages"] += int(not text)
            quality["messages_over_280_chars"] += int(len(text) > 280)
            quality["customer_tweets" if is_customer else "brand_tweets"] += 1
            if is_customer and text:
                length, words, flags = metrics(text)
                quality["customer_characters"] += length
                quality["customer_words"] += words
                for key, value in flags.items():
                    quality[f"text_{key}"] += value
            if not is_customer:
                stats = brands.setdefault(author, BrandStats())
                stats.brand_tweets += 1
                if tweet_id:
                    tweet_brands[tweet_id] = author
            if date is None:
                quality["invalid_dates"] += int(bool(clean(row["created_at"])))
            else:
                date_min = date if date_min is None else min(date_min, date)
                date_max = date if date_max is None else max(date_max, date)
            row_hash = fingerprint(values)
            quality["duplicate_rows"] += int(row_hash in hashes)
            hashes.add(row_hash)

    broken = references - tweet_ids
    quality.update({
        "unique_tweets": len(tweet_ids), "duplicate_tweet_ids": len(duplicate_ids),
        "unique_authors": len(authors), "unique_references": len(references),
        "broken_references": len(broken), "conversation_nodes": len(nodes),
        "conversation_components": len({uf.find(value) for value in nodes.values()}),
    })
    return nodes, meta, tweet_brands, uf, brands, quality, date_min, date_max, broken


def second_pass(path: Path, size: int, nodes: dict[str, int], meta: dict[str, tuple[bool, pd.Timestamp | None, str]],
                tweet_brands: dict[str, str], uf: UnionFind, brands: dict[str, BrandStats]):
    threads: dict[int, dict[str, Any]] = defaultdict(lambda: {"tweets": 0, "customer": 0, "brand": 0, "brands": set(), "last_date": None, "last_inbound": None})
    directions = Counter()
    response_seconds: list[float] = []
    for chunk in chunks(path, size):
        parsed = pd.to_datetime(chunk["created_at"], format="%a %b %d %H:%M:%S %z %Y", errors="coerce", utc=True)
        for index, values in enumerate(chunk.itertuples(index=False, name=None)):
            row = dict(zip(chunk.columns, values))
            tweet_id = clean(row["tweet_id"])
            if tweet_id not in nodes:
                continue
            is_customer = inbound(row["inbound"])
            date = parsed.iloc[index]
            date = None if pd.isna(date) else date
            thread = threads[uf.find(nodes[tweet_id])]
            thread["tweets"] += 1
            thread["customer" if is_customer else "brand"] += 1
            if date is not None and (thread["last_date"] is None or date >= thread["last_date"]):
                thread["last_date"], thread["last_inbound"] = date, is_customer
            parent = clean(row["in_response_to_tweet_id"])
            response = clean(row["response_tweet_id"])
            target = parent if parent in meta else response if response in meta else ""
            brand = clean(row["author_id"]) if not is_customer else ""
            if target:
                target_customer, target_date, target_author = meta[target]
                if is_customer == target_customer:
                    directions["customer_to_customer" if is_customer else "brand_to_brand"] += 1
                elif is_customer:
                    directions["customer_to_brand"] += 1
                    brand = tweet_brands.get(target, "")
                else:
                    directions["brand_to_customer"] += 1
                    if target_customer and target_date is not None and date is not None and date >= target_date:
                        response_seconds.append((date - target_date).total_seconds())
                        if brand in brands:
                            brands[brand].response_pairs += 1
            if brand:
                thread["brands"].add(brand)
                if is_customer:
                    stats = brands.setdefault(brand, BrandStats())
                    stats.customer_messages += 1
                    if clean(row["author_id"]):
                        stats.customers.add(clean(row["author_id"]))
                    text_length, text_words, text_flags = metrics(clean(row["text"]))
                    text_flags["characters"] = text_length
                    text_flags["words"] = text_words
                    stats.text.update(text_flags)

    rows = []
    for conversation_id, thread in threads.items():
        names = sorted(thread["brands"])
        outcome = "customer_response" if thread["last_inbound"] else "brand_response" if thread["last_inbound"] is False else "no_further_response"
        row = {"conversation_id": conversation_id, "tweets": thread["tweets"], "customer_tweets": thread["customer"],
               "brand_tweets": thread["brand"], "brands": ",".join(names), "brand_count": len(names),
               "multi_brand": len(names) > 1, "outcome": outcome}
        rows.append(row)
        for brand in names:
            brands[brand].conversations.add(conversation_id)
    return rows, directions, response_seconds


def stats(values: list[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "median": None, "p25": None, "p75": None, "p90": None, "p95": None, "max": None}
    series = pd.Series(values)
    return {"count": len(values), "median": round(float(series.quantile(.5)), 2), "p25": round(float(series.quantile(.25)), 2),
            "p75": round(float(series.quantile(.75)), 2), "p90": round(float(series.quantile(.9)), 2), "p95": round(float(series.quantile(.95)), 2), "max": max(values)}


def write_reports(output: Path, input_path: Path, quality: Counter, date_min: pd.Timestamp | None, date_max: pd.Timestamp | None,
                  brands: dict[str, BrandStats], conversations: list[dict[str, Any]], directions: Counter,
                  response_seconds: list[float], broken: set[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    turns = [row["tweets"] for row in conversations]
    brand_rows = []
    for name, brand in brands.items():
        lengths = [row["tweets"] for row in conversations if name in row["brands"].split(",")]
        brand_rows.append({"brand": name, "brand_tweets": brand.brand_tweets, "customer_messages": brand.customer_messages,
                           "response_pairs": brand.response_pairs, "conversations": len(brand.conversations),
                           "unique_customers": len(brand.customers), "avg_turns": round(sum(lengths) / len(lengths), 2) if lengths else 0,
                   "median_turns": stats(lengths)["median"],
                   "avg_customer_chars": round(brand.text["characters"] / brand.customer_messages, 2) if brand.customer_messages else 0,
                   "avg_customer_words": round(brand.text["words"] / brand.customer_messages, 2) if brand.customer_messages else 0,
                   "text_urls": brand.text["urls"], "text_mentions": brand.text["mentions"],
                   "text_hashtags": brand.text["hashtags"], "text_emojis": brand.text["emojis"],
                   "text_questions": brand.text["questions"], "text_low_information": brand.text["low_information"]})
    brand_rows.sort(key=lambda row: (row["customer_messages"], row["response_pairs"]), reverse=True)
    total_agent = sum(row["brand_tweets"] for row in brand_rows)
    for rank, row in enumerate(brand_rows, 1):
        row["candidate_rank"] = rank
        row["agent_tweet_share"] = round(row["brand_tweets"] / total_agent, 6) if total_agent else 0
    pd.DataFrame(brand_rows).to_csv(output / "candidate_brands.csv", index=False)
    pd.DataFrame(conversations).sort_values("tweets", ascending=False).to_csv(output / "conversation_summary.csv", index=False)
    directions = Counter({key: directions[key] for key in ("customer_to_brand", "brand_to_customer", "customer_to_customer", "brand_to_brand")})
    response_distribution = stats(response_seconds)
    response_minutes = {key: (round(value / 60, 2) if key != "count" and isinstance(value, (int, float)) else value)
                        for key, value in response_distribution.items()}
    response = {"direction_counts": dict(directions), "response_time_seconds": response_distribution,
                "response_time_minutes": response_minutes}
    (output / "response_analysis.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    text = {key.removeprefix("text_"): value for key, value in quality.items() if key.startswith("text_")}
    (output / "customer_text_analysis.json").write_text(json.dumps(text, indent=2), encoding="utf-8")
    multi = sum(row["multi_brand"] for row in conversations)
    overview = {"input_file": str(input_path), "date_range": {"min": date_min.isoformat() if date_min is not None else None, "max": date_max.isoformat() if date_max is not None else None},
                "quality": dict(quality), "broken_reference_count": len(broken), "single_brand_conversations": len(conversations) - multi,
                "multi_brand_conversations": multi, "conversation_turns": stats(turns), "response_analysis": response, "brand_count": len(brand_rows),
                "top_5_agent_tweet_share": round(sum(row["brand_tweets"] for row in brand_rows[:5]) / total_agent, 6) if total_agent else 0,
                "top_10_agent_tweet_share": round(sum(row["brand_tweets"] for row in brand_rows[:10]) / total_agent, 6) if total_agent else 0}
    (output / "overview.json").write_text(json.dumps(overview, indent=2), encoding="utf-8")
    (output / "broken_references.txt").write_text("\n".join(sorted(broken)), encoding="utf-8")
    buckets = Counter("1" if n == 1 else "2" if n == 2 else "3" if n == 3 else "4" if n == 4 else "5-10" if n <= 10 else "10+" for n in turns)
    md = ["# TWCS Dataset Analysis", "", f"Input: `{input_path}`", "", "## Dataset Health", "", "| Metric | Value |", "|---|---:|"]
    health = [("Rows", quality["rows_read"]), ("Unique tweets", quality["unique_tweets"]), ("Duplicate rows", quality["duplicate_rows"]), ("Duplicate IDs", quality["duplicate_tweet_ids"]),
              ("Unique authors", quality["unique_authors"]), ("Customer tweets", quality["customer_tweets"]), ("Brand tweets", quality["brand_tweets"]), ("Empty messages", quality["empty_messages"]),
              ("Invalid dates", quality["invalid_dates"]), ("Unique references", quality["unique_references"]), ("Broken references", len(broken)), ("Date range", f"{date_min} to {date_max}")]
    md += [f"| {key} | {value} |" for key, value in health] + ["", "## Response Analysis", "", "| Direction | Count |", "|---|---:|"]
    md += [f"| {key} | {value} |" for key, value in directions.items()] + ["", "| Response-time statistic | Seconds | Minutes |", "|---|---:|---:|"]
    md += [f"| {key} | {value} | {round(value / 60, 2) if isinstance(value, (int, float)) else value} |" for key, value in stats(response_seconds).items()]
    md += ["", "## Conversation Statistics", "", f"Single-brand conversations: {len(conversations) - multi}", f"Multi-brand conversations: {multi}", "", "| Statistic | Turns |", "|---|---:|"]
    md += [f"| {key} | {value} |" for key, value in stats(turns).items()] + ["", "| Bucket | Count |", "|---|---:|"]
    md += [f"| {key} | {buckets[key]} |" for key in ("1", "2", "3", "4", "5-10", "10+")]
    md += ["", "## Customer Message Quality", "", "| Metric | Count |", "|---|---:|"] + [f"| {key.removeprefix('text_')} | {value} |" for key, value in sorted(quality.items()) if key.startswith("text_")]
    md += ["", "## Candidate Brands", "", "Ranking is for manual inspection; no arbitrary score selects a target brand.", "", "| Rank | Brand | Customer messages | Responses | Conversations | Customers | Agent tweet share |", "|---:|---|---:|---:|---:|---:|---:|"]
    md += [f"| {row['candidate_rank']} | {row['brand']} | {row['customer_messages']} | {row['response_pairs']} | {row['conversations']} | {row['unique_customers']} | {row['agent_tweet_share']:.2%} |" for row in brand_rows[:20]]
    md += ["", f"Top 5 brands contain {overview['top_5_agent_tweet_share']:.2%} of agent tweets.", f"Top 10 brands contain {overview['top_10_agent_tweet_share']:.2%} of agent tweets.", "", "## Risks", "", "- Broken references are computed after all tweet IDs are collected, so CSV order cannot create false positives.", "- The final observed direction is not proof that an issue was resolved.", "- Exclude or separately handle multi-brand conversations for target-brand training and evaluation.", ""]
    (output / "data_analysis.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunked full TWCS dataset analysis")
    parser.add_argument("--input", type=Path, default=Path("dataset/twcs/twcs.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/twcs"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    print(f"[1/3] Collecting all tweet IDs, references, metadata, and text metrics from {args.input}")
    nodes, meta, tweet_brands, uf, brands, quality, date_min, date_max, broken = first_pass(args.input, args.chunksize)
    print(f"[2/3] Computing response directions, response times, conversations, and outcomes; broken references: {len(broken)}")
    conversations, directions, response_seconds = second_pass(args.input, args.chunksize, nodes, meta, tweet_brands, uf, brands)
    print(f"[3/3] Writing one final report to {args.output}")
    write_reports(args.output, args.input, quality, date_min, date_max, brands, conversations, directions, response_seconds, broken)
    print(f"Done. Reports written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
