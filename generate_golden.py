"""Generate a stratified golden-set template for manual annotation.

Samples 200 rows from the GOLDEN POOL (apple_support_golden.csv), which is
completely separate from the training pool. Zero conversation overlap.

Sampling strategy (60/20/20):
  - 60% representative random across intents (hard-capped per intent)
  - 20% low rule_confidence rows from meaningful intents only
  - 20% edge cases (short messages or multi-turn) from meaningful intents only

The weak_label_hint is stored in a SEPARATE hints file, NOT in golden_set.csv,
so the annotator is not anchored to the machine label.

IMPORTANT: Never use golden_set.csv rows for training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# insufficient_information is capped low — it adds little annotation value
# and dominates the pool because many tweets are bare mentions.
HARD_CAPS: dict[str, int] = {
    "insufficient_information": 10,
}
DEFAULT_CAP = 22   # ~11% of 200 per meaningful intent
MIN_SLOTS = 3
# Intents excluded from hard/edge sampling blocks (low signal)
LOW_SIGNAL = {"insufficient_information"}


def allocate_slots(counts: pd.Series, total: int) -> dict[str, int]:
    """Hard-capped proportional allocation that sums exactly to total."""
    caps = {i: HARD_CAPS.get(i, DEFAULT_CAP) for i in counts.index}
    raw = (counts / counts.sum() * total).round().astype(int)
    for i in raw.index:
        raw[i] = min(raw[i], caps[i])
        if counts[i] >= MIN_SLOTS:
            raw[i] = max(raw[i], MIN_SLOTS)
    while raw.sum() > total:
        raw[raw.idxmax()] -= 1
    while raw.sum() < total:
        eligible = [i for i in raw.index if raw[i] < caps[i] and raw[i] < counts[i]]
        if not eligible:
            break
        raw[eligible[0]] += 1
    return raw.to_dict()


def sample_golden(labeled: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    conf_col = "rule_confidence" if "rule_confidence" in labeled.columns else "intent_confidence"
    df = labeled[labeled["intent"].astype(str).str.strip().ne("")].copy()
    if conf_col in df.columns:
        df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce").fillna(1.0)

    n_repr = int(n * 0.60)
    n_hard = int(n * 0.20)
    n_edge = n - n_repr - n_hard

    used: set = set()
    frames = []

    # 60% — representative, hard-capped
    counts = df["intent"].value_counts()
    slots = allocate_slots(counts, n_repr)
    for intent, k in slots.items():
        pool = df[df["intent"] == intent]
        k = min(k, len(pool), HARD_CAPS.get(intent, DEFAULT_CAP))
        if k > 0:
            s = pool.sample(n=k, random_state=seed)
            frames.append(s)
            used.update(s.index)

    # 20% — hardest (lowest confidence), meaningful intents only
    rem = df[~df.index.isin(used) & ~df["intent"].isin(LOW_SIGNAL)].copy()
    if conf_col in rem.columns:
        rem = rem.sort_values(conf_col)
    hard_n = min(n_hard, len(rem))
    if hard_n > 0:
        s = rem.head(hard_n * 3).sample(n=hard_n, random_state=seed + 1)
        frames.append(s)
        used.update(s.index)

    # 20% — edge cases, meaningful intents only
    rem = df[~df.index.isin(used) & ~df["intent"].isin(LOW_SIGNAL)].copy()
    short = rem["customer_message"].astype(str).str.len() < 40
    multi = rem.get("conversation_turns", pd.Series(0, index=rem.index)).astype(int) > 5
    edge_pool = rem[short | multi] if (short | multi).sum() >= n_edge else rem
    edge_n = min(n_edge, len(edge_pool))
    if edge_n > 0:
        s = edge_pool.sample(n=edge_n, random_state=seed + 2)
        frames.append(s)

    return pd.concat(frames).drop_duplicates(subset=["customer_message"]).head(n).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stratified golden-set template from the golden pool")
    parser.add_argument("--input", type=Path, default=Path("data/processed/apple_support_golden.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/golden_set.csv"))
    parser.add_argument("--hints-output", type=Path, default=Path("data/golden_set_hints.csv"))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} already exists. Use --overwrite only if you haven't started annotating.")
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run extract_apple_support.py first.")

    golden_pool = pd.read_csv(args.input).fillna("")
    if "customer_message" not in golden_pool.columns:
        raise ValueError("Golden pool CSV must contain customer_message")

    # Label the golden pool using assign_intent (weak labels for sampling only)
    from label_data import assign_intent
    assignments = [assign_intent(str(t), 0.60) for t in golden_pool["customer_message"]]
    golden_pool["intent"] = [a[0] for a in assignments]
    golden_pool["rule_confidence"] = [a[1] for a in assignments]

    sample = sample_golden(golden_pool, args.n, args.seed)

    # golden_set.csv — NO weak label hint shown to annotator
    keep = [c for c in ("conversation_id", "customer_tweet_id", "timestamp",
                        "customer_message", "brand_response", "conversation_history",
                        "conversation_turns") if c in sample.columns]
    golden = sample[keep].copy()
    golden.insert(0, "id", range(1, len(golden) + 1))
    for col in ("intent", "should_escalate", "expected_resolution", "difficulty", "evidence_conversation_id"):
        golden[col] = ""

    args.output.parent.mkdir(parents=True, exist_ok=True)
    golden.to_csv(args.output, index=False)

    # hints file — separate, for post-annotation comparison only
    hints = pd.DataFrame({"id": range(1, len(sample) + 1)})
    if "customer_tweet_id" in sample.columns:
        hints["customer_tweet_id"] = sample["customer_tweet_id"].values
    hints["weak_label_hint"] = sample["intent"].values
    hints["weak_rule_confidence"] = sample["rule_confidence"].values
    args.hints_output.parent.mkdir(parents=True, exist_ok=True)
    hints.to_csv(args.hints_output, index=False)

    print(f"Generated {len(golden)}-row golden-set template at {args.output}")
    print(f"Hints saved separately at {args.hints_output} (open only after annotating)")
    print(f"\nIntent distribution (weak labels, sampling reference only):")
    for intent, count in sample["intent"].value_counts().items():
        print(f"  {intent}: {count}")
    print(f"\nSampling: 60% representative (capped), 20% low-confidence, 20% edge cases")
    print(f"\nNext steps:")
    print(f"  1. python annotate_golden.py")
    print(f"  2. Manually review and correct data/golden_set.csv")
    print(f"  3. NEVER use golden_set.csv rows for training")


if __name__ == "__main__":
    main()
