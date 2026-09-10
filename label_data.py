"""Create constrained intent labels for the AppleSupport working dataset.

This is weak supervision, not human ground truth. The taxonomy is frozen in
TAXONOMY; the script may only assign one of those intents. Low-confidence and
ambiguous rows are written to a review file and should not be treated as
final evaluation labels.

Column `rule_confidence` is a rule-score ratio, NOT a calibrated probability.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Taxonomy changes vs previous version:
#   - Removed `technical_support` (was a garbage bucket; generic words like
#     "help/problem/issue" now fall through to insufficient_information).
#   - Renamed `other` → `insufficient_information` (more descriptive).
#   - Tightened `device_issue` to require hardware-specific terms only.
#   - `app_software_issue` now takes priority over `device_issue` when both
#     match, because software intent is more actionable.
#   - `complaint` patterns tightened to avoid false positives.
TAXONOMY = {
    "account_access": {
        "definition": "Apple ID, iCloud account login/lockout, password reset, sign-in verification. NOT iCloud storage (that is subscription).",
        # icloud alone is ambiguous (could be storage=subscription or sync=app_software)
        # so we require it alongside account-specific terms
        "patterns": [r"apple\s*id", r"icloud.*account", r"account.*icloud", r"password", r"log\s*in", r"sign\s*in", r"locked out", r"verification", r"two.factor", r"2fa"],
    },
    "billing_payment": {
        "definition": "Charges, invoices, payment methods, billing errors, or unexpected payments.",
        "patterns": [r"charg(?:e|ed|ing)", r"payment", r"billing", r"bill\b", r"credit card", r"debit card", r"invoice", r"paid twice", r"double charge"],
    },
    "device_issue": {
        "definition": "Physical hardware only: battery drain, cracked/broken screen, device won't turn on, physical damage. NOT software crashes.",
        "patterns": [r"battery", r"screen\b", r"broken", r"won.?t turn on", r"hardware", r"physical damage", r"cracked", r"dead\b", r"not charging"],
    },
    "app_software_issue": {
        "definition": "iOS/macOS/app bugs, updates, crashes, freezes, slow performance, software errors. Device names alone are NOT evidence of software intent.",
        # Device names (iphone/ipad/mac) intentionally excluded — they appear in all intents
        # and caused device_issue vs app_software_issue confusion when used as signals.
        "patterns": [r"ios\b", r"macos\b", r"update\b", r"app\b", r"itunes", r"imessage", r"facetime", r"crash", r"freez", r"software", r"error\b", r"slow\b", r"lag\b", r"bug\b", r"glitch"],
    },
    "subscription": {
        "definition": "Apple Music, Apple TV+, iCloud storage subscription, renewal, cancellation of a recurring plan.",
        "patterns": [r"subscription", r"subscrib", r"renewal", r"renew\b", r"cancel.*plan", r"apple music", r"apple tv\b", r"icloud storage", r"icloud.*plan", r"storage plan"],
    },
    "refund_return": {
        "definition": "Refund, return, reimbursement, cancellation of a purchase, or money back request.",
        "patterns": [r"refund", r"return\b", r"money back", r"reimburse", r"cancel.*order", r"charged.*wrong"],
    },
    "order_purchase": {
        "definition": "Buying, ordering, delivery, shipping, product availability, or purchase status.",
        "patterns": [r"order\b", r"purchase", r"buy\b", r"bought", r"shipping", r"delivery", r"deliver\b", r"tracking", r"available\b"],
    },
    "connectivity_issue": {
        "definition": "Wi-Fi, Bluetooth, cellular, mobile data, network, or connection problem.",
        "patterns": [r"wi[ -]?fi", r"bluetooth", r"internet\b", r"network\b", r"cellular", r"mobile data", r"signal\b", r"connect\b", r"connection"],
    },
    "general_information": {
        "definition": "General question, how-to request, product information, or policy information.",
        "patterns": [r"how do i\b", r"how can i\b", r"where can i\b", r"what is\b", r"can i\b", r"information\b", r"question\b"],
    },
    "complaint": {
        "definition": "Explicit complaint or strongly negative feedback without a clearer actionable issue.",
        "patterns": [r"complaint\b", r"terrible\b", r"worst\b", r"disappoint", r"unacceptable\b", r"poor service", r"awful\b"],
    },
    "insufficient_information": {
        "definition": "Message that does not match another frozen intent with enough evidence (e.g. bare mentions, URLs only, very short messages).",
        "patterns": [],
    },
}

# Priority: more specific / higher-stakes intents win ties.
# app_software_issue is listed BEFORE device_issue so software intent wins
# when both device name and software terms appear.
PRIORITY = [
    "account_access", "billing_payment", "refund_return", "subscription",
    "connectivity_issue", "order_purchase", "app_software_issue", "device_issue",
    "complaint", "general_information", "insufficient_information",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def assign_intent(text: str, min_confidence: float) -> tuple[str, float, bool, str]:
    normalized = normalize(text)
    scores: dict[str, int] = {}
    matches: dict[str, list[str]] = {}
    for intent, definition in TAXONOMY.items():
        matches[intent] = []
        for pattern in definition["patterns"]:
            if re.search(pattern, normalized, flags=re.IGNORECASE):
                scores[intent] = scores.get(intent, 0) + 1
                matches[intent].append(pattern)
    ranked = sorted(scores.items(), key=lambda item: (-item[1], PRIORITY.index(item[0])))
    if not ranked:
        return "insufficient_information", 0.0, True, ""
    best_intent, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    # rule_confidence is a score ratio, NOT a calibrated probability.
    rule_confidence = min(0.99, best_score / max(best_score + second_score, 1))
    needs_review = rule_confidence < min_confidence or best_score == 1 or best_score == second_score
    return best_intent, round(rule_confidence, 4), needs_review, ";".join(matches[best_intent])


def write_discovery_report(data: pd.DataFrame, assignments: list[tuple[str, float, bool, str]], path: Path) -> None:
    texts = data["customer_message"].astype(str).tolist()
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_features=5000)
    matrix = vectorizer.fit_transform(texts)
    terms = vectorizer.get_feature_names_out()
    labels = [item[0] for item in assignments]
    report: dict[str, object] = {
        "status": "proposed_taxonomy_review",
        "warning": "Counts and terms are exploratory weak-supervision evidence, not human-validated intent labels.",
        "rows": len(data),
        "proposed_taxonomy": TAXONOMY,
        "intents": {},
    }
    intent_report: dict[str, object] = {}
    for intent in PRIORITY:
        indexes = [index for index, label in enumerate(labels) if label == intent]
        top_terms: list[str] = []
        if indexes:
            scores = matrix[indexes].mean(axis=0).A1
            top_indexes = scores.argsort()[::-1][:15]
            top_terms = [str(terms[index]) for index in top_indexes if scores[index] > 0]
        intent_report[intent] = {
            "weak_label_count": len(indexes),
            "top_terms": top_terms,
            "sample_messages": [texts[index] for index in indexes[:5]],
            "definition": TAXONOMY[intent]["definition"],
        }
    report["intents"] = intent_report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign constrained AppleSupport intents (weak supervision)")
    parser.add_argument("--input", type=Path, default=Path("data/processed/apple_support_train.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/labeled_training.csv"))
    parser.add_argument("--review-output", type=Path, default=Path("data/label_review.csv"))
    parser.add_argument("--high-confidence-output", type=Path, default=Path("data/labeled_high_confidence.csv"),
                        help="Rows with rule_confidence >= --high-confidence-threshold (safer training subset)")
    parser.add_argument("--taxonomy-output", type=Path, default=Path("data/taxonomy.json"))
    parser.add_argument("--discovery-output", type=Path, default=Path("reports/intent_discovery.json"))
    parser.add_argument("--min-confidence", type=float, default=0.60)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.90,
                        help="rule_confidence threshold for the high-confidence training subset")
    args = parser.parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    data = pd.read_csv(args.input).fillna("")
    if "customer_message" not in data.columns:
        raise ValueError("Input must contain customer_message")
    assignments = [assign_intent(str(text), args.min_confidence) for text in data["customer_message"]]
    write_discovery_report(data, assignments, args.discovery_output)
    data["intent"] = [item[0] for item in assignments]
    # Renamed from intent_confidence: this is a rule-score ratio, not a calibrated probability.
    data["rule_confidence"] = [item[1] for item in assignments]
    data["needs_manual_review"] = [item[2] for item in assignments]
    data["matched_patterns"] = [item[3] for item in assignments]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    data[data["needs_manual_review"]].to_csv(args.review_output, index=False)
    high_conf = data[data["rule_confidence"] >= args.high_confidence_threshold]
    high_conf.to_csv(args.high_confidence_output, index=False)
    args.taxonomy_output.write_text(json.dumps(TAXONOMY, indent=2), encoding="utf-8")
    print(f"Labeled {len(data):,} rows using {len(TAXONOMY)} frozen intents")
    print(f"Manual-review rows (rule_confidence < {args.min_confidence} or ambiguous): {int(data['needs_manual_review'].sum()):,}")
    print(f"High-confidence rows (rule_confidence >= {args.high_confidence_threshold}): {len(high_conf):,}")
    print(f"Wrote all weak labels:        {args.output}")
    print(f"Wrote review queue:           {args.review_output}")
    print(f"Wrote high-confidence subset: {args.high_confidence_output}")
    print(f"Wrote taxonomy:               {args.taxonomy_output}")
    print(f"Wrote discovery report:       {args.discovery_output}")
    print("WARNING: These are weak labels generated by keyword rules. Do NOT use as ground truth.")
    print("         Use the golden set (generate_golden.py) for independent evaluation.")


if __name__ == "__main__":
    main()
