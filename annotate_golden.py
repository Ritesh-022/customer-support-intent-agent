"""Auto-annotate data/golden_set.csv using a local Ollama model.

For each row the model reads customer_message + conversation_history and fills:
  intent, should_escalate, expected_resolution, difficulty

Runs row-by-row with a retry on parse failure. Saves progress after every row
so you can interrupt and resume safely.

Usage:
    python annotate_golden.py
    python annotate_golden.py --model llama3.1:8b
    python annotate_golden.py --resume   # skip already-annotated rows
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"

INTENTS = [
    "account_access",
    "billing_payment",
    "app_software_issue",
    "device_issue",
    "subscription",
    "refund_return",
    "order_purchase",
    "connectivity_issue",
    "general_information",
    "complaint",
    "insufficient_information",
]

SYSTEM_PROMPT = """You are an Apple Support intent annotator. Given a customer message, assign labels.

INTENTS (pick exactly one):
- account_access: Apple ID, iCloud, password, login, account lockout
- billing_payment: charges, invoices, payment errors, unexpected payments
- app_software_issue: iOS/macOS/app bugs, updates, crashes, software errors, slow performance (use this when device name + software problem)
- device_issue: physical hardware only — battery drain, cracked screen, won't turn on, physical damage
- subscription: Apple Music, Apple TV+, iCloud storage subscription, renewal, cancellation
- refund_return: refund request, return, money back, cancel order
- order_purchase: buying, ordering, shipping, delivery, tracking
- connectivity_issue: Wi-Fi, Bluetooth, cellular, mobile data, network connection
- general_information: how-to questions, product info, policy questions
- complaint: explicit complaint or strongly negative feedback with no clearer actionable issue
- insufficient_information: bare mention, URL only, too short to classify, no actionable content

RULES:
- app_software_issue beats device_issue when both device name AND software problem appear
- Return ONLY valid JSON, no explanation, no markdown
"""


def build_prompt(message: str, history: str) -> str:
    context = f"\nConversation history:\n{history}" if history and str(history).strip() else ""
    return f"""{SYSTEM_PROMPT}

Customer message: {message}{context}

Return JSON with exactly these keys:
{{
  "intent": "<one of the 11 intents>",
  "should_escalate": <true or false>,
  "expected_resolution": "<one sentence describing the ideal resolution>",
  "difficulty": "<easy|medium|hard>"
}}"""


def call_ollama(prompt: str, model: str, timeout: int = 60) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"].strip()


def extract_json(text: str) -> dict:
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON found in: {text[:200]}")


def validate(data: dict) -> dict:
    intent = str(data.get("intent", "")).strip().lower().replace(" ", "_")
    if intent not in INTENTS:
        # fuzzy match
        for i in INTENTS:
            if i in intent or intent in i:
                intent = i
                break
        else:
            intent = "insufficient_information"
    escalate = data.get("should_escalate", False)
    if isinstance(escalate, str):
        escalate = escalate.lower() in ("true", "yes", "1")
    resolution = str(data.get("expected_resolution", "")).strip() or "Provide relevant troubleshooting steps."
    difficulty = str(data.get("difficulty", "medium")).strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    return {
        "intent": intent,
        "should_escalate": escalate,
        "expected_resolution": resolution,
        "difficulty": difficulty,
    }


def annotate(golden_path: Path, model: str, resume: bool, retries: int, delay: float) -> None:
    df = pd.read_csv(golden_path).fillna("")

    for col in ("intent", "should_escalate", "expected_resolution", "difficulty", "annotation_status"):
        if col not in df.columns:
            df[col] = ""

    total = len(df)
    annotated = 0
    skipped = 0

    for idx, row in df.iterrows():
        already_done = str(row.get("intent", "")).strip() != ""
        if resume and already_done:
            skipped += 1
            continue

        message = str(row["customer_message"]).strip()
        if not message:
            df.at[idx, "intent"] = "insufficient_information"
            df.at[idx, "should_escalate"] = False
            df.at[idx, "expected_resolution"] = "No message content to act on."
            df.at[idx, "difficulty"] = "easy"
            df.at[idx, "annotation_status"] = "human_review_required"
            annotated += 1
            continue

        history = str(row.get("conversation_history", "")).strip()
        prompt = build_prompt(message, history)

        result = None
        for attempt in range(1, retries + 1):
            try:
                raw = call_ollama(prompt, model)
                result = validate(extract_json(raw))
                break
            except Exception as exc:
                print(f"  Row {idx} attempt {attempt}/{retries} failed: {exc}")
                if attempt < retries:
                    time.sleep(delay)

        if result is None:
            # Never turn an annotation failure into a claimed golden label.
            df.at[idx, "annotation_status"] = "human_review_required"
            print(f"  Row {idx}: no automatic label; left blank for human review")
            df.to_csv(golden_path, index=False)
            continue

        df.at[idx, "intent"] = result["intent"]
        df.at[idx, "should_escalate"] = result["should_escalate"]
        df.at[idx, "expected_resolution"] = result["expected_resolution"]
        df.at[idx, "difficulty"] = result["difficulty"]
        df.at[idx, "annotation_status"] = "ollama_suggestion_pending_human_review"
        annotated += 1

        # Save after every row so progress is never lost
        df.to_csv(golden_path, index=False)

        print(f"[{annotated + skipped}/{total}] id={row.get('id', idx)}  intent={result['intent']}  escalate={result['should_escalate']}  difficulty={result['difficulty']}")

    print(f"\nDone. Annotated {annotated} rows, skipped {skipped} already-done rows.")
    print(f"Saved to {golden_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-annotate golden_set.csv using Ollama")
    parser.add_argument("--input", type=Path, default=Path("data/golden_set.csv"))
    parser.add_argument("--model", type=str, default="qwen2.5:7b")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip rows that already have an intent label (default: True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Re-annotate all rows even if already labeled")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Seconds to wait between retries")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found. Run generate_golden.py first.")

    print(f"Annotating {args.input} with model={args.model}  resume={args.resume}")
    annotate(args.input, args.model, args.resume, args.retries, args.delay)


if __name__ == "__main__":
    main()
