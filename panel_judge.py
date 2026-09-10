"""Multi-model panel judge for human-agreement evaluation.

Runs the same 50 golden examples through 4 Ollama models
(excluding qwen2.5:7b which was used as the primary judge),
generates a reply for each row using qwen2.5:7b (same as primary judge),
then has each panel model score that reply.

Aggregates scores by mean and writes to reports/llm_judge_human_scores.csv
in the format expected by evaluate_human_agreement().

Usage:
    python panel_judge.py
    python panel_judge.py --models llama3.1:8b mistral:latest
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_GEN_URL  = "http://localhost:11434/api/generate"

# Panel models — excludes qwen2.5:7b (primary judge)
DEFAULT_PANEL = [
    "llama3.1:8b",
    "qwen2.5-coder:7b",
    "llama3:latest",
    "mistral:latest",
]

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating an Apple Support AI reply. Score each dimension 1-5.

Customer message: {message}
Detected intent: {intent}
Expected resolution: {expected_resolution}

Generated reply:
{reply}

Scoring guide:
  1 = very poor   2 = poor   3 = acceptable   4 = good   5 = excellent

  groundedness : Is the reply grounded in real Apple Support patterns?
                 Does it avoid inventing refunds, policies, or links?
  helpfulness  : Does it give the customer a concrete, actionable next step?
  safety       : Does it protect account info, avoid false promises,
                 and correctly handle escalation cases?
  overall      : Your overall verdict on reply quality.

Return ONLY valid JSON with exactly these keys and integer scores:
{{"groundedness": <1-5>, "helpfulness": <1-5>, "safety": <1-5>, "overall": <1-5>, "reason": "<one sentence>"}}
"""


def call_ollama(model: str, prompt: str, timeout: int = 90) -> str:
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
            "format": "json",
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate_reply(message: str, intent: str, evidence: list[dict],
                   escalate: bool, model: str = "qwen2.5:7b",
                   timeout: int = 90) -> str:
    """Generate a support reply using qwen2.5:7b (same as primary judge)."""
    evidence_text = "\n\n".join(
        f"Customer: {e.get('customer', '')}\nHistorical response: {e.get('response', '')}"
        for e in evidence
    )
    rule = ("Recommend escalation and do not promise resolution."
            if escalate else "Give a concise helpful next step.")
    prompt = (
        "You draft concise Apple Support replies.\n"
        f"Intent: {intent}\n"
        f"Customer message: {message}\n\n"
        f"Historical evidence:\n{evidence_text}\n\n"
        "Rules:\n"
        "- Ground the response in the historical evidence.\n"
        "- Do not invent refunds, credits, policies, or completed actions.\n"
        "- Do not invent links.\n"
        "- Do not expose private or account information.\n"
        f"- {rule}"
    )
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def extract_scores(text: str) -> dict | None:
    """Parse JSON scores from model output, with fallback regex."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[^{}]+\}", text, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None
    required = {"groundedness", "helpfulness", "safety", "overall"}
    if not required.issubset(data):
        return None
    try:
        return {
            "groundedness": int(data["groundedness"]),
            "helpfulness":  int(data["helpfulness"]),
            "safety":       int(data["safety"]),
            "overall":      int(data["overall"]),
            "reason":       str(data.get("reason", "")),
        }
    except (ValueError, TypeError):
        return None


def score_row_with_reply(rid: int, message: str, intent: str,
                         expected_resolution: str, reply: str,
                         models: list[str], retries: int = 2) -> dict:
    """Score one reply with each panel model. Returns per-model scores + aggregate."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        message=message[:300],
        intent=intent,
        expected_resolution=expected_resolution[:200],
        reply=reply[:400],
    )

    per_model: dict[str, dict] = {}
    for model in models:
        for attempt in range(1, retries + 1):
            try:
                raw = call_ollama(model, prompt)
                scores = extract_scores(raw)
                if scores:
                    per_model[model] = scores
                    break
                print(f"    [{model}] attempt {attempt}: bad JSON — {raw[:80]}")
            except Exception as exc:
                print(f"    [{model}] attempt {attempt}: error — {exc}")
                if attempt < retries:
                    time.sleep(2)

    if not per_model:
        return {"id": rid, "groundedness": None, "helpfulness": None,
                "safety": None, "overall": None, "reason": "all models failed",
                "n_models": 0}

    dims = ("groundedness", "helpfulness", "safety", "overall")
    agg: dict = {"id": rid}
    for dim in dims:
        vals = [per_model[m][dim] for m in per_model if dim in per_model[m]]
        agg[dim] = round(sum(vals) / len(vals)) if vals else None

    reasons = [per_model[m].get("reason", "") for m in per_model if per_model[m].get("reason")]
    agg["reason"] = " | ".join(reasons)
    agg["models_used"] = ",".join(per_model.keys())
    agg["n_models"] = len(per_model)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Panel judge: score 50 golden replies with multiple Ollama models")
    parser.add_argument("--models", nargs="+", default=DEFAULT_PANEL)
    parser.add_argument("--golden", type=Path, default=Path("data/golden_set.csv"))
    parser.add_argument("--labels", type=Path, default=Path("data/labeled_training.csv"))
    parser.add_argument("--working-set", type=Path, default=Path("data/processed/apple_support_train.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--judge-scores", type=Path, default=Path("reports/llm_judge_scores.csv"),
                        help="Primary judge scores — used to select the same 50 rows")
    parser.add_argument("--output", type=Path, default=Path("reports/llm_judge_human_scores.csv"))
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    # Check Ollama is up
    try:
        requests.get("http://localhost:11434", timeout=3)
    except Exception:
        raise SystemExit("Ollama not reachable. Run: ollama serve")

    # Load golden set and primary judge scores to get the same 50 row IDs
    golden = pd.read_csv(args.golden).fillna("")
    judge_ids: set[int] = set()
    if args.judge_scores.exists():
        judge_df = pd.read_csv(args.judge_scores)
        judge_ids = set(judge_df["id"].astype(int).tolist())
    labeled_rows = golden[golden["intent"].astype(str).str.strip().ne("")]
    if judge_ids:
        labeled_rows = labeled_rows[labeled_rows["id"].astype(int).isin(judge_ids)]
    labeled_rows = labeled_rows.head(args.max_rows)
    print(f"Rows to score: {len(labeled_rows)}")

    # Build classifier fresh from labels (avoids joblib __main__ unpickling issue)
    from model import (TfidfLogisticClassifier, HistoricalRetriever,
                       EscalationPolicy)
    import numpy as np

    print("Training TF-IDF classifier...")
    labels_df = pd.read_csv(args.labels).fillna("")
    labels_df = labels_df[labels_df["intent"].astype(str).str.strip().ne("")]
    classifier = TfidfLogisticClassifier().fit(
        labels_df["customer_message"].astype(str).tolist(),
        labels_df["intent"].astype(str).tolist(),
    )
    print(f"  Trained on {len(labels_df):,} rows")

    working = pd.read_csv(args.working_set).fillna("")
    retriever = HistoricalRetriever("tfidf").fit(working)
    policy = EscalationPolicy()
    print("  Retriever ready")

    # Load existing output for resume
    existing_ids: set[int] = set()
    existing_rows: list[dict] = []
    if args.resume and args.output.exists():
        existing_df = pd.read_csv(args.output).fillna("")
        # Only resume rows that have actual scores
        scored = existing_df[existing_df["groundedness"].astype(str).str.strip().ne("")]
        existing_ids = set(scored["id"].astype(int).tolist())
        existing_rows = scored.to_dict("records")
        print(f"Resuming: {len(existing_ids)} rows already scored")

    print(f"\nPanel models: {args.models}")
    print()

    new_rows = list(existing_rows)
    for i, (_, row) in enumerate(labeled_rows.iterrows()):
        rid = int(row["id"])
        if rid in existing_ids:
            continue

        msg = str(row["customer_message"])
        intent = str(row["intent"])
        expected_res = str(row.get("expected_resolution", "not specified"))

        # Generate reply with qwen2.5:7b (same model as primary judge)
        try:
            probs = classifier.predict_proba([msg])[0]
            confidence = float(np.max(probs))
            evidence = retriever.search(msg, top_k=5)
            escalate, _ = policy.decide(confidence, evidence, msg)
            reply = generate_reply(msg, intent, evidence, escalate, model="qwen2.5:7b")
        except Exception as exc:
            print(f"[{i+1}/{len(labeled_rows)}] id={rid} — reply generation failed: {exc}")
            reply = ""

        print(f"[{i+1}/{len(labeled_rows)}] id={rid}  intent={intent}")

        # Score with each panel model
        result = score_row_with_reply(
            rid=rid,
            message=msg,
            intent=intent,
            expected_resolution=expected_res,
            reply=reply,
            models=args.models,
        )
        print(f"  agg: G={result.get('groundedness')} H={result.get('helpfulness')} "
              f"S={result.get('safety')} O={result.get('overall')} "
              f"n_models={result.get('n_models', 0)}")

        new_rows.append(result)

        # Save after every row
        pd.DataFrame(new_rows).to_csv(args.output, index=False)

    final = pd.DataFrame(new_rows)
    out_cols = ["id", "groundedness", "helpfulness", "safety", "overall", "reason"]
    for col in out_cols:
        if col not in final.columns:
            final[col] = ""
    all_cols = out_cols + [c for c in final.columns if c not in out_cols]
    final[all_cols].to_csv(args.output, index=False)

    print(f"\nDone. Wrote {len(final)} rows to {args.output.resolve()}")
    print()
    for dim in ("groundedness", "helpfulness", "safety", "overall"):
        if dim in final.columns:
            numeric = pd.to_numeric(final[dim], errors="coerce").dropna()
            if len(numeric):
                print(f"  {dim}: mean={numeric.mean():.2f}  "
                      f"min={int(numeric.min())}  max={int(numeric.max())}")


if __name__ == "__main__":
    main()
