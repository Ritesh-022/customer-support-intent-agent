"""AppleSupport AI support system.

Components:
- Majority and TF-IDF + Logistic Regression intent classifiers.
- Optional SentenceTransformer + Logistic Regression classifier.
- Historical TF-IDF or embedding retrieval.
- Evidence-constrained reply generation via local Ollama (default: qwen2.5:7b).
- Transparent escalation policy.

Training labels must contain: customer_message, intent.
The extracted AppleSupport working set is unlabeled and is used for retrieval only.

Ollama must be running locally: https://ollama.com
  ollama pull qwen2.5:7b
  ollama serve

Override model:  OLLAMA_MODEL=llama3.1:8b
Override URL:    OLLAMA_URL=http://localhost:11434/api/chat
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize


class MajorityClassifier:
    def fit(self, texts: list[str], labels: list[str]) -> "MajorityClassifier":
        if not labels:
            raise ValueError("At least one labeled example is required")
        values, counts = np.unique(labels, return_counts=True)
        self.label_ = str(values[int(np.argmax(counts))])
        self.classes_ = sorted(set(labels))
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return [self.label_] * len(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        result = np.zeros((len(texts), len(self.classes_)))
        result[:, self.classes_.index(self.label_)] = 1.0
        return result


class TfidfLogisticClassifier:
    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

    def fit(self, texts: list[str], labels: list[str]) -> "TfidfLogisticClassifier":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return self.pipeline.predict(texts).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)


class SentenceEmbeddingLogisticClassifier:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.encoder = None
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    def _load_encoder(self):
        if self.encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "SentenceTransformer is optional but required for Model C. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self.encoder = SentenceTransformer(self.model_name)
        return self.encoder

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._load_encoder().encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def fit(self, texts: list[str], labels: list[str]) -> "SentenceEmbeddingLogisticClassifier":
        self.classifier.fit(self.encode(texts), labels)
        return self

    def predict(self, texts: list[str]) -> list[str]:
        return self.classifier.predict(self.encode(texts)).tolist()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.classifier.predict_proba(self.encode(texts))


class HistoricalRetriever:
    def __init__(self, backend: str = "tfidf", embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.backend = backend
        self.embedding_model = embedding_model
        self.examples: list[dict[str, str]] = []
        self.vectorizer = None
        self.matrix = None
        self.encoder = None

    def fit(self, examples: pd.DataFrame) -> "HistoricalRetriever":
        required = {"customer_message", "brand_response"}
        missing = required.difference(examples.columns)
        if missing:
            raise ValueError(f"Retrieval data is missing columns: {sorted(missing)}")
        self.examples = examples.fillna("").to_dict("records")
        texts = [str(row["customer_message"]) for row in self.examples]
        if self.backend == "embedding":
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Embedding retrieval requires sentence-transformers. "
                    "Install it with: pip install sentence-transformers"
                ) from exc
            self.encoder = SentenceTransformer(self.embedding_model)
            self.matrix = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        else:
            self.vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
            self.matrix = normalize(self.vectorizer.fit_transform(texts))
        return self

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.matrix is None:
            raise RuntimeError("Retriever has not been fitted")
        if self.backend == "embedding":
            query_vector = self.encoder.encode([query], normalize_embeddings=True)[0]
            scores = np.asarray(self.matrix @ query_vector).reshape(-1)
        else:
            query_vector = normalize(self.vectorizer.transform([query]))
            scores = (query_vector @ self.matrix.T).toarray().reshape(-1)
        indexes = np.argsort(-scores)[:top_k]
        return [{"customer": self.examples[index]["customer_message"],
                 "response": self.examples[index]["brand_response"],
                 "similarity": round(float(scores[index]), 4),
                 "conversation_id": self.examples[index].get("conversation_id", "")}
                for index in indexes]


@dataclass
class EscalationPolicy:
    confidence_threshold: float = 0.60
    similarity_threshold: float = 0.35

    def decide(self, confidence: float, evidence: list[dict[str, Any]], message: str) -> tuple[bool, str]:
        risk_terms = re.compile(r"\b(fraud|scam|stolen|hack|hacked|legal|lawyer|police|unsafe|threat)\b", re.I)
        if confidence < self.confidence_threshold:
            return True, "Intent confidence is below the auto-handle threshold."
        if not evidence or evidence[0]["similarity"] < self.similarity_threshold:
            return True, "Historical evidence is insufficiently similar."
        if risk_terms.search(message):
            return True, "The message contains a sensitive or high-risk signal."
        return False, "High-confidence intent with strong historical evidence."


class ReplyGenerator:
    """Generate Apple Support replies via a local Ollama model.

    Environment variables:
        OLLAMA_MODEL  -- model tag (default: qwen2.5:7b)
        OLLAMA_URL    -- Ollama chat endpoint (default: http://localhost:11434/api/chat)

    Ollama must be running locally: https://ollama.com
        ollama pull qwen2.5:7b
        ollama serve
    """

    def __init__(self, model=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    def generate(self, message: str, intent: str, evidence: list, escalate: bool) -> str:
        evidence_text = "\n\n".join(
            f"Customer: {item['customer']}\nHistorical response: {item['response']}"
            for item in evidence
        )
        rule = (
            "Recommend escalation and do not promise resolution."
            if escalate
            else "Give a concise helpful next step."
        )
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
            "- If account-specific information is required, ask the customer to "
            "contact support privately.\n"
            f"- {rule}"
        )
        try:
            import requests as _requests
            response = _requests.post(
                self.url,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0},
                },
                timeout=120,
            )
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except Exception:
            # Safe fallback when Ollama is unavailable
            if escalate:
                return (
                    "I'm sorry you're dealing with this. Please contact Apple Support "
                    "through a secure channel so a specialist can review your case."
                )
            if evidence:
                return (
                    "Thanks for reaching out. Based on similar Apple Support cases, "
                    "please follow the relevant troubleshooting steps and contact us "
                    "privately if you need account-specific assistance."
                )
            return "Thanks for reaching out. We need a bit more information to determine the correct next step."

class SupportAgent:
    def __init__(self, classifier: Any, retriever: HistoricalRetriever,
                 policy: EscalationPolicy | None = None, generator: ReplyGenerator | None = None) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.policy = policy or EscalationPolicy()
        self.generator = generator or ReplyGenerator()

    def respond(self, message: str, top_k: int = 5) -> dict[str, Any]:
        probabilities = self.classifier.predict_proba([message])[0]
        intent_index = int(np.argmax(probabilities))
        if hasattr(self.classifier, "pipeline"):
            intent = self.classifier.pipeline.classes_[intent_index]
        elif hasattr(self.classifier, "classifier"):
            intent = self.classifier.classifier.classes_[intent_index]
        else:
            intent = self.classifier.classes_[intent_index]
        confidence = float(probabilities[intent_index])
        evidence = self.retriever.search(message, top_k=top_k)
        escalate, reason = self.policy.decide(confidence, evidence, message)
        return {"intent": intent, "confidence": round(confidence, 4), "reply": self.generator.generate(message, intent, evidence, escalate),
                "escalate": escalate, "reason": reason, "historical_evidence": evidence}


def train_models(labels_path: Path, model_dir: Path, high_confidence_only: bool = False,
                 confidence_threshold: float = 0.90) -> None:
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Training labels not found: {labels_path}. "
            "Run label_data.py first, then rerun training."
        )
    data = pd.read_csv(labels_path).fillna("")
    required = {"customer_message", "intent"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Training labels are missing columns: {sorted(missing)}")
    data["intent"] = data["intent"].astype(str).str.strip()
    data = data[data["intent"].ne("")].copy()
    if data.empty:
        raise ValueError(
            f"No intent labels found in {labels_path}. Run label_data.py first."
        )
    # Support both old column name (intent_confidence) and new (rule_confidence)
    conf_col = "rule_confidence" if "rule_confidence" in data.columns else (
        "intent_confidence" if "intent_confidence" in data.columns else None)
    if high_confidence_only:
        if conf_col is None:
            raise ValueError("--high-confidence-only requires a rule_confidence or intent_confidence column")
        before = len(data)
        data = data[data[conf_col].astype(float) >= confidence_threshold].copy()
        print(f"High-confidence filter: {len(data):,} / {before:,} rows kept (rule_confidence >= {confidence_threshold})")
        if data.empty:
            raise ValueError(f"No rows remain after confidence filter >= {confidence_threshold}")
    texts = data["customer_message"].astype(str).tolist()
    labels = data["intent"].astype(str).tolist()
    if len(set(labels)) < 2:
        raise ValueError(
            f"At least two intent classes are required; found {sorted(set(labels))}. "
            "Add labels from at least two taxonomy categories before training."
        )
    model_dir.mkdir(parents=True, exist_ok=True)
    models = {"majority": MajorityClassifier().fit(texts, labels), "tfidf_logistic": TfidfLogisticClassifier().fit(texts, labels)}
    for name, model in models.items():
        joblib.dump(model, model_dir / f"{name}.joblib")
    try:
        models["embedding_logistic"] = SentenceEmbeddingLogisticClassifier().fit(texts, labels)
        joblib.dump(models["embedding_logistic"], model_dir / "embedding_logistic.joblib")
        print("Saved Model C: sentence embeddings + Logistic Regression")
    except RuntimeError as error:
        print(f"Skipped Model C: {error}")
    suffix = f" (high-confidence subset, threshold={confidence_threshold})" if high_confidence_only else " (all weak labels)"
    print(f"Saved {len(models)} classifier models to {model_dir.resolve()}{suffix}")


def create_label_template(source_path: Path, output_path: Path, rows: int) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Working dataset not found: {source_path}")
    source = pd.read_csv(source_path, nrows=rows).fillna("")
    if "customer_message" not in source.columns:
        raise ValueError("Working dataset must contain customer_message")
    columns = [column for column in ("conversation_id", "customer_tweet_id", "customer_message", "brand_response", "timestamp") if column in source]
    template = source[columns].copy()
    template["intent"] = ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(output_path, index=False)
    print(f"Created {len(template):,} unlabeled rows at {output_path}")
    print("Annotate the intent column with your frozen taxonomy before training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train AppleSupport intent models")
    parser.add_argument("--labels", type=Path, help="Labeled CSV with customer_message and intent")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--create-label-template", action="store_true")
    parser.add_argument("--source", type=Path, default=Path("data/processed/apple_support_train.csv"))
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--high-confidence-only", action="store_true",
                        help="Train only on rows with rule_confidence >= --confidence-threshold")
    parser.add_argument("--confidence-threshold", type=float, default=0.90,
                        help="Minimum rule_confidence for --high-confidence-only (default: 0.90)")
    args = parser.parse_args()
    if args.create_label_template:
        output = args.labels or Path("data/labeled_training.csv")
        create_label_template(args.source, output, args.rows)
        return
    if not args.labels:
        parser.error("--labels is required; run label_data.py first")
    train_models(args.labels, args.model_dir, args.high_confidence_only, args.confidence_threshold)


if __name__ == "__main__":
    main()
