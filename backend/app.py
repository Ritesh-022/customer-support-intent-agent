"""Flask API for the AppleSupport AI customer support agent.

Endpoints:
    POST   /api/conversations              — Start a new conversation
    GET    /api/conversations              — List all conversations
    GET    /api/conversations/<id>         — Get conversation with messages
    POST   /api/conversations/<id>/chat    — Send a customer message, get AI response
    PATCH  /api/conversations/<id>         — Update conversation status
    GET    /api/stats                      — Dashboard stats
    GET    /api/taxonomy                   — Intent taxonomy
    GET    /api/health                     — Health check

Loads trained models from ../models/ and retrieval data from ../data/processed/.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS

# Add project root to path so we can import model.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
import model as _model_module
from model import (
    EscalationPolicy, HistoricalRetriever, MajorityClassifier,
    ReplyGenerator, SentenceEmbeddingLogisticClassifier, SupportAgent,
    TfidfLogisticClassifier,
)

# The trained models were pickled when model.py ran as __main__,
# so the classes are stored as __main__.TfidfLogisticClassifier etc.
# Register them under __main__ so joblib.load can find them.
import __main__ as _main
for _cls in (MajorityClassifier, TfidfLogisticClassifier,
             SentenceEmbeddingLogisticClassifier):
    setattr(_main, _cls.__name__, _cls)

from database import AgentDecision, Conversation, Message, init_db

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(db_path: str | None = None) -> Flask:
    app = Flask(__name__)
    CORS(app)

    # --- Database ---
    if db_path is None:
        db_file = PROJECT_ROOT / "support_agent.db"
        db_path = f"sqlite:///{db_file}"
    engine, Session = init_db(db_path)

    # --- Load ML models ---
    model_dir = PROJECT_ROOT / "models"
    classifier_path = model_dir / "tfidf_logistic.joblib"
    if not classifier_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {classifier_path}. "
            "Run: python model.py --labels data/labeled_training.csv"
        )
    classifier = joblib.load(classifier_path)

    # --- Load retrieval data ---
    working_set_path = PROJECT_ROOT / "data" / "processed" / "apple_support_train.csv"
    if not working_set_path.exists():
        raise FileNotFoundError(
            f"Working set not found at {working_set_path}. "
            "Run: python extract_apple_support.py"
        )
    retriever = HistoricalRetriever("tfidf").fit(pd.read_csv(working_set_path))

    # --- Build agent ---
    policy = EscalationPolicy()
    generator = ReplyGenerator()
    agent = SupportAgent(classifier, retriever, policy, generator)

    # --- Load taxonomy ---
    taxonomy_path = PROJECT_ROOT / "data" / "taxonomy.json"
    taxonomy = {}
    if taxonomy_path.exists():
        taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": "tfidf_logistic"})

    @app.route("/api/taxonomy", methods=["GET"])
    def get_taxonomy():
        return jsonify(taxonomy)

    @app.route("/api/conversations", methods=["POST"])
    def create_conversation():
        session = Session()
        try:
            conv = Conversation()
            session.add(conv)
            session.commit()
            return jsonify(conv.to_dict()), 201
        finally:
            session.close()

    @app.route("/api/conversations", methods=["GET"])
    def list_conversations():
        session = Session()
        try:
            conversations = (
                session.query(Conversation)
                .order_by(Conversation.updated_at.desc())
                .limit(50)
                .all()
            )
            return jsonify([c.to_dict() for c in conversations])
        finally:
            session.close()

    @app.route("/api/conversations/<int:conv_id>", methods=["GET"])
    def get_conversation(conv_id: int):
        session = Session()
        try:
            conv = session.query(Conversation).get(conv_id)
            if not conv:
                return jsonify({"error": "Conversation not found"}), 404
            result = conv.to_dict()
            result["messages"] = [m.to_dict() for m in conv.messages]
            return jsonify(result)
        finally:
            session.close()

    @app.route("/api/conversations/<int:conv_id>/chat", methods=["POST"])
    def chat(conv_id: int):
        """Send a customer message and get the AI agent's response."""
        session = Session()
        try:
            conv = session.query(Conversation).get(conv_id)
            if not conv:
                return jsonify({"error": "Conversation not found"}), 404

            data = request.get_json(force=True)
            customer_message = data.get("message", "").strip()
            if not customer_message:
                return jsonify({"error": "message is required"}), 400

            # --- Run the agent ---
            result = agent.respond(customer_message)

            # --- Store customer message ---
            customer_msg = Message(
                conversation_id=conv_id,
                sender="customer",
                message=customer_message,
                intent=result["intent"],
                confidence=result["confidence"],
            )
            session.add(customer_msg)
            session.flush()  # get the ID

            # --- Store agent decision ---
            top_evidence = result["historical_evidence"][0] if result["historical_evidence"] else {}
            decision = AgentDecision(
                message_id=customer_msg.id,
                decision="escalate" if result["escalate"] else "auto_handle",
                escalation_reason=result["reason"],
                evidence_conversation_id=str(top_evidence.get("conversation_id", "")),
                evidence_similarity=top_evidence.get("similarity"),
                generated_reply=result["reply"],
            )
            session.add(decision)

            # --- Store agent reply as a message ---
            agent_msg = Message(
                conversation_id=conv_id,
                sender="agent",
                message=result["reply"],
            )
            session.add(agent_msg)

            # --- Update conversation status ---
            if result["escalate"]:
                conv.should_escalate = True
                conv.status = "escalated"

            session.commit()

            return jsonify({
                "customer_message": customer_msg.to_dict(),
                "agent_response": agent_msg.to_dict(),
                "analysis": {
                    "intent": result["intent"],
                    "confidence": result["confidence"],
                    "escalate": result["escalate"],
                    "reason": result["reason"],
                    "historical_evidence": result["historical_evidence"][:3],
                },
            })
        finally:
            session.close()

    @app.route("/api/conversations/<int:conv_id>", methods=["PATCH"])
    def update_conversation(conv_id: int):
        session = Session()
        try:
            conv = session.query(Conversation).get(conv_id)
            if not conv:
                return jsonify({"error": "Conversation not found"}), 404
            data = request.get_json(force=True)
            if "status" in data:
                conv.status = data["status"]
            session.commit()
            return jsonify(conv.to_dict())
        finally:
            session.close()

    @app.route("/api/stats", methods=["GET"])
    def stats():
        session = Session()
        try:
            total = session.query(Conversation).count()
            escalated = session.query(Conversation).filter_by(should_escalate=True).count()
            resolved = session.query(Conversation).filter_by(status="resolved").count()
            active = session.query(Conversation).filter_by(status="active").count()

            # Intent distribution from messages
            from sqlalchemy import func
            intent_counts = (
                session.query(Message.intent, func.count(Message.id))
                .filter(Message.intent.isnot(None))
                .group_by(Message.intent)
                .all()
            )
            intent_distribution = {intent: count for intent, count in intent_counts}

            return jsonify({
                "total_conversations": total,
                "active": active,
                "escalated": escalated,
                "resolved": resolved,
                "auto_handle_rate": round((total - escalated) / total, 4) if total else 0,
                "intent_distribution": intent_distribution,
            })
        finally:
            session.close()

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    print(f"Starting AppleSupport AI API on http://localhost:{port}")
    print("Endpoints:")
    print("  POST   /api/conversations         — New conversation")
    print("  GET    /api/conversations         — List conversations")
    print("  GET    /api/conversations/<id>    — Get conversation")
    print("  POST   /api/conversations/<id>/chat — Send message")
    print("  GET    /api/stats                 — Dashboard stats")
    print("  GET    /api/taxonomy              — Intent taxonomy")
    print("  GET    /api/health                — Health check")
    app.run(host="0.0.0.0", port=port, debug=True)
