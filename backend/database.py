"""SQLAlchemy models and database setup for conversation persistence.

Tables:
    conversations  — top-level session with status and escalation flag
    messages       — each customer or agent message in a conversation
    agent_decisions — the ML/AI decision record for each customer message
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))
    status = Column(String(32), default="active")  # active | resolved | escalated
    should_escalate = Column(Boolean, default=False)

    messages = relationship("Message", back_populates="conversation",
                            order_by="Message.timestamp", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "status": self.status,
            "should_escalate": self.should_escalate,
            "message_count": len(self.messages) if self.messages else 0,
        }


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    sender = Column(String(16), nullable=False)  # "customer" | "agent"
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    intent = Column(String(64), nullable=True)
    confidence = Column(Float, nullable=True)

    conversation = relationship("Conversation", back_populates="messages")
    decision = relationship("AgentDecision", back_populates="message",
                            uselist=False, cascade="all, delete-orphan")

    def to_dict(self):
        result = {
            "id": self.id,
            "conversation_id": self.conversation_id,
            "sender": self.sender,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "intent": self.intent,
            "confidence": self.confidence,
        }
        if self.decision:
            result["decision"] = self.decision.to_dict()
        return result


class AgentDecision(Base):
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    message_id = Column(Integer, ForeignKey("messages.id"), nullable=False, unique=True)
    decision = Column(String(32), nullable=False)  # "auto_handle" | "escalate"
    escalation_reason = Column(Text, nullable=True)
    evidence_conversation_id = Column(String(128), nullable=True)
    evidence_similarity = Column(Float, nullable=True)
    generated_reply = Column(Text, nullable=True)

    message = relationship("Message", back_populates="decision")

    def to_dict(self):
        return {
            "id": self.id,
            "message_id": self.message_id,
            "decision": self.decision,
            "escalation_reason": self.escalation_reason,
            "evidence_conversation_id": self.evidence_conversation_id,
            "evidence_similarity": self.evidence_similarity,
            "generated_reply": self.generated_reply,
        }


def init_db(db_path: str = "sqlite:///support_agent.db"):
    """Create engine, tables, and return a session factory."""
    engine = create_engine(db_path, echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return engine, Session
