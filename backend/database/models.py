"""SQLAlchemy models for AlloyGen 2.0."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.orm import relationship

from backend.database.engine import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    display_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    models = relationship("TrainedModel", back_populates="user", cascade="all, delete-orphan")


class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    state = Column(String, default="requirements")
    data_path = Column(String, nullable=True)
    model_path = Column(String, nullable=True)
    targets_json = Column(JSON, default=list)
    task_types_json = Column(JSON, default=list)
    created_at = Column(DateTime, default=_utcnow)
    last_active = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan", order_by="ChatMessage.timestamp")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # "user", "assistant", "status"
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=_utcnow)

    session = relationship("ChatSession", back_populates="messages")


class TrainedModel(Base):
    __tablename__ = "trained_models"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String, nullable=False)
    best_model_name = Column(String, nullable=True)
    targets_json = Column(JSON, default=list)
    task_type = Column(String, nullable=True)
    score = Column(Float, nullable=True)
    cv_score = Column(Float, nullable=True)
    features_json = Column(JSON, default=list)
    feature_stats_json = Column(JSON, default=dict)
    target_stats_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow)

    user = relationship("User", back_populates="models")
    experiments = relationship("Experiment", back_populates="model", cascade="all, delete-orphan")


class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String, primary_key=True, default=_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    model_id = Column(String, ForeignKey("trained_models.id"), nullable=True)
    experiment_type = Column(String, nullable=False)  # "prediction", "inverse", "batch"
    input_json = Column(JSON, default=dict)
    output_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=_utcnow)

    model = relationship("TrainedModel", back_populates="experiments")
