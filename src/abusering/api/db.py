"""SQLAlchemy persistence layer.

Sec 28 asks for PostgreSQL. This sandbox has no reachable Postgres server
(network is allow-listed to package registries only), so we use SQLite via
the same SQLAlchemy ORM — swapping DATABASE_URL to a postgres:// DSN in a
real deployment requires no code changes here, only a connection string.
This is called out explicitly rather than silently substituted.
"""

from __future__ import annotations

import datetime as dt
import os

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get("ABUSERING_DATABASE_URL", "sqlite:///./artifacts/abusering.db")

Base = declarative_base()


class PredictionAudit(Base):
    __tablename__ = "prediction_audit"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)
    transaction_id = Column(String, index=True)
    model_version = Column(String)
    feature_version = Column(String)
    risk_score = Column(Float)
    prediction = Column(String)
    threshold = Column(Float)
    top_signals = Column(JSON)
    inference_latency_ms = Column(Float)


engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()
