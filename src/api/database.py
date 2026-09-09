"""
Database layer — SQLite via SQLAlchemy.

Stores every completed generation session:
  - The prompt
  - The full AI response
  - Aggregated metrics (hallucination risk, entropy, gradient norm, mc variance)
  - Timestamp
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./hallucination_monitor.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Session(Base):
    """One row = one completed prompt + response analysis."""
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    response = Column(Text, nullable=False)

    # Aggregated signal metrics (averages across all tokens)
    hallucination_risk = Column(Float, nullable=False)   # Overall lie/hallucination score 0–1
    avg_entropy = Column(Float, nullable=False)
    avg_gradient_norm = Column(Float, nullable=False)
    avg_mc_variance = Column(Float, nullable=False)

    # Token-level detail counts
    total_tokens = Column(Integer, nullable=False)
    flagged_tokens = Column(Integer, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency: yields a DB session, always closes after."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
