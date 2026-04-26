"""
Long-term memory: persistent conversation facts and summaries backed by SQLite.

Design:
- SQLAlchemy async ORM with SQLite (swap to Postgres via DATABASE_URL).
- Three tables:
    - conversations: session-level metadata and summaries.
    - facts: extracted structured facts per customer/user.
    - interactions: full message log for audit/analytics.
- LLM-generated summaries compress long sessions into retrievable facts.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from sqlalchemy import String, Text, DateTime, select, func
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config import get_settings

log = structlog.get_logger(__name__)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    agent_type: Mapped[str] = mapped_column(String(64))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Fact(Base):
    """Extracted customer/user facts for personalisation."""
    __tablename__ = "facts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String(128), index=True)
    key: Mapped[str] = mapped_column(String(128))         # e.g. "preferred_contact", "tier"
    value: Mapped[str] = mapped_column(Text)
    source_session: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float] = mapped_column(default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class Interaction(Base):
    """Full message audit log."""
    __tablename__ = "interactions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(128), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    agent_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------
class LongTermMemory:
    """Async repository for persistent memory operations."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._engine = create_async_engine(
            cfg.database_url,
            echo=cfg.api_debug,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create tables (idempotent)."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        log.info("memory.long_term.db_initialised")

    # --- Conversation ---

    async def create_conversation(
        self,
        session_id: str,
        agent_type: str,
        user_id: str | None = None,
    ) -> Conversation:
        async with self._session_factory() as session:
            conv = Conversation(
                session_id=session_id,
                agent_type=agent_type,
                user_id=user_id,
            )
            session.add(conv)
            await session.commit()
            return conv

    async def get_conversation(self, session_id: str) -> Conversation | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .order_by(Conversation.created_at.desc())
                .limit(1)
            )
            return result.scalar_one_or_none()

    async def update_summary(self, session_id: str, summary: str) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .limit(1)
            )
            conv = result.scalar_one_or_none()
            if conv:
                conv.summary = summary
                conv.updated_at = _now()
                await session.commit()

    async def close_conversation(self, session_id: str) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Conversation).where(Conversation.session_id == session_id)
            )
            for conv in result.scalars():
                conv.status = "closed"
                conv.updated_at = _now()
            await session.commit()

    # --- Facts ---

    async def upsert_fact(
        self,
        user_id: str,
        key: str,
        value: str,
        source_session: str | None = None,
        confidence: float = 1.0,
    ) -> None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Fact)
                .where(Fact.user_id == user_id, Fact.key == key)
                .limit(1)
            )
            fact = result.scalar_one_or_none()
            if fact:
                fact.value = value
                fact.confidence = confidence
                fact.updated_at = _now()
            else:
                fact = Fact(
                    user_id=user_id,
                    key=key,
                    value=value,
                    source_session=source_session,
                    confidence=confidence,
                )
                session.add(fact)
            await session.commit()

    async def get_facts(self, user_id: str) -> dict[str, str]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Fact).where(Fact.user_id == user_id)
            )
            return {f.key: f.value for f in result.scalars()}

    # --- Interactions ---

    async def log_interaction(
        self,
        session_id: str,
        role: str,
        content: str,
        agent_type: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        async with self._session_factory() as session:
            interaction = Interaction(
                session_id=session_id,
                role=role,
                content=content,
                agent_type=agent_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )
            session.add(interaction)
            await session.commit()

    async def get_session_interactions(self, session_id: str) -> list[Interaction]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(Interaction)
                .where(Interaction.session_id == session_id)
                .order_by(Interaction.created_at)
            )
            return list(result.scalars())

    # --- Analytics ---

    async def total_cost_today(self) -> float:
        from datetime import date
        today = date.today()
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.sum(Interaction.cost_usd)).where(
                    func.date(Interaction.created_at) == today.isoformat()
                )
            )
            return float(result.scalar() or 0.0)

    async def conversation_count_today(self) -> int:
        from datetime import date
        today = date.today()
        async with self._session_factory() as session:
            result = await session.execute(
                select(func.count(Conversation.id)).where(
                    func.date(Conversation.created_at) == today.isoformat()
                )
            )
            return int(result.scalar() or 0)
