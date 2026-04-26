"""
Short-term memory: per-session conversation history backed by Redis.

Design:
- Each session has a Redis list of serialised Message dicts.
- A sliding window keeps the N most recent messages to bound context size.
- TTL auto-expires idle sessions to control memory footprint.
- Falls back gracefully to an in-process dict when Redis is unavailable
  (development / testing convenience).
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import structlog

from src.config import get_settings
from src.llm.client import Message

log = structlog.get_logger(__name__)


class ShortTermMemory:
    """Redis-backed sliding window conversation history."""

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._redis: Any = None
        self._fallback: dict[str, list[dict]] = defaultdict(list)
        self._use_fallback = False

    async def _get_redis(self) -> Any:
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            client = aioredis.from_url(
                self._cfg.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await client.ping()
            self._redis = client
            log.info("memory.short_term.redis_connected")
        except Exception as exc:
            log.warning(
                "memory.short_term.redis_unavailable",
                error=str(exc),
                fallback="in-process dict",
            )
            self._use_fallback = True
            self._redis = None
        return self._redis

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def add_message(self, session_id: str, message: Message) -> None:
        payload = json.dumps({"role": message.role, "content": message.content})

        if self._use_fallback:
            self._fallback[session_id].append(json.loads(payload))
            # Keep window
            max_n = self._cfg.max_short_term_messages
            if len(self._fallback[session_id]) > max_n:
                self._fallback[session_id] = self._fallback[session_id][-max_n:]
            return

        redis = await self._get_redis()
        if redis is None:
            return

        key = self._session_key(session_id)
        pipe = redis.pipeline()
        pipe.rpush(key, payload)
        # Trim to window
        pipe.ltrim(key, -self._cfg.max_short_term_messages, -1)
        pipe.expire(key, self._cfg.redis_ttl_seconds)
        await pipe.execute()

    async def get_history(self, session_id: str) -> list[Message]:
        if self._use_fallback:
            return [
                Message(role=m["role"], content=m["content"])
                for m in self._fallback.get(session_id, [])
            ]

        redis = await self._get_redis()
        if redis is None:
            return []

        raw = await redis.lrange(self._session_key(session_id), 0, -1)
        messages: list[Message] = []
        for item in raw:
            try:
                d = json.loads(item)
                messages.append(Message(role=d["role"], content=d["content"]))
            except (json.JSONDecodeError, KeyError):
                continue
        return messages

    async def clear_session(self, session_id: str) -> None:
        if self._use_fallback:
            self._fallback.pop(session_id, None)
            return

        redis = await self._get_redis()
        if redis:
            await redis.delete(self._session_key(session_id))

    async def get_token_count(self, session_id: str) -> int:
        """Rough token estimate for context window management."""
        messages = await self.get_history(session_id)
        # ~4 chars per token heuristic
        return sum(len(m.content) // 4 for m in messages)

    async def set_metadata(self, session_id: str, key: str, value: str) -> None:
        """Store arbitrary session metadata (user info, flags, etc.)."""
        meta_key = f"session:{session_id}:meta"
        if self._use_fallback:
            return

        redis = await self._get_redis()
        if redis:
            await redis.hset(meta_key, key, value)
            await redis.expire(meta_key, self._cfg.redis_ttl_seconds)

    async def get_metadata(self, session_id: str, key: str) -> str | None:
        meta_key = f"session:{session_id}:meta"
        if self._use_fallback:
            return None

        redis = await self._get_redis()
        if redis:
            return await redis.hget(meta_key, key)
        return None
