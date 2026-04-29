"""
Response caching layer.

Two-level cache:
1. Exact-match cache: identical messages get cached responses (e.g. FAQs).
2. Semantic cache: similar messages reuse cached responses (future: embed queries).

Cache key strategy:
- SHA-256 of (agent_type + normalised_message).
- TTL: 5 minutes for customer interactions, 1 hour for KB queries.

Important: we only cache SAFE responses (no PII, no financial transactions).
Tool-calling responses are NEVER cached — they reflect live state.
"""
from __future__ import annotations

import hashlib

import structlog

from src.config import get_settings

log = structlog.get_logger(__name__)

_CACHEABLE_AGENTS = {"knowledge_base"}  # Only KB responses are safe to cache
_KB_CACHE_TTL = 3600                            # 1 hour for KB
_DEFAULT_CACHE_TTL = 300                        # 5 min for others


def _cache_key(message: str, namespace: str = "kb") -> str:
    """
    Cache key based on message content + namespace only.
    Not agent-type-scoped: callers that don't know the agent type up front
    (e.g. the chat endpoint before routing) can still do lookups.
    Namespace separates KB responses from other response types.
    """
    normalised = " ".join(message.lower().split())
    payload = f"{namespace}:{normalised}"
    return "cache:" + hashlib.sha256(payload.encode()).hexdigest()


def _invalidation_tag_key(namespace: str) -> str:
    """Set key tracking all cache keys for a given namespace (for bulk invalidation)."""
    return f"cache_tag:{namespace}"


class ResponseCache:
    """Redis-backed response cache with graceful fallback."""

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._redis = None
        self._local: dict[str, str] = {}  # in-process fallback

    async def _get_redis(self):
        if self._redis is not None:
            return self._redis
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._cfg.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
        except Exception:
            self._redis = None
        return self._redis

    def _is_cacheable(self, agent_type: str, response_text: str) -> bool:
        """Don't cache responses that contain dynamic data."""
        if agent_type not in _CACHEABLE_AGENTS:
            return False
        # Don't cache responses mentioning specific order IDs, prices, names
        dynamic_markers = ["ORD-", "TKT-", "REF-", "ESC-", "$", "your order"]
        lower = response_text.lower()
        return not any(m.lower() in lower for m in dynamic_markers)

    async def get(self, message: str, namespace: str = "kb") -> str | None:
        """
        Look up a cached response for a message.

        Uses namespace-scoped keys so callers do not need to know the agent type
        up front (the chat endpoint calls this before routing).
        """
        key = _cache_key(message, namespace)
        redis = await self._get_redis()

        if redis:
            try:
                cached = await redis.get(key)
                if cached:
                    log.info("cache.hit", namespace=namespace, key=key[:16])
                    return cached
            except Exception as exc:
                log.warning(
                    "cache.redis_get_failed", namespace=namespace, error=str(exc)
                )
        elif key in self._local:
            log.info("cache.hit.local", namespace=namespace)
            return self._local[key]

        return None

    async def set(
        self,
        agent_type: str,
        message: str,
        response: str,
        ttl: int | None = None,
    ) -> None:
        if not self._is_cacheable(agent_type, response):
            return

        namespace = "kb"  # All cacheable responses currently belong to KB namespace
        key = _cache_key(message, namespace)
        ttl = ttl or _KB_CACHE_TTL

        redis = await self._get_redis()
        if redis:
            try:
                pipe = redis.pipeline()
                pipe.setex(key, ttl, response)
                # Track key in namespace tag set for O(1) bulk invalidation
                pipe.sadd(_invalidation_tag_key(namespace), key)
                pipe.expire(_invalidation_tag_key(namespace), ttl + 60)
                await pipe.execute()
                log.info(
                    "cache.set",
                    agent_type=agent_type,
                    namespace=namespace,
                    ttl=ttl,
                )
            except Exception as exc:
                log.warning(
                    "cache.redis_set_failed",
                    agent_type=agent_type,
                    namespace=namespace,
                    error=str(exc),
                )
                self._local[key] = response
        else:
            self._local[key] = response

    async def invalidate(self, namespace: str) -> int:
        """
        Invalidate all cache entries for a namespace (e.g. after KB document ingestion).

        Uses a Redis tag set to track keys — avoids KEYS glob scanning which
        would never match SHA-256 hashed keys by pattern.
        """
        redis = await self._get_redis()
        if not redis:
            return 0

        tag_key = _invalidation_tag_key(namespace)
        try:
            keys = await redis.smembers(tag_key)
        except Exception as exc:
            log.warning(
                "cache.redis_invalidate_read_failed",
                namespace=namespace,
                error=str(exc),
            )
            return 0
        if not keys:
            return 0

        try:
            pipe = redis.pipeline()
            for k in keys:
                pipe.delete(k)
            pipe.delete(tag_key)
            results = await pipe.execute()
            deleted = sum(1 for r in results[:-1] if r)
            log.info("cache.invalidated", namespace=namespace, deleted=deleted)
            return deleted
        except Exception as exc:
            log.warning(
                "cache.redis_invalidate_failed",
                namespace=namespace,
                error=str(exc),
            )
            return 0
