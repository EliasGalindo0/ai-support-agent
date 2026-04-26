"""
Rate limiting: per-IP and per-user limits using a token bucket algorithm.

Two layers:
1. API-level: slowapi middleware on FastAPI (per IP).
2. Application-level: per-session / per-user limits with Redis.

Token bucket parameters:
- rate: tokens added per second.
- capacity: maximum burst size.
- cost: tokens consumed per request (default 1).

This prevents:
- Runaway automation hammering the LLM API.
- Cost spirals from a single user.
- API abuse in multi-tenant SaaS.
"""
from __future__ import annotations

import time
from typing import Optional

import structlog

from src.config import get_settings

log = structlog.get_logger(__name__)


class TokenBucket:
    """
    In-process token bucket for rate limiting.

    For distributed rate limiting (multiple API replicas), use the Redis
    implementation below instead.
    """

    def __init__(self, rate: float, capacity: float) -> None:
        self.rate = rate        # tokens per second
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._last_refill = now

    def consume(self, tokens: float = 1.0) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available(self) -> float:
        self._refill()
        return self._tokens


class RedisRateLimiter:
    """
    Distributed token bucket using Redis for multi-instance deployments.

    Uses a sliding window counter for simplicity and correctness.
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._redis = None

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
        except Exception as exc:
            log.warning("rate_limiter.redis_unavailable", error=str(exc))
            self._redis = None
        return self._redis

    async def is_allowed(
        self,
        identifier: str,       # e.g. user_id or IP
        window_seconds: int = 60,
        limit: int | None = None,
    ) -> tuple[bool, int]:
        """
        Check if a request is within rate limits.

        Returns (allowed, remaining_requests).
        """
        cfg = self._cfg
        max_requests = limit or cfg.rate_limit_per_minute
        redis = await self._get_redis()

        if redis is None:
            # No Redis: allow everything (fail open)
            return True, max_requests

        key = f"rate:{identifier}:{int(time.time()) // window_seconds}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_seconds)
        results = await pipe.execute()
        count = results[0]

        allowed = count <= max_requests
        remaining = max(0, max_requests - count)

        if not allowed:
            log.warning(
                "rate_limiter.exceeded",
                identifier=identifier,
                count=count,
                limit=max_requests,
            )

        return allowed, remaining


class RateLimitMiddleware:
    """
    FastAPI middleware factory for per-IP rate limiting.

    Usage in main.py::

        from src.services.rate_limiter import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware.build())
    """

    @staticmethod
    def build():
        from slowapi import Limiter
        from slowapi.util import get_remote_address
        cfg = get_settings()
        return Limiter(
            key_func=get_remote_address,
            default_limits=[
                f"{cfg.rate_limit_per_minute}/minute",
                f"{cfg.rate_limit_per_hour}/hour",
            ],
        )
