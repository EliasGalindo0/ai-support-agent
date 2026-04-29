from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.services.cache import ResponseCache
from src.llm.client import LLMClient, Message


@pytest.mark.asyncio
async def test_cache_get_redis_failure_falls_back_to_none():
    cache = ResponseCache()

    class _BadRedis:
        async def get(self, _key: str):
            raise ConnectionError("redis down")

    with patch.object(
        cache, "_get_redis", new=AsyncMock(return_value=_BadRedis())
    ):
        assert await cache.get("hello", namespace="kb") is None


@pytest.mark.asyncio
async def test_cache_set_redis_failure_falls_back_to_local():
    cache = ResponseCache()

    class _BadRedis:
        def pipeline(self):
            raise ConnectionError("redis down")

    with patch.object(
        cache, "_get_redis", new=AsyncMock(return_value=_BadRedis())
    ):
        await cache.set(
            agent_type="knowledge_base",
            message="hello",
            response="world",
            ttl=60,
        )

    # Now simulate Redis unavailable entirely; should hit local cache
    with patch.object(cache, "_get_redis", new=AsyncMock(return_value=None)):
        assert await cache.get("hello", namespace="kb") == "world"


@pytest.mark.asyncio
async def test_llm_client_enforces_timeout():
    # Patch backend to a slow coroutine; ensure LLMClient wraps it with wait_for
    client = LLMClient()

    async def _slow_complete(**_kwargs):
        await asyncio.sleep(1.0)
        raise AssertionError("should have timed out before finishing")

    client._backend.complete = _slow_complete  # type: ignore[method-assign]

    with patch.object(client._cfg, "llm_timeout_seconds", 0.01):
        with pytest.raises(asyncio.TimeoutError):
            await client.complete(messages=[Message(role="user", content="hi")])

