"""Pytest configuration and shared fixtures."""
import asyncio
import pytest

# Use a single event loop for all async tests
@pytest.fixture(scope="session")
def event_loop():
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()
