"""Central configuration loaded from environment variables."""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    llm_model_heavy: str = "claude-opus-4-6"
    llm_model_standard: str = "claude-sonnet-4-6"
    llm_model_light: str = "claude-haiku-4-5-20251001"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_ttl_seconds: int = 3600

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/sqlite/support.db"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_secret_key: str = "change-me"
    api_debug: bool = False

    # Rate limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 500
    rate_limit_burst: int = 10

    # Cost control
    cost_budget_daily_usd: float = 50.0
    cost_alert_threshold_usd: float = 40.0
    max_tokens_per_request: int = 4096
    max_context_tokens: int = 100_000

    # Memory
    max_short_term_messages: int = 20
    long_term_search_top_k: int = 5
    embedding_dim: int = 1536

    # Observability
    log_level: str = "INFO"
    prometheus_port: int = 9090
    environment: Environment = Environment.DEVELOPMENT

    # Agent behaviour
    max_agent_iterations: int = 10
    agent_timeout_seconds: int = 60
    escalation_confidence_threshold: float = 0.4

    @field_validator("anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _strip(cls, v: str) -> str:
        return (v or "").strip()

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION

    def model_for(self, tier: Literal["heavy", "standard", "light"]) -> str:
        return {
            "heavy": self.llm_model_heavy,
            "standard": self.llm_model_standard,
            "light": self.llm_model_light,
        }[tier]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
