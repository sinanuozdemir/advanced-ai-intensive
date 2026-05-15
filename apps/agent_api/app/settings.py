"""Env-driven config. Read once at import, mutated only in tests via monkeypatch."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_DEFAULT_RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


class Settings(BaseSettings):
    """Runtime configuration for the agent_api service.

    All fields are env-overridable. ``Settings()`` is called by FastAPI at
    boot; tests construct their own instance with explicit kwargs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str | None = Field(default=None, alias="OPENROUTER_API_KEY")
    serpapi_api_key: str | None = Field(default=None, alias="SERPAPI_API_KEY")
    firecrawl_api_key: str | None = Field(default=None, alias="FIRECRAWL_API_KEY")

    plan_model: str = Field(default="openai/gpt-5.4-mini", alias="PLAN_MODEL")
    research_model: str = Field(default="openai/gpt-5.4-mini", alias="RESEARCH_MODEL")
    reflect_model: str = Field(default="openai/gpt-5.4-mini", alias="REFLECT_MODEL")
    artifact_model: str = Field(default="anthropic/claude-opus-4.7", alias="ARTIFACT_MODEL")

    max_iterations: int = Field(default=3, alias="MAX_ITERATIONS")

    artifacts_db_path: Path = Field(
        default=_DEFAULT_RUNTIME / "artifacts.db",
        alias="ARTIFACTS_DB_PATH",
    )
    log_file_path: Path = Field(
        default=_DEFAULT_RUNTIME / "agent_api.log",
        alias="LOG_FILE_PATH",
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def runtime_dir(self) -> Path:
        return self.artifacts_db_path.parent


_settings: Settings | None = None


def get_settings() -> Settings:
    """Module-level singleton. Tests can override via ``set_settings``."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.runtime_dir().mkdir(parents=True, exist_ok=True)
        _settings.log_file_path.parent.mkdir(parents=True, exist_ok=True)
    return _settings


def set_settings(s: Settings) -> None:
    """Used by tests to inject a temp-dir-backed Settings."""
    global _settings
    _settings = s
    s.runtime_dir().mkdir(parents=True, exist_ok=True)
    s.log_file_path.parent.mkdir(parents=True, exist_ok=True)
