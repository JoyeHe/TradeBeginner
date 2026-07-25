from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from config.settings import Settings


def _clear_ats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(Settings.model_fields):
        monkeypatch.delenv(f"ATS_{key.upper()}", raising=False)


def _diag(function_tested: str, env_name: str, input_value: Any, expected: Any, actual: Any, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | ENV={env_name} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUNCTION TESTED: config.settings.Settings.__init__"""
    _clear_ats_env(monkeypatch)
    injected = {
        "ATS_LLM_PROVIDER": "deepseek",
        "ATS_LLM_API_KEY": "sk-test",
        "ATS_LLM_MODEL": "deepseek-chat",
        "ATS_LLM_BASE_URL": "https://api.deepseek.com/v1",
        "ATS_LLM_TEMPERATURE": "0.55",
        "ATS_POSTGRES_URL": "postgresql+asyncpg://u:p@localhost:5432/t",
        "ATS_CHROMA_PERSIST_DIR": "/tmp/chroma",
        "ATS_NEWS_API_KEY": "news-key",
        "ATS_FINNHUB_API_KEY": "finn-key",
        "ATS_OPENBB_PAT": "openbb-pat",
        "ATS_BROKER_NAME": "alpaca",
        "ATS_BROKER_API_KEY": "alpaca-key",
        "ATS_BROKER_API_SECRET": "alpaca-secret",
        "ATS_BROKER_BASE_URL": "https://paper-api.alpaca.markets",
        "ATS_PAPER_TRADING": "false",
        "ATS_MAX_POSITION_SIZE_PCT": "9.5",
        "ATS_MAX_TOTAL_EXPOSURE_PCT": "95",
        "ATS_MAX_STOP_LOSS_PCT": "4.5",
        "ATS_MAX_CONCURRENT_POSITIONS": "7",
        "ATS_MAX_SECTOR_CONCENTRATION_PCT": "28.5",
        "ATS_MAX_PORTFOLIO_DRAWDOWN_PCT": "12.5",
        "ATS_MIN_AVG_VOLUME": "600000",
        "ATS_BACKTEST_WINDOW_DAYS": "30",
        "ATS_REWARD_WEIGHT_SHARPE": "0.6",
        "ATS_REWARD_WEIGHT_DRAWDOWN": "0.25",
        "ATS_REWARD_WEIGHT_WINRATE": "0.15",
        "ATS_API_HOST": "127.0.0.1",
        "ATS_API_PORT": "8080",
    }
    for k, v in injected.items():
        monkeypatch.setenv(k, v)

    s = Settings(_env_file=None)
    expected = {
        "llm_provider": "deepseek",
        "llm_api_key": "sk-test",
        "llm_model": "deepseek-chat",
        "llm_base_url": "https://api.deepseek.com/v1",
        "llm_temperature": 0.55,
        "postgres_url": "postgresql+asyncpg://u:p@localhost:5432/t",
        "chroma_persist_dir": "/tmp/chroma",
        "news_api_key": "news-key",
        "finnhub_api_key": "finn-key",
        "openbb_pat": "openbb-pat",
        "broker_name": "alpaca",
        "broker_api_key": "alpaca-key",
        "broker_api_secret": "alpaca-secret",
        "broker_base_url": "https://paper-api.alpaca.markets",
        "paper_trading": False,
        "max_position_size_pct": 9.5,
        "max_total_exposure_pct": 95.0,
        "max_stop_loss_pct": 4.5,
        "max_concurrent_positions": 7,
        "max_sector_concentration_pct": 28.5,
        "max_portfolio_drawdown_pct": 12.5,
        "min_avg_volume": 600000,
        "backtest_window_days": 30,
        "reward_weight_sharpe": 0.6,
        "reward_weight_drawdown": 0.25,
        "reward_weight_winrate": 0.15,
        "api_host": "127.0.0.1",
        "api_port": 8080,
    }
    for field, expected_value in expected.items():
        actual_value = getattr(s, field)
        assert actual_value == expected_value, _diag(
            "config.settings.Settings.__init__",
            f"ATS_{field.upper()}",
            injected.get(f"ATS_{field.upper()}"),
            expected_value,
            actual_value,
            "TYPE_COERCION_FAILURE",
        )


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUNCTION TESTED: config.settings.Settings.__init__"""
    _clear_ats_env(monkeypatch)
    s = Settings(_env_file=None)
    defaults = {
        name: field.default for name, field in Settings.model_fields.items() if field.default is not None
    }
    for field, expected in defaults.items():
        actual = getattr(s, field)
        assert actual == expected, _diag(
            "config.settings.Settings.__init__",
            f"ATS_{field.upper()}",
            None,
            expected,
            actual,
            "WRONG_DEFAULT",
        )
    assert s.paper_trading is True, _diag(
        "config.settings.Settings.__init__",
        "ATS_PAPER_TRADING",
        None,
        True,
        s.paper_trading,
        "SAFETY_CRITICAL_DEFAULT",
    )


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [("ATS_MAX_POSITION_SIZE_PCT", "not_a_number"), ("ATS_API_PORT", "abc")],
)
def test_settings_type_validation_rejects_invalid(
    monkeypatch: pytest.MonkeyPatch, env_name: str, env_value: str
) -> None:
    """FUNCTION TESTED: config.settings.Settings.__init__"""
    _clear_ats_env(monkeypatch)
    monkeypatch.setenv(env_name, env_value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_type_validation_accepts_valid_numeric_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUNCTION TESTED: config.settings.Settings.__init__"""
    _clear_ats_env(monkeypatch)
    monkeypatch.setenv("ATS_API_PORT", "8080")
    s = Settings(_env_file=None)
    assert s.api_port == 8080, _diag(
        "config.settings.Settings.__init__",
        "ATS_API_PORT",
        "8080",
        8080,
        s.api_port,
        "TYPE_COERCION_FAILURE",
    )


def test_settings_sensitive_defaults_are_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """FUNCTION TESTED: config.settings.Settings.__init__"""
    _clear_ats_env(monkeypatch)
    s = Settings(_env_file=None)
    sensitive_fields = ["llm_api_key", "broker_api_key", "broker_api_secret", "news_api_key", "finnhub_api_key"]
    for field in sensitive_fields:
        value = getattr(s, field)
        preview = value[:4] if value else ""
        assert value == "", _diag(
            "config.settings.Settings.__init__",
            f"ATS_{field.upper()}",
            None,
            "",
            preview,
            "SAFETY_CRITICAL_DEFAULT",
        )
