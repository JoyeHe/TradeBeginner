"""Tests for strategy library, alpha eval, and constraints."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import Settings
from memory.strategy_library_store import StrategyLibraryStore
from schemas.strategy import MarketRegime, Position, PositionAction, RiskMetrics, Strategy
from tools.alpha_eval import eval_alpha, flatten_indicator_context
from tools.strategy_constraints import apply_strategy_constraints, strategies_materially_different


def test_alpha_eval_basic():
    ctx = flatten_indicator_context(100.0, {"sma_20": 95.0, "momentum_roc_20": 3.0, "rsi_14": 55.0})
    assert eval_alpha("(close > sma_20) and (momentum_roc_20 > 0)", ctx) is True
    assert eval_alpha("(close < sma_20)", ctx) is False


def test_alpha_eval_rejects_unsafe():
    assert eval_alpha("__import__('os').system('x')", {"close": 1}) is False


@pytest.mark.asyncio
async def test_library_seed_bootstrap(tmp_path: Path):
    seed = Path("data/strategy_library/seed_strategies.json")
    store = StrategyLibraryStore(
        postgres_url="postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid",
        seed_path=str(seed),
        learned_path=str(tmp_path / "learned.json"),
    )
    await store.initialize()
    entries = store.list_entries(user_id="default", limit=100, include_global=True)
    seeds = [e for e in entries if e.user_id == "*"]
    assert len(seeds) >= 20
    assert any(e.template_id == "tpl_sma20_trend_long" for e in seeds)
    assert seeds[0].alphas
    await store.shutdown()


def test_constraints_scale_oversize():
    settings = Settings(_env_file=None, max_position_size_pct=15, max_total_exposure_pct=80, max_crypto_pct=3)
    s = Strategy(
        market_regime=MarketRegime.SIDEWAYS,
        positions=[
            Position(asset="SPY", action=PositionAction.LONG, size_pct=50, stop_loss_pct=4, take_profit_pct=6, time_horizon_days=10, confidence=0.5),
            Position(asset="BTC-USD", action=PositionAction.LONG, size_pct=10, stop_loss_pct=6, take_profit_pct=None, time_horizon_days=10, confidence=0.4),
        ],
        rationale="x",
        risk_metrics=RiskMetrics(total_exposure_pct=60),
    )
    out, notes = apply_strategy_constraints(s, settings, requested_tickers=["SPY", "QQQ"])
    assert out.positions[0].size_pct <= 15
    assert all(p.size_pct <= 3 for p in out.positions if p.asset == "BTC-USD")
    assert out.risk_metrics.total_exposure_pct <= 80 + 1e-6
    assert any("missing_requested_tickers" in n for n in notes)


def test_material_change_detects_size():
    a = Strategy(
        market_regime=MarketRegime.BULL,
        positions=[Position(asset="AAPL", action=PositionAction.LONG, size_pct=10, stop_loss_pct=4, take_profit_pct=6, time_horizon_days=10, confidence=0.5)],
        rationale="a",
        risk_metrics=RiskMetrics(total_exposure_pct=10),
    )
    b = a.model_copy(deep=True)
    b.positions[0].size_pct = 20
    assert strategies_materially_different(a, b) is True
    assert strategies_materially_different(a, a.model_copy(deep=True)) is False
