from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from agents.agent6_backtest import Agent6Backtest
from config.settings import Settings
from memory.working import WorkingMemory
from schemas.rewards import BacktestResult
from schemas.strategy import Position, PositionAction, RiskMetrics, Strategy
from tools import backtest_tools


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _make_df(prices: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    rows = []
    for p in prices:
        rows.append({"open": p, "high": p + 1, "low": p - 1, "close": p, "volume": 1_000_000})
    return pd.DataFrame(rows, index=idx)


@pytest.fixture
def simple_uptrend_data():
    return {"AAPL": _make_df([100 + i for i in range(50)])}


@pytest.fixture
def simple_downtrend_data():
    return {"AAPL": _make_df([100 - i for i in range(50)])}


@pytest.fixture
def sideways_data():
    vals = [100 + ((-1) ** i) * 2 for i in range(50)]
    return {"AAPL": _make_df(vals)}


@pytest.fixture
def stop_loss_trigger_data():
    prices = [100, 99, 90, 92, 95, 98, 102, 105, 108, 110]
    return {"AAPL": _make_df(prices)}


@pytest.fixture
def take_profit_trigger_data():
    prices = [100, 103, 106, 109, 115, 110, 100, 90, 85, 80]
    return {"AAPL": _make_df(prices)}


@pytest.fixture
def strategy_long_100():
    return Strategy(
        market_regime="bull",
        positions=[
            Position(
                asset="AAPL",
                action=PositionAction.LONG,
                size_pct=100,
                entry_price_target=None,
                stop_loss_pct=50,
                take_profit_pct=None,
                time_horizon_days=30,
                confidence=0.8,
            )
        ],
        rationale="test",
        risk_metrics=RiskMetrics(total_exposure_pct=100),
    )


@pytest.mark.asyncio
async def test_backtest_long_position_uptrend(strategy_long_100: Strategy, simple_uptrend_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest"""
    result = await backtest_tools.run_backtest(strategy_long_100, simple_uptrend_data, commission_pct=0.001)
    assert 0.25 <= result.total_return <= 0.35, _diag(
        "tools.backtest_tools.run_backtest",
        "long uptrend 30-day horizon",
        "about 30% net return",
        result.total_return,
        "WRONG_CALCULATION",
    )
    assert result.total_trades == 1
    assert result.win_rate == 1.0


@pytest.mark.asyncio
async def test_backtest_short_position_downtrend(strategy_long_100: Strategy, simple_downtrend_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest short PnL"""
    short = strategy_long_100.model_copy(
        update={"positions": [strategy_long_100.positions[0].model_copy(update={"action": PositionAction.SHORT})]}
    )
    result = await backtest_tools.run_backtest(short, simple_downtrend_data)
    assert result.total_return > 0, _diag(
        "tools.backtest_tools.run_backtest",
        "short in downtrend",
        "positive return",
        result.total_return,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_backtest_stop_loss_triggered(strategy_long_100: Strategy, stop_loss_trigger_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest stop-loss logic"""
    s = strategy_long_100.model_copy(
        update={
            "positions": [
                strategy_long_100.positions[0].model_copy(update={"stop_loss_pct": 5, "take_profit_pct": None, "time_horizon_days": 9})
            ]
        }
    )
    result = await backtest_tools.run_backtest(s, stop_loss_trigger_data)
    assert result.total_return < 0, _diag(
        "tools.backtest_tools.run_backtest",
        "stop-loss scenario",
        "negative return near stop level",
        result.total_return,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_backtest_take_profit_triggered(strategy_long_100: Strategy, take_profit_trigger_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest take-profit logic"""
    s = strategy_long_100.model_copy(
        update={
            "positions": [
                strategy_long_100.positions[0].model_copy(update={"take_profit_pct": 10, "stop_loss_pct": 50, "time_horizon_days": 9})
            ]
        }
    )
    result = await backtest_tools.run_backtest(s, take_profit_trigger_data)
    assert result.total_return > 0, _diag(
        "tools.backtest_tools.run_backtest",
        "take-profit scenario",
        "positive return near target",
        result.total_return,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_backtest_time_horizon_expiry(strategy_long_100: Strategy, sideways_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest time horizon exit"""
    s = strategy_long_100.model_copy(
        update={"positions": [strategy_long_100.positions[0].model_copy(update={"time_horizon_days": 10, "stop_loss_pct": 99})]}
    )
    result = await backtest_tools.run_backtest(s, sideways_data, commission_pct=0)
    assert abs(result.total_return) < 0.1


@pytest.mark.asyncio
async def test_backtest_limit_entry_price(strategy_long_100: Strategy):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest limit entry behavior"""
    data = {"AAPL": _make_df([100, 100, 100, 95, 94, 96, 98])}
    s = strategy_long_100.model_copy(
        update={"positions": [strategy_long_100.positions[0].model_copy(update={"entry_price_target": 95, "time_horizon_days": 6})]}
    )
    result = await backtest_tools.run_backtest(s, data, commission_pct=0)
    assert result.total_return >= 0.0


@pytest.mark.asyncio
async def test_backtest_market_entry_price(strategy_long_100: Strategy):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest market entry uses next open"""
    data = {"AAPL": _make_df([100, 101, 102, 103])}
    s = strategy_long_100.model_copy(
        update={"positions": [strategy_long_100.positions[0].model_copy(update={"entry_price_target": None, "time_horizon_days": 3})]}
    )
    result = await backtest_tools.run_backtest(s, data, commission_pct=0)
    # Entry should be df.iloc[1].open == 101, exit at day3 close==103 -> ~1.98%
    assert 0.015 <= result.total_return <= 0.025, _diag(
        "tools.backtest_tools.run_backtest",
        "market entry",
        "about 1.98%",
        result.total_return,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_backtest_multiple_positions_allocation():
    """FUNCTION TESTED: tools.backtest_tools.run_backtest multi-position aggregation"""
    data = {"AAPL": _make_df([100, 110, 120]), "MSFT": _make_df([100, 100, 100]), "TSLA": _make_df([100, 90, 80])}
    s = Strategy(
        market_regime="bull",
        positions=[
            Position(asset="AAPL", action=PositionAction.LONG, size_pct=30, entry_price_target=None, stop_loss_pct=50, take_profit_pct=None, time_horizon_days=2, confidence=0.7),
            Position(asset="MSFT", action=PositionAction.LONG, size_pct=30, entry_price_target=None, stop_loss_pct=50, take_profit_pct=None, time_horizon_days=2, confidence=0.7),
            Position(asset="TSLA", action=PositionAction.LONG, size_pct=40, entry_price_target=None, stop_loss_pct=50, take_profit_pct=None, time_horizon_days=2, confidence=0.7),
        ],
        rationale="multi",
        risk_metrics=RiskMetrics(total_exposure_pct=100),
    )
    result = await backtest_tools.run_backtest(s, data, commission_pct=0)
    assert result.total_trades == 3


@pytest.mark.asyncio
async def test_backtest_missing_ticker_data_partial(strategy_long_100: Strategy):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest missing data handling"""
    data = {"MSFT": _make_df([100, 101, 102])}
    result = await backtest_tools.run_backtest(strategy_long_100, data)
    assert result.partial_data is True
    assert result.total_trades == 0


@pytest.mark.asyncio
async def test_backtest_close_reduce_actions_skipped(strategy_long_100: Strategy, simple_uptrend_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest CLOSE/REDUCE skip"""
    close_only = strategy_long_100.model_copy(
        update={"positions": [strategy_long_100.positions[0].model_copy(update={"action": PositionAction.CLOSE})]}
    )
    result = await backtest_tools.run_backtest(close_only, simple_uptrend_data)
    assert result.total_trades == 0


@pytest.mark.asyncio
async def test_backtest_commission_deduction(strategy_long_100: Strategy, simple_uptrend_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest commission handling"""
    no_fee = await backtest_tools.run_backtest(strategy_long_100, simple_uptrend_data, commission_pct=0)
    high_fee = await backtest_tools.run_backtest(strategy_long_100, simple_uptrend_data, commission_pct=0.01)
    assert high_fee.total_return < no_fee.total_return, _diag(
        "tools.backtest_tools.run_backtest",
        "commission compare",
        "higher commission => lower return",
        (no_fee.total_return, high_fee.total_return),
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_compute_reward_all_positive(valid_backtest_result: BacktestResult):
    """FUNCTION TESTED: tools.backtest_tools.compute_reward"""
    settings = Settings(_env_file=None, reward_weight_sharpe=0.5, reward_weight_drawdown=0.3, reward_weight_winrate=0.2)
    result = valid_backtest_result.model_copy(update={"sharpe_ratio": 2.0, "max_drawdown": 0.05, "win_rate": 0.65})
    reward = await backtest_tools.compute_reward(result, settings)
    assert abs(reward.terminal_reward - 0.7335) <= 0.02, _diag(
        "tools.backtest_tools.compute_reward",
        {"sharpe": 2.0, "dd": 0.05, "win": 0.65},
        0.7335,
        reward.terminal_reward,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_compute_reward_edge_cases(valid_backtest_result: BacktestResult):
    """FUNCTION TESTED: tools.backtest_tools.compute_reward bounds"""
    settings = Settings(_env_file=None)
    high = await backtest_tools.compute_reward(valid_backtest_result.model_copy(update={"sharpe_ratio": 10.0, "max_drawdown": 0.0, "win_rate": 1.0}), settings)
    low = await backtest_tools.compute_reward(valid_backtest_result.model_copy(update={"sharpe_ratio": -5.0, "max_drawdown": 0.5, "win_rate": 0.0}), settings)
    assert -1.0 <= high.terminal_reward <= 1.0
    assert -1.0 <= low.terminal_reward <= 1.0
    assert high.component_rewards["sharpe"] == 1.0
    assert low.component_rewards["sharpe"] == 0.0


@pytest.mark.asyncio
async def test_compute_reward_zero_trades(valid_backtest_result: BacktestResult):
    """FUNCTION TESTED: tools.backtest_tools.compute_reward zero-trade handling"""
    settings = Settings(_env_file=None)
    result = valid_backtest_result.model_copy(update={"total_trades": 0, "sharpe_ratio": None, "win_rate": 0.0})
    reward = await backtest_tools.compute_reward(result, settings)
    assert not np.isnan(reward.terminal_reward)


@pytest.mark.asyncio
async def test_walk_forward_analysis_windows(strategy_long_100: Strategy):
    """FUNCTION TESTED: tools.backtest_tools.walk_forward_analysis"""
    data = {"AAPL": _make_df([100 + i for i in range(120)])}
    settings = Settings(_env_file=None)
    out = await backtest_tools.walk_forward_analysis(
        strategy_long_100,
        data,
        settings,
        total_lookback_days=120,
        window_size_days=20,
        step_days=10,
    )
    assert len(out["results"]) == 11, _diag(
        "tools.backtest_tools.walk_forward_analysis",
        "120/20 step10",
        11,
        len(out["results"]),
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_backtest_deterministic(strategy_long_100: Strategy, simple_uptrend_data):
    """FUNCTION TESTED: tools.backtest_tools.run_backtest determinism"""
    r1 = await backtest_tools.run_backtest(strategy_long_100, simple_uptrend_data)
    r2 = await backtest_tools.run_backtest(strategy_long_100, simple_uptrend_data)
    assert r1.model_dump(mode="json") == r2.model_dump(mode="json"), _diag(
        "tools.backtest_tools.run_backtest",
        "same inputs twice",
        "identical results",
        {"r1": r1.model_dump(mode="json"), "r2": r2.model_dump(mode="json")},
        "STATE_LEAKAGE",
    )


class _MemoryStub:
    def __init__(self):
        self.working = WorkingMemory(max_items=100)


class _DummyAgent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_agent6_evaluate_strategy(monkeypatch: pytest.MonkeyPatch, strategy_long_100: Strategy):
    """FUNCTION TESTED: agents.agent6_backtest.Agent6Backtest.evaluate_strategy"""
    monkeypatch.setattr("agents.agent6_backtest.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent6_backtest.build_agno_model", lambda s: object())

    class _Agent2:
        async def get_backtest_data(self, tickers, start, end):
            return {"AAPL": _make_df([100 + i for i in range(80)])}

    agent = Agent6Backtest(Settings(_env_file=None), _MemoryStub(), _Agent2())
    reward = await agent.evaluate_strategy(strategy_long_100)
    assert -1 <= reward.terminal_reward <= 1
    stored = await agent.memory.working.get(f"backtest:{strategy_long_100.strategy_id}")
    assert stored is not None


@pytest.mark.asyncio
async def test_agent6_batch_evaluate(monkeypatch: pytest.MonkeyPatch, strategy_long_100: Strategy):
    """FUNCTION TESTED: agents.agent6_backtest.Agent6Backtest.batch_evaluate"""
    monkeypatch.setattr("agents.agent6_backtest.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent6_backtest.build_agno_model", lambda s: object())

    class _Agent2:
        async def get_backtest_data(self, tickers, start, end):
            return {"AAPL": _make_df([100 + i for i in range(80)])}

    agent = Agent6Backtest(Settings(_env_file=None), _MemoryStub(), _Agent2())
    strategies = [strategy_long_100.model_copy(update={"strategy_id": f"s{i}"}) for i in range(5)]
    out = await agent.batch_evaluate(strategies)
    assert len(out) == 5
