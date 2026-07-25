from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from orchestrator.pipeline import TradingPipeline
from schemas.execution import Order, OrderType, PortfolioState
from schemas.rewards import BacktestResult, RewardSignal
from schemas.risk import RiskAssessment, RiskCheckResult
from schemas.strategy import Position, PositionAction, RiskMetrics, Strategy


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _mk_strategy(strategy_id: str = "s-1", size: float = 10.0) -> Strategy:
    return Strategy(
        strategy_id=strategy_id,
        market_regime="bull",
        positions=[
            Position(
                asset="AAPL",
                action=PositionAction.LONG,
                size_pct=size,
                entry_price_target=None,
                stop_loss_pct=3.0,
                take_profit_pct=8.0,
                time_horizon_days=10,
                confidence=0.7,
            )
        ],
        rationale="pipeline-test",
        risk_metrics=RiskMetrics(total_exposure_pct=size),
    )


def _mk_portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime.utcnow(),
        cash=100_000.0,
        total_value=100_000.0,
        positions={},
        daily_pnl=0.0,
        total_pnl=0.0,
    )


def _mk_reward(strategy_id: str) -> RewardSignal:
    result = BacktestResult(
        strategy_id=strategy_id,
        backtest_start=datetime.utcnow() - timedelta(days=20),
        backtest_end=datetime.utcnow(),
        total_return=0.1,
        sharpe_ratio=1.2,
        max_drawdown=0.05,
        win_rate=0.6,
        total_trades=5,
        avg_trade_return=0.02,
        volatility=0.1,
    )
    return RewardSignal(strategy_id=strategy_id, terminal_reward=0.6, component_rewards={"sharpe": 0.5}, backtest_result=result)


def _mk_pipeline_stub() -> TradingPipeline:
    p = TradingPipeline.__new__(TradingPipeline)
    p._watchlist = ["AAPL", "MSFT"]
    p._pending = {}
    p._tasks = []
    p._eval_tasks = {}
    p.settings = type("S", (), {"paper_trading": True})()
    p.memory = type("M", (), {})()
    p.memory.working = type("W", (), {})()
    p.memory.working.get = AsyncMock(return_value=None)
    p.memory.working.set = AsyncMock(return_value=None)
    p.memory.episodic = type("E", (), {})()
    p.memory.episodic._session_factory = None
    p.memory.record_episode = AsyncMock(return_value="ep-1")
    p.agent1 = type("A1", (), {})()
    p.agent1.run_news_cycle = AsyncMock(return_value=None)
    p.agent2 = type("A2", (), {})()
    p.agent2.run_market_update = AsyncMock(return_value=None)
    p.agent3 = type("A3", (), {})()
    p.agent3.generate = AsyncMock(return_value=_mk_strategy("s-1"))
    p.agent3.refine_strategy = AsyncMock(return_value=_mk_strategy("s-2", size=8.0))
    p.agent4 = type("A4", (), {})()
    p.agent4.get_current_portfolio = AsyncMock(return_value=_mk_portfolio())
    p.agent4.execute_strategy = AsyncMock(
        return_value=[Order(order_id="o1", strategy_id="s-1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=10)]
    )
    p.agent4.open_orders = {}
    p.agent5 = type("A5", (), {})()
    p.agent5.on_strategy_approved = AsyncMock(return_value=None)
    p.agent5.on_strategy_rejected = AsyncMock(return_value=None)
    p.agent6 = type("A6", (), {})()
    p.agent6.evaluate_strategy = AsyncMock(return_value=_mk_reward("s-1"))
    p.risk = type("R", (), {})()
    p.risk.assess = AsyncMock(return_value=RiskAssessment(result=RiskCheckResult.PASS, summary="ok"))
    return p


@pytest.mark.asyncio
async def test_generate_strategy_full_flow():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline.request_strategy"""
    p = _mk_pipeline_stub()
    out = await p.request_strategy()
    assert out["status"] == "generated"
    assert p.agent1.run_news_cycle.await_count == 1
    assert p.agent2.run_market_update.await_count == 1
    assert "strategy" in out
    assert out["strategy"]["strategy_id"] == "s-1"


@pytest.mark.asyncio
async def test_generate_strategy_refreshes_stale_data():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline._ensure_fresh_context"""
    p = _mk_pipeline_stub()
    stale_obj = type("Obj", (), {"timestamp": datetime.utcnow() - timedelta(hours=1)})()
    p.memory.working.get = AsyncMock(side_effect=[stale_obj, stale_obj, _mk_portfolio()])
    await p._ensure_fresh_context()
    assert p.agent1.run_news_cycle.await_count == 1
    assert p.agent2.run_market_update.await_count == 1


@pytest.mark.asyncio
async def test_user_approve_without_modification():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline.user_approve"""
    p = _mk_pipeline_stub()
    s = _mk_strategy("s-approve")
    p._pending[s.strategy_id] = {"strategy": s, "reward": _mk_reward(s.strategy_id)}
    out = await p.user_approve(s.strategy_id)
    assert "risk_check" in out and "execution" in out
    assert p.risk.assess.await_count == 1
    assert p.agent4.execute_strategy.await_count == 1
    assert p.agent5.on_strategy_approved.await_count == 1
    assert p.memory.record_episode.await_count == 1


@pytest.mark.asyncio
async def test_user_approve_risk_fail_blocks_execution():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline.user_approve risk gate"""
    p = _mk_pipeline_stub()
    s = _mk_strategy("s-risk-fail")
    p._pending[s.strategy_id] = {"strategy": s, "reward": None}
    p.risk.assess = AsyncMock(return_value=RiskAssessment(result=RiskCheckResult.FAIL, summary="blocked"))
    out = await p.user_approve(s.strategy_id)
    assert out["execution"] is None
    assert p.agent4.execute_strategy.await_count == 0, _diag(
        "orchestrator.pipeline.TradingPipeline.user_approve",
        "risk fail",
        "no execution",
        p.agent4.execute_strategy.await_count,
        "SAFETY_CRITICAL_DEFAULT",
    )


@pytest.mark.asyncio
async def test_user_reject():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline.user_reject"""
    p = _mk_pipeline_stub()
    s = _mk_strategy("s-reject")
    p._pending[s.strategy_id] = {"strategy": s, "reward": None}
    out = await p.user_reject(s.strategy_id, reason="Too aggressive")
    assert out["status"] == "rejected"
    assert p.agent5.on_strategy_rejected.await_count == 1
    assert s.strategy_id not in p._pending


@pytest.mark.asyncio
async def test_user_request_refinement():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline.user_request_refinement"""
    p = _mk_pipeline_stub()
    s = _mk_strategy("s-refine")
    p._pending[s.strategy_id] = {"strategy": s, "reward": None}
    out = await p.user_request_refinement(s.strategy_id, feedback="Less tech exposure")
    assert out["status"] == "generated"
    assert out["strategy"]["strategy_id"] == "s-2"
    assert p.agent3.refine_strategy.await_count == 1


@pytest.mark.asyncio
async def test_get_system_status():
    """FUNCTION TESTED: orchestrator.pipeline.TradingPipeline.get_system_status"""
    p = _mk_pipeline_stub()
    fresh_obj = type("Obj", (), {"timestamp": datetime.utcnow()})()
    p.memory.working.get = AsyncMock(side_effect=[fresh_obj, fresh_obj])
    p.agent4.open_orders = {"o1": object(), "o2": object()}
    p._pending = {"s1": {}, "s2": {}}
    status = await p.get_system_status()
    assert status["news_fresh"] is True
    assert status["market_fresh"] is True
    assert status["open_orders"] == 2
    assert status["pending_strategies"] == 2
