from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agents.agent4_executor import Agent4Executor
from config.settings import Settings
from memory.working import WorkingMemory
from schemas.execution import Order, OrderStatus, OrderType
from schemas.risk import RiskAssessment, RiskCheckResult
from schemas.strategy import Position, PositionAction, RiskMetrics, Strategy
from tools import execution_tools


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _make_strategy() -> Strategy:
    return Strategy(
        market_regime="bull",
        positions=[
            Position(
                asset="AAPL",
                action=PositionAction.LONG,
                size_pct=10,
                entry_price_target=None,
                stop_loss_pct=3,
                take_profit_pct=8,
                time_horizon_days=10,
                confidence=0.7,
            )
        ],
        rationale="exec-test",
        risk_metrics=RiskMetrics(total_exposure_pct=10),
    )


class _DummyMemory:
    def __init__(self):
        self.working = WorkingMemory(max_items=100)


class _DummyAgent:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_paper_engine_initial_state():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.__init__"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    assert engine.cash == 100000
    assert engine.positions == {}
    assert engine.trade_history == []
    zero = execution_tools.PaperTradingEngine(initial_capital=0)
    assert zero.cash == 0


@pytest.mark.asyncio
async def test_paper_engine_market_buy():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.submit_order buy"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    order = Order(
        order_id="o1",
        strategy_id="s1",
        asset="AAPL",
        side="buy",
        order_type=OrderType.MARKET,
        quantity=100,
    )
    out = await engine.submit_order(order, current_price=150)
    assert out.status == OrderStatus.FILLED
    assert abs(engine.cash - 85000) < 1e-9
    assert engine.positions["AAPL"]["quantity"] == 100
    assert engine.positions["AAPL"]["avg_cost"] == 150


@pytest.mark.asyncio
async def test_paper_engine_market_sell():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.submit_order sell"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    buy = Order(order_id="b1", strategy_id="s1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=100)
    await engine.submit_order(buy, current_price=150)
    sell = Order(order_id="s1", strategy_id="s1", asset="AAPL", side="sell", order_type=OrderType.MARKET, quantity=100)
    out = await engine.submit_order(sell, current_price=160)
    assert out.status == OrderStatus.FILLED
    assert abs(engine.cash - 101000) < 1e-9
    assert engine.positions["AAPL"]["quantity"] == 0


@pytest.mark.asyncio
async def test_paper_engine_partial_sell():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.submit_order partial sell"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    await engine.submit_order(
        Order(order_id="b1", strategy_id="s1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=100),
        current_price=150,
    )
    await engine.submit_order(
        Order(order_id="s1", strategy_id="s1", asset="AAPL", side="sell", order_type=OrderType.MARKET, quantity=50),
        current_price=160,
    )
    assert engine.positions["AAPL"]["quantity"] == 50


@pytest.mark.asyncio
async def test_paper_engine_insufficient_funds():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.submit_order insufficient funds"""
    engine = execution_tools.PaperTradingEngine(initial_capital=1000)
    order = Order(order_id="o1", strategy_id="s1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=100)
    out = await engine.submit_order(order, current_price=150)
    assert out.status == OrderStatus.REJECTED, _diag(
        "tools.execution_tools.PaperTradingEngine.submit_order",
        {"cash": 1000, "cost": 15000},
        "REJECTED",
        out.status.value,
        "SAFETY_CRITICAL_DEFAULT",
    )
    assert engine.cash == 1000
    assert engine.positions == {}


@pytest.mark.asyncio
async def test_paper_engine_portfolio_value():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.get_portfolio_value"""
    engine = execution_tools.PaperTradingEngine(initial_capital=50000)
    await engine.submit_order(
        Order(order_id="b1", strategy_id="s1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=100),
        current_price=150,
    )
    total = await engine.get_portfolio_value({"AAPL": 150})
    assert abs(total - 50000) < 1e-9


@pytest.mark.asyncio
async def test_paper_engine_portfolio_state():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.get_portfolio_state"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    await engine.submit_order(
        Order(order_id="b1", strategy_id="s1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=100),
        current_price=150,
    )
    state = await engine.get_portfolio_state({"AAPL": 160})
    assert state.positions["AAPL"]["unrealized_pnl"] == 1000
    assert abs(state.total_value - 101000) < 1e-9


@pytest.mark.asyncio
async def test_paper_engine_short_selling():
    """FUNCTION TESTED: tools.execution_tools.PaperTradingEngine.submit_order short flow"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    short = Order(order_id="s0", strategy_id="s1", asset="AAPL", side="sell", order_type=OrderType.MARKET, quantity=100)
    await engine.submit_order(short, current_price=150)
    cover = Order(order_id="c0", strategy_id="s1", asset="AAPL", side="buy", order_type=OrderType.MARKET, quantity=100)
    await engine.submit_order(cover, current_price=140)
    assert abs(engine.cash - 101000) < 1e-9


@pytest.mark.asyncio
async def test_translate_strategy_long_position():
    """FUNCTION TESTED: tools.execution_tools.translate_strategy_to_orders"""
    strategy = _make_strategy()
    orders = await execution_tools.translate_strategy_to_orders(strategy, portfolio_value=100000, prices={"AAPL": 150})
    assert len(orders) == 1
    order = orders[0]
    assert order.side == "buy"
    assert order.order_type == OrderType.MARKET
    assert 66.0 <= order.quantity <= 67.0


@pytest.mark.asyncio
async def test_translate_strategy_with_limit_entry():
    """FUNCTION TESTED: tools.execution_tools.translate_strategy_to_orders limit order"""
    strategy = _make_strategy().model_copy(
        update={"positions": [_make_strategy().positions[0].model_copy(update={"entry_price_target": 145.0})]}
    )
    orders = await execution_tools.translate_strategy_to_orders(strategy, portfolio_value=100000, prices={"AAPL": 150})
    assert orders[0].order_type == OrderType.LIMIT
    assert orders[0].limit_price == 145.0


@pytest.mark.asyncio
async def test_translate_strategy_close_position():
    """FUNCTION TESTED: tools.execution_tools.translate_strategy_to_orders close action"""
    strategy = _make_strategy().model_copy(
        update={"positions": [_make_strategy().positions[0].model_copy(update={"action": PositionAction.CLOSE, "size_pct": 100})]}
    )
    orders = await execution_tools.translate_strategy_to_orders(strategy, portfolio_value=100000, prices={"AAPL": 150})
    assert orders[0].side == "sell"


def test_paper_trading_default():
    """FUNCTION TESTED: config.settings.Settings.paper_trading default"""
    settings = Settings(_env_file=None)
    assert settings.paper_trading is True, _diag(
        "config.settings.Settings",
        "default paper_trading",
        True,
        settings.paper_trading,
        "SAFETY_CRITICAL_DEFAULT",
    )


@pytest.mark.asyncio
async def test_live_trading_blocked_without_config():
    """FUNCTION TESTED: tools.execution_tools.submit_orders_live safeguards"""
    settings = Settings(_env_file=None, paper_trading=True)
    with pytest.raises(RuntimeError):
        await execution_tools.submit_orders_live(settings, [])


@pytest.mark.asyncio
async def test_execute_strategy_rejects_failed_risk_check(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent4_executor.Agent4Executor.execute_strategy"""
    monkeypatch.setattr("agents.agent4_executor.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent4_executor.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent4_executor.fetch_price_data", AsyncMock(return_value={"AAPL": []}))
    agent = Agent4Executor(Settings(_env_file=None), _DummyMemory())
    strategy = _make_strategy()
    risk_fail = RiskAssessment(result=RiskCheckResult.FAIL, summary="blocked")
    with pytest.raises(RuntimeError):
        await agent.execute_strategy(strategy, risk_fail)


@pytest.mark.asyncio
async def test_execute_strategy_accepts_warn_risk_check(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent4_executor.Agent4Executor.execute_strategy"""
    monkeypatch.setattr("agents.agent4_executor.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent4_executor.build_agno_model", lambda s: object())

    async def fake_fetch_price_data(symbols, start_date, end_date):
        bar = SimpleNamespace(close=150)
        return {s: [bar] for s in symbols}

    monkeypatch.setattr("agents.agent4_executor.fetch_price_data", fake_fetch_price_data)
    agent = Agent4Executor(Settings(_env_file=None, paper_trading=True), _DummyMemory())
    strategy = _make_strategy()
    risk_warn = RiskAssessment(result=RiskCheckResult.WARN, summary="warn", warnings=["sector concentration"])
    orders = await agent.execute_strategy(strategy, risk_warn)
    assert len(orders) == 1
    assert orders[0].status in {OrderStatus.FILLED, OrderStatus.SUBMITTED}


@pytest.mark.asyncio
async def test_check_order_status_paper_limit_fill():
    """FUNCTION TESTED: tools.execution_tools.check_order_status_paper"""
    engine = execution_tools.PaperTradingEngine(initial_capital=100000)
    order = Order(
        order_id="l1",
        strategy_id="s1",
        asset="AAPL",
        side="buy",
        order_type=OrderType.LIMIT,
        quantity=10,
        limit_price=95,
        status=OrderStatus.PENDING,
    )
    pending = await execution_tools.check_order_status_paper(engine, order, current_price=100)
    assert pending.status == OrderStatus.SUBMITTED
    filled = await execution_tools.check_order_status_paper(engine, order, current_price=94)
    assert filled.status == OrderStatus.FILLED


@pytest.mark.asyncio
async def test_cancel_order():
    """FUNCTION TESTED: tools.execution_tools.cancel_order_paper"""
    order = Order(
        order_id="c1",
        strategy_id="s1",
        asset="AAPL",
        side="buy",
        order_type=OrderType.LIMIT,
        quantity=10,
        limit_price=95,
        status=OrderStatus.PENDING,
    )
    out = await execution_tools.cancel_order_paper(order)
    assert out.status == OrderStatus.CANCELLED
