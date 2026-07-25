from __future__ import annotations

from datetime import datetime

import pytest

from config.settings import Settings
from risk.controller import RiskController
from schemas.execution import PortfolioState
from schemas.risk import RiskCheckResult
from schemas.strategy import Position, PositionAction, RiskMetrics, Strategy


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _make_strategy(positions: list[Position], regime: str = "bull") -> Strategy:
    exposure = sum(p.size_pct for p in positions if p.action not in (PositionAction.CLOSE, PositionAction.REDUCE))
    return Strategy(
        market_regime=regime,
        positions=positions,
        rationale="risk test",
        risk_metrics=RiskMetrics(total_exposure_pct=exposure),
    )


def _p(asset: str, size: float, action: PositionAction = PositionAction.LONG, stop: float = 3.0) -> Position:
    return Position(
        asset=asset,
        action=action,
        size_pct=size,
        entry_price_target=None,
        stop_loss_pct=stop,
        take_profit_pct=8.0,
        time_horizon_days=10,
        confidence=0.7,
    )


@pytest.fixture
def risk_controller() -> RiskController:
    settings = Settings(
        _env_file=None,
        max_position_size_pct=10.0,
        max_total_exposure_pct=100.0,
        max_stop_loss_pct=5.0,
        max_concurrent_positions=10,
        max_sector_concentration_pct=30.0,
        max_portfolio_drawdown_pct=15.0,
    )
    return RiskController(settings)


@pytest.fixture
def default_portfolio() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime.utcnow(),
        cash=100_000.0,
        total_value=100_000.0,
        positions={},
        daily_pnl=0.0,
        total_pnl=0.0,
    )


def test_check_position_size_over_limit(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_position_size"""
    s = _make_strategy([_p("AAPL", 15)])
    violations = risk_controller.check_position_size(s)
    assert len(violations) == 1
    v = violations[0]
    assert v.severity == "hard"
    assert v.current_value == 15
    assert v.limit_value == 10.0


def test_check_position_size_at_exact_limit(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_position_size"""
    s = _make_strategy([_p("AAPL", 10)])
    violations = risk_controller.check_position_size(s)
    assert violations == [], _diag(
        "risk.controller.RiskController.check_position_size",
        10,
        "no violation at exact limit",
        violations,
        "CONSTRAINT_NOT_ENFORCED",
    )


def test_check_position_size_barely_over(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_position_size"""
    s = _make_strategy([_p("AAPL", 10.01)])
    violations = risk_controller.check_position_size(s)
    assert len(violations) == 1


def test_check_total_exposure_over_limit(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_total_exposure"""
    s = _make_strategy([_p("AAPL", 40), _p("MSFT", 40), _p("NVDA", 40)])
    v = risk_controller.check_total_exposure(s)
    assert v is not None
    assert v.current_value == 120
    assert v.limit_value == 100.0


def test_check_stop_loss_mandatory(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_stop_losses"""
    s = _make_strategy([_p("AAPL", 9, stop=0)])
    violations = risk_controller.check_stop_losses(s)
    names = {v.rule_name for v in violations}
    assert "mandatory_stop_loss" in names


@pytest.mark.parametrize("stop,expect_violation", [(5.0, False), (4.99, False), (8.0, True)])
def test_check_stop_loss_too_wide(risk_controller: RiskController, stop: float, expect_violation: bool):
    """FUNCTION TESTED: risk.controller.RiskController.check_stop_losses"""
    s = _make_strategy([_p("AAPL", 9, stop=stop)])
    violations = risk_controller.check_stop_losses(s)
    has_wide = any(v.rule_name == "max_stop_loss_pct" for v in violations)
    assert has_wide == expect_violation


def test_check_max_concurrent_positions(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_concurrent_positions"""
    s = _make_strategy([_p(f"T{i}", 5) for i in range(12)])
    v = risk_controller.check_concurrent_positions(s)
    assert v is not None
    assert v.current_value == 12.0


def test_check_sector_concentration_soft_rule(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_sector_concentration"""
    s = _make_strategy([_p("AAPL", 15), _p("MSFT", 12), _p("GOOGL", 8)])
    violations = risk_controller.check_sector_concentration(s)
    assert len(violations) == 1
    assert violations[0].severity == "soft"


def test_sector_mapping_for_known_tickers(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController._sector"""
    assert risk_controller._sector("AAPL") == "Technology"
    assert risk_controller._sector("JPM") == "Financial"
    assert risk_controller._sector("XOM") == "Energy"
    assert risk_controller._sector("JNJ") == "Unknown"


def test_check_drawdown_circuit_breaker(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_drawdown_limit"""
    p_ok = PortfolioState(
        timestamp=datetime.utcnow(),
        cash=88_000,
        total_value=100_000,
        positions={},
        daily_pnl=0,
        total_pnl=-12_000,
    )
    p_bad = PortfolioState(
        timestamp=datetime.utcnow(),
        cash=84_000,
        total_value=100_000,
        positions={},
        daily_pnl=0,
        total_pnl=-16_000,
    )
    assert risk_controller.check_drawdown_limit(p_ok) is None
    assert risk_controller.check_drawdown_limit(p_bad) is not None


@pytest.mark.asyncio
async def test_evaluate_strategy_all_pass(risk_controller: RiskController, default_portfolio: PortfolioState):
    """FUNCTION TESTED: risk.controller.RiskController.assess"""
    s = _make_strategy([_p("AAPL", 9), _p("JPM", 8)])
    assessment = await risk_controller.assess(s, default_portfolio)
    assert assessment.result == RiskCheckResult.PASS
    assert assessment.violations == []


@pytest.mark.asyncio
async def test_evaluate_strategy_hard_fail(risk_controller: RiskController, default_portfolio: PortfolioState):
    """FUNCTION TESTED: risk.controller.RiskController.assess"""
    s = _make_strategy([_p("AAPL", 15)])
    assessment = await risk_controller.assess(s, default_portfolio)
    assert assessment.result == RiskCheckResult.FAIL
    assert any(v.severity == "hard" for v in assessment.violations)


@pytest.mark.asyncio
async def test_evaluate_strategy_soft_warn(risk_controller: RiskController, default_portfolio: PortfolioState):
    """FUNCTION TESTED: risk.controller.RiskController.assess"""
    s = _make_strategy([_p("AAPL", 12), _p("MSFT", 10), _p("GOOGL", 10)])
    assessment = await risk_controller.assess(s, default_portfolio)
    assert assessment.result == RiskCheckResult.FAIL, _diag(
        "risk.controller.RiskController.assess",
        "sector concentration with oversized position",
        "FAIL due to hard rule precedence",
        assessment.result.value,
        "CONSTRAINT_NOT_ENFORCED",
    )


@pytest.mark.asyncio
async def test_evaluate_strategy_mixed_violations(risk_controller: RiskController, default_portfolio: PortfolioState):
    """FUNCTION TESTED: risk.controller.RiskController.assess"""
    s = _make_strategy([_p("AAPL", 15), _p("MSFT", 10), _p("GOOGL", 10), _p("NVDA", 5)])
    assessment = await risk_controller.assess(s, default_portfolio)
    assert assessment.result == RiskCheckResult.FAIL
    assert any(v.rule_name == "max_position_size" for v in assessment.violations)
    assert any(v.rule_name == "max_sector_concentration" for v in assessment.violations)


@pytest.mark.asyncio
async def test_auto_adjust_oversized_position(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.auto_adjust"""
    s = _make_strategy([_p("AAPL", 15), _p("JPM", 5)])
    violations = risk_controller.check_position_size(s)
    adjusted = await risk_controller.auto_adjust(s, violations)
    assert adjusted.positions[0].size_pct == 10.0
    assert adjusted.positions[1].size_pct == 5.0


@pytest.mark.asyncio
async def test_auto_adjust_stop_loss(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.auto_adjust"""
    s = _make_strategy([_p("AAPL", 8, stop=8)])
    violations = risk_controller.check_stop_losses(s)
    adjusted = await risk_controller.auto_adjust(s, violations)
    assert adjusted.positions[0].stop_loss_pct == 5.0


@pytest.mark.asyncio
async def test_auto_adjust_does_not_fix_unfixable_count(risk_controller: RiskController, default_portfolio: PortfolioState):
    """FUNCTION TESTED: risk.controller.RiskController.auto_adjust limits"""
    s = _make_strategy([_p(f"T{i}", 5) for i in range(12)])
    assessment = await risk_controller.assess(s, default_portfolio)
    assert assessment.result == RiskCheckResult.FAIL
    assert any(v.rule_name == "max_concurrent_positions" for v in assessment.violations)


@pytest.mark.asyncio
async def test_evaluate_empty_strategy(risk_controller: RiskController, default_portfolio: PortfolioState):
    """FUNCTION TESTED: risk.controller.RiskController.assess empty strategy"""
    s = _make_strategy([])
    assessment = await risk_controller.assess(s, default_portfolio)
    assert assessment.result == RiskCheckResult.PASS


def test_evaluate_close_actions_not_counted_as_exposure(risk_controller: RiskController):
    """FUNCTION TESTED: risk.controller.RiskController.check_total_exposure close action handling"""
    s = _make_strategy([_p("AAPL", 100, action=PositionAction.CLOSE), _p("MSFT", 50, action=PositionAction.REDUCE)])
    v = risk_controller.check_total_exposure(s)
    assert v is None, _diag(
        "risk.controller.RiskController.check_total_exposure",
        "close/reduce only",
        "no exposure violation",
        v,
        "WRONG_CALCULATION",
    )
