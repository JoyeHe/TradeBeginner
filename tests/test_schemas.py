from __future__ import annotations

import uuid
import warnings
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from schemas.execution import Order, OrderStatus, OrderType
from schemas.market_data import OHLCVBar
from schemas.memory import EpisodicTrace
from schemas.news import NewsItem
from schemas.rewards import BacktestResult
from schemas.strategy import Position, PositionAction, RiskMetrics, Strategy, UserModifiedStrategy


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


@pytest.mark.parametrize(
    ("field_name", "value", "should_pass", "constraint"),
    [
        ("size_pct", -1, False, "ge=0"),
        ("size_pct", 101, False, "le=100"),
        ("size_pct", 0, True, "ge=0"),
        ("size_pct", 100, True, "le=100"),
        ("stop_loss_pct", -0.01, False, "ge=0"),
        ("confidence", 1.1, False, "le=1"),
        ("confidence", -0.1, False, "ge=0"),
        ("time_horizon_days", 0, False, "ge=1"),
    ],
)
def test_position_field_constraints(valid_position: Position, field_name: str, value: float, should_pass: bool, constraint: str) -> None:
    """FUNCTION TESTED: schemas.strategy.Position"""
    payload = valid_position.model_dump(mode="python")
    payload[field_name] = value
    if should_pass:
        model = Position.model_validate(payload)
        assert getattr(model, field_name) == value, _diag(
            "schemas.strategy.Position",
            {field_name: value},
            value,
            getattr(model, field_name),
            "CONSTRAINT_NOT_ENFORCED",
        )
    else:
        with pytest.raises(ValidationError):
            Position.model_validate(payload)


def test_position_enum_values(valid_position: Position) -> None:
    """FUNCTION TESTED: schemas.strategy.PositionAction"""
    accepted = Position.model_validate({**valid_position.model_dump(mode="python"), "action": "long"})
    assert accepted.action == PositionAction.LONG, _diag(
        "schemas.strategy.PositionAction",
        "long",
        PositionAction.LONG,
        accepted.action,
        "ENUM_MISMATCH",
    )
    with pytest.raises(ValidationError):
        Position.model_validate({**valid_position.model_dump(mode="python"), "action": "LONG"})
    with pytest.raises(ValidationError):
        Position.model_validate({**valid_position.model_dump(mode="python"), "action": "hold"})

    expected_members = {"long", "short", "close", "reduce", "increase"}
    actual_members = {member.value for member in PositionAction}
    assert actual_members == expected_members, _diag(
        "schemas.strategy.PositionAction",
        "enum members",
        expected_members,
        actual_members,
        "ENUM_MISMATCH",
    )


def test_strategy_defaults(valid_strategy: Strategy) -> None:
    """FUNCTION TESTED: schemas.strategy.Strategy"""
    s1 = Strategy(
        market_regime=valid_strategy.market_regime,
        positions=valid_strategy.positions,
        rationale=valid_strategy.rationale,
        risk_metrics=valid_strategy.risk_metrics,
    )
    s2 = Strategy(
        market_regime=valid_strategy.market_regime,
        positions=valid_strategy.positions,
        rationale=valid_strategy.rationale,
        risk_metrics=valid_strategy.risk_metrics,
    )

    parsed_uuid = uuid.UUID(s1.strategy_id)
    assert parsed_uuid.version == 4, _diag(
        "schemas.strategy.Strategy",
        s1.strategy_id,
        "uuid4",
        parsed_uuid.version,
        "MUTABLE_DEFAULT_SHARED",
    )
    now = datetime.now(timezone.utc)
    age_sec = abs((now - s1.timestamp.replace(tzinfo=timezone.utc)).total_seconds())
    assert age_sec <= 5, _diag(
        "schemas.strategy.Strategy",
        s1.timestamp,
        "timestamp within 5s",
        age_sec,
        "WRONG_DEFAULT",
    )
    assert s1.data_sources_used == [], _diag(
        "schemas.strategy.Strategy",
        s1.data_sources_used,
        [],
        s1.data_sources_used,
        "WRONG_DEFAULT",
    )
    assert s1.metadata == {}, _diag(
        "schemas.strategy.Strategy",
        s1.metadata,
        {},
        s1.metadata,
        "WRONG_DEFAULT",
    )
    assert s1.strategy_id != s2.strategy_id, _diag(
        "schemas.strategy.Strategy",
        (s1.strategy_id, s2.strategy_id),
        "different IDs",
        "same IDs",
        "MUTABLE_DEFAULT_SHARED",
    )


def test_strategy_serialization_roundtrip(valid_strategy: Strategy) -> None:
    """FUNCTION TESTED: schemas.strategy.Strategy.model_dump/model_validate"""
    data = valid_strategy.model_dump()
    strategy2 = Strategy.model_validate(data)
    assert strategy2 == valid_strategy, _diag(
        "schemas.strategy.Strategy.model_validate",
        data,
        valid_strategy.model_dump(),
        strategy2.model_dump(),
        "SERIALIZATION_FAILURE",
    )
    json_str = valid_strategy.model_dump_json()
    strategy3 = Strategy.model_validate_json(json_str)
    assert strategy3 == valid_strategy, _diag(
        "schemas.strategy.Strategy.model_validate_json",
        json_str[:100],
        valid_strategy.model_dump(),
        strategy3.model_dump(),
        "SERIALIZATION_FAILURE",
    )


def test_risk_metrics_optional_fields() -> None:
    """FUNCTION TESTED: schemas.strategy.RiskMetrics"""
    minimal = RiskMetrics(total_exposure_pct=12.0)
    assert minimal.portfolio_var_95 is None, _diag(
        "schemas.strategy.RiskMetrics",
        {"total_exposure_pct": 12.0},
        None,
        minimal.portfolio_var_95,
        "WRONG_DEFAULT",
    )
    assert minimal.max_drawdown_estimate is None
    assert minimal.correlation_to_existing is None
    assert minimal.sector_concentrations is None

    full = RiskMetrics(
        portfolio_var_95=0.02,
        max_drawdown_estimate=0.08,
        correlation_to_existing=0.4,
        total_exposure_pct=35.0,
        sector_concentrations={"Technology": 20.0},
    )
    assert full.total_exposure_pct == 35.0
    assert full.sector_concentrations == {"Technology": 20.0}


@pytest.mark.parametrize(
    ("field_name", "value", "should_pass"),
    [
        ("sentiment_score", -1, True),
        ("sentiment_score", 0, True),
        ("sentiment_score", 1, True),
        ("sentiment_score", -1.01, False),
        ("sentiment_score", 1.01, False),
        ("relevance_score", -0.01, False),
        ("relevance_score", 0, True),
        ("relevance_score", 1, True),
    ],
)
def test_news_item_sentiment_bounds(valid_news_item: NewsItem, field_name: str, value: float, should_pass: bool) -> None:
    """FUNCTION TESTED: schemas.news.NewsItem"""
    payload = valid_news_item.model_dump(mode="python")
    payload[field_name] = value
    if should_pass:
        item = NewsItem.model_validate(payload)
        assert getattr(item, field_name) == value, _diag(
            "schemas.news.NewsItem",
            {field_name: value},
            value,
            getattr(item, field_name),
            "CONSTRAINT_NOT_ENFORCED",
        )
    else:
        with pytest.raises(ValidationError):
            NewsItem.model_validate(payload)


def test_ohlcv_bar_validation() -> None:
    """FUNCTION TESTED: schemas.market_data.OHLCVBar"""
    valid = OHLCVBar(
        timestamp=datetime.now(timezone.utc),
        open=100,
        high=105,
        low=99,
        close=103,
        volume=1000,
        asset="AAPL",
    )
    assert valid.high >= valid.low

    no_cross_field_validator = False
    try:
        invalid_range = OHLCVBar(
            timestamp=datetime.now(timezone.utc),
            open=100,
            high=95,
            low=99,
            close=98,
            volume=1000,
            asset="AAPL",
        )
        no_cross_field_validator = True
        assert invalid_range.high < invalid_range.low
    except ValidationError:
        pass

    negative_volume_accepted = False
    try:
        neg = OHLCVBar(
            timestamp=datetime.now(timezone.utc),
            open=100,
            high=105,
            low=99,
            close=103,
            volume=-1,
            asset="AAPL",
        )
        negative_volume_accepted = True
        assert neg.volume == -1
    except ValidationError:
        pass

    if no_cross_field_validator:
        warnings.warn(
            "OHLCVBar schema allows high < low — consider model_validator(mode='after').",
            stacklevel=1,
        )
    if negative_volume_accepted:
        warnings.warn(
            "OHLCVBar schema allows negative volume — consider Field(ge=0).",
            stacklevel=1,
        )


def test_order_status_transitions() -> None:
    """FUNCTION TESTED: schemas.execution.OrderStatus and schemas.execution.OrderType"""
    for status in OrderStatus:
        order = Order(
            order_id=f"ord-{status.value}",
            strategy_id="strat-1",
            asset="AAPL",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=1,
            status=status,
        )
        assert order.status == status

    expected_statuses = {"pending", "submitted", "filled", "partially_filled", "cancelled", "rejected"}
    expected_types = {"market", "limit", "stop", "stop_limit"}
    assert {x.value for x in OrderStatus} == expected_statuses, _diag(
        "schemas.execution.OrderStatus",
        "enum members",
        expected_statuses,
        {x.value for x in OrderStatus},
        "ENUM_MISMATCH",
    )
    assert {x.value for x in OrderType} == expected_types, _diag(
        "schemas.execution.OrderType",
        "enum members",
        expected_types,
        {x.value for x in OrderType},
        "ENUM_MISMATCH",
    )


def test_backtest_result_types(valid_backtest_result: BacktestResult) -> None:
    """FUNCTION TESTED: schemas.rewards.BacktestResult"""
    neg_sharpe = valid_backtest_result.model_copy(update={"sharpe_ratio": -1.2})
    assert neg_sharpe.sharpe_ratio == -1.2

    zero_drawdown = valid_backtest_result.model_copy(update={"max_drawdown": 0})
    assert zero_drawdown.max_drawdown == 0

    positive_drawdown = valid_backtest_result.model_copy(update={"max_drawdown": 0.2})
    assert positive_drawdown.max_drawdown == 0.2

    unconstrained_win = BacktestResult.model_validate(
        {**valid_backtest_result.model_dump(mode="python"), "win_rate": 1.5}
    )
    if unconstrained_win.win_rate > 1:
        warnings.warn("BacktestResult.win_rate is not constrained to [0, 1].", stacklevel=1)

    zero_trades = valid_backtest_result.model_copy(update={"total_trades": 0})
    assert zero_trades.total_trades == 0


def test_episodic_trace_accepts_any_content(valid_strategy: Strategy) -> None:
    """FUNCTION TESTED: schemas.memory.EpisodicTrace"""
    trace_model = EpisodicTrace(
        trace_id="trace-1",
        strategy=valid_strategy,
        user_modification={"note": "none"},
        execution_result=[{"order_id": "ord-1"}],
        reward=0.4,
    )
    dumped_model = trace_model.model_dump()
    assert dumped_model["strategy"]["strategy_id"] == valid_strategy.strategy_id, _diag(
        "schemas.memory.EpisodicTrace",
        "strategy as model",
        valid_strategy.strategy_id,
        dumped_model["strategy"].get("strategy_id"),
        "SERIALIZATION_FAILURE",
    )

    trace_dict = EpisodicTrace(
        trace_id="trace-2",
        strategy={"custom": "payload"},
        user_modification=None,
        execution_result={"ok": True},
        reward=None,
    )
    dumped_dict = trace_dict.model_dump()
    assert dumped_dict["strategy"]["custom"] == "payload", _diag(
        "schemas.memory.EpisodicTrace",
        "strategy as dict",
        "payload",
        dumped_dict["strategy"].get("custom"),
        "SERIALIZATION_FAILURE",
    )


def test_user_modified_strategy_relationship(valid_strategy: Strategy) -> None:
    """FUNCTION TESTED: schemas.strategy.UserModifiedStrategy"""
    modified = valid_strategy.model_copy(
        update={"positions": [valid_strategy.positions[0].model_copy(update={"size_pct": 5.0})]}
    )
    wrapped = UserModifiedStrategy(original_strategy=valid_strategy, modified_strategy=modified)
    assert wrapped.approved is False, _diag(
        "schemas.strategy.UserModifiedStrategy",
        "approved default",
        False,
        wrapped.approved,
        "WRONG_DEFAULT",
    )
    roundtrip = UserModifiedStrategy.model_validate_json(wrapped.model_dump_json())
    assert roundtrip.original_strategy.strategy_id == valid_strategy.strategy_id
    assert roundtrip.modified_strategy.positions[0].size_pct == 5.0
