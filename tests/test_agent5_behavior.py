from __future__ import annotations

from copy import deepcopy

import pytest

from agents.agent5_behavior import Agent5BehaviorStub, BehaviorDataCollector
from memory.working import WorkingMemory
from schemas.strategy import Position, PositionAction, RiskMetrics, Strategy
from schemas.analysis_feedback import AnalysisFeedback


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


class _MemoryStub:
    def __init__(self):
        self.working = WorkingMemory(max_items=500)


def _make_strategy(
    strategy_id: str | None = None,
    aapl_size: float = 10.0,
    include_tsla: bool = True,
    include_googl: bool = False,
) -> Strategy:
    positions = [
        Position(
            asset="AAPL",
            action=PositionAction.LONG,
            size_pct=aapl_size,
            entry_price_target=None,
            stop_loss_pct=3.0,
            take_profit_pct=8.0,
            time_horizon_days=10,
            confidence=0.7,
        ),
        Position(
            asset="MSFT",
            action=PositionAction.LONG,
            size_pct=15.0,
            entry_price_target=None,
            stop_loss_pct=3.0,
            take_profit_pct=8.0,
            time_horizon_days=10,
            confidence=0.7,
        ),
    ]
    if include_tsla:
        positions.append(
            Position(
                asset="TSLA",
                action=PositionAction.LONG,
                size_pct=20.0,
                entry_price_target=None,
                stop_loss_pct=4.0,
                take_profit_pct=10.0,
                time_horizon_days=12,
                confidence=0.65,
            )
        )
    if include_googl:
        positions.append(
            Position(
                asset="GOOGL",
                action=PositionAction.LONG,
                size_pct=8.0,
                entry_price_target=None,
                stop_loss_pct=3.0,
                take_profit_pct=7.0,
                time_horizon_days=10,
                confidence=0.66,
            )
        )
    kwargs = {}
    if strategy_id is not None:
        kwargs["strategy_id"] = strategy_id
    return Strategy(
        market_regime="bull",
        positions=positions,
        rationale="behavior test",
        risk_metrics=RiskMetrics(total_exposure_pct=sum(p.size_pct for p in positions)),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_record_modification():
    """FUNCTION TESTED: agents.agent5_behavior.BehaviorDataCollector.record_modification"""
    memory = _MemoryStub()
    collector = BehaviorDataCollector(memory)
    original = _make_strategy(strategy_id="orig-1")
    modified = _make_strategy(strategy_id="mod-1", aapl_size=5.0, include_tsla=False, include_googl=True)

    record_id = await collector.record_modification(original, modified, user_notes="I prefer less volatile positions")
    rec = await memory.working.get(f"behavior:{record_id}")
    assert rec is not None
    assert rec["modification_type"] != "none", _diag(
        "agents.agent5_behavior.BehaviorDataCollector.record_modification",
        "modification record",
        "modification_type != none",
        rec["modification_type"],
        "DATA_INTEGRITY",
    )
    assert rec["original"]["strategy_id"] == "orig-1"
    assert rec["modified"]["strategy_id"] == "mod-1"


@pytest.mark.asyncio
async def test_record_approval_without_modification():
    """FUNCTION TESTED: agents.agent5_behavior.BehaviorDataCollector.record_approval_without_modification"""
    memory = _MemoryStub()
    collector = BehaviorDataCollector(memory)
    strategy = _make_strategy(strategy_id="same-1")
    record_id = await collector.record_approval_without_modification(strategy)
    assert record_id
    stats = await collector.get_statistics()
    assert stats["total_records"] == 1
    data = await collector.get_training_data(min_records=1)
    assert data[0]["label"] == "none"
    assert data[0]["original_output"] == data[0]["modified_output"]


@pytest.mark.asyncio
async def test_record_rejection():
    """FUNCTION TESTED: agents.agent5_behavior.BehaviorDataCollector.record_rejection"""
    memory = _MemoryStub()
    collector = BehaviorDataCollector(memory)
    strategy = _make_strategy()
    await collector.record_rejection(strategy, reason="Too risky")
    data = await collector.get_training_data(min_records=1)
    assert data[0]["label"] == "rejected"
    assert data[0]["modified_output"] is None


@pytest.mark.asyncio
async def test_get_training_data_format():
    """FUNCTION TESTED: agents.agent5_behavior.BehaviorDataCollector.get_training_data"""
    memory = _MemoryStub()
    collector = BehaviorDataCollector(memory)
    s = _make_strategy()
    for i in range(5):
        await collector.record_modification(s, _make_strategy(aapl_size=5.0 + i), user_notes=f"mod-{i}")
    for _ in range(3):
        await collector.record_approval_without_modification(s)
    for i in range(2):
        await collector.record_rejection(s, reason=f"reject-{i}")
    rows = await collector.get_training_data(min_records=5)
    assert len(rows) == 10
    required = {"original_prompt", "original_output", "modified_output", "label"}
    assert required <= set(rows[0].keys()), _diag(
        "agents.agent5_behavior.BehaviorDataCollector.get_training_data",
        "10 records",
        required,
        set(rows[0].keys()),
        "API_CONTRACT",
    )


@pytest.mark.asyncio
async def test_get_training_data_below_minimum():
    """FUNCTION TESTED: agents.agent5_behavior.BehaviorDataCollector.get_training_data"""
    memory = _MemoryStub()
    collector = BehaviorDataCollector(memory)
    s = _make_strategy()
    for _ in range(3):
        await collector.record_approval_without_modification(s)
    rows = await collector.get_training_data(min_records=100)
    assert rows == []


@pytest.mark.asyncio
async def test_get_statistics():
    """FUNCTION TESTED: agents.agent5_behavior.BehaviorDataCollector.get_statistics"""
    memory = _MemoryStub()
    collector = BehaviorDataCollector(memory)
    s = _make_strategy()
    for i in range(5):
        await collector.record_modification(s, _make_strategy(aapl_size=6.0 + i))
    for _ in range(3):
        await collector.record_approval_without_modification(s)
    for _ in range(2):
        await collector.record_rejection(s)
    stats = await collector.get_statistics()
    assert stats["total_records"] == 10
    assert stats["modification_rate"] == 0.5
    assert stats["rejection_rate"] == 0.2


@pytest.mark.asyncio
async def test_on_strategy_approved_routes_correctly(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent5_behavior.Agent5BehaviorStub.on_strategy_approved"""
    agent = Agent5BehaviorStub(_MemoryStub())

    called = {"approval": 0, "modification": 0}

    async def fake_approval(strategy):
        called["approval"] += 1
        return "a"

    async def fake_modification(original, modified, notes=None):
        called["modification"] += 1
        return "m"

    monkeypatch.setattr(agent.collector, "record_approval_without_modification", fake_approval)
    monkeypatch.setattr(agent.collector, "record_modification", fake_modification)

    s1 = _make_strategy(strategy_id="id-1")
    await agent.on_strategy_approved(s1, s1)
    await agent.on_strategy_approved(s1, _make_strategy(strategy_id="id-2"))
    assert called["approval"] == 1
    assert called["modification"] == 1


@pytest.mark.asyncio
async def test_on_analysis_feedback_records_preference():
    """FUNCTION TESTED: agents.agent5_behavior.Agent5BehaviorStub.on_analysis_feedback"""
    from schemas.analysis import MarketAnalysis, MarketOutlook, OutlookDirection, SentimentView

    agent = Agent5BehaviorStub(_MemoryStub())
    analysis = MarketAnalysis(
        query_context="AAPL",
        sentiment=SentimentView(overall_score=0.3),
        outlook=MarketOutlook(direction=OutlookDirection.BULLISH, horizon_days=10, confidence=0.6),
    )
    revised = analysis.model_copy(deep=True)
    revised.outlook.direction = OutlookDirection.BEARISH
    feedback = AnalysisFeedback(analysis_id=analysis.analysis_id, overall_verdict="disagree", free_text="Too bullish")
    record_id = await agent.on_analysis_feedback(analysis, revised, feedback)
    assert record_id
    report = await agent.report()
    assert report["analysis_preference_records"] == 1


@pytest.mark.asyncio
async def test_strategy_equality_detection():
    """FUNCTION TESTED: agents.agent5_behavior.Agent5BehaviorStub.on_strategy_approved equality behavior"""
    memory = _MemoryStub()
    agent = Agent5BehaviorStub(memory)

    same_content_same_id = _make_strategy(strategy_id="eq-1")
    await agent.on_strategy_approved(same_content_same_id, same_content_same_id)

    same_content_diff_id_a = _make_strategy(strategy_id="eq-a")
    same_content_diff_id_b = _make_strategy(strategy_id="eq-b")
    await agent.on_strategy_approved(same_content_diff_id_a, same_content_diff_id_b)

    same_id_diff_content_orig = _make_strategy(strategy_id="eq-c")
    same_id_diff_content_mod = _make_strategy(strategy_id="eq-c", aapl_size=5.0)
    await agent.on_strategy_approved(same_id_diff_content_orig, same_id_diff_content_mod)

    data = await agent.collector.get_training_data(min_records=1)
    labels = [d["label"] for d in data]
    assert labels.count("none") == 1
    assert labels.count("edited") == 2, _diag(
        "agents.agent5_behavior.Agent5BehaviorStub.on_strategy_approved",
        "equality triad",
        "1 none + 2 edited",
        labels,
        "DATA_INTEGRITY",
    )
