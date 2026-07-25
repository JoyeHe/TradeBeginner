from __future__ import annotations

from unittest.mock import AsyncMock
from datetime import datetime, timedelta, timezone
import json
import math
from types import SimpleNamespace

import pytest
import pytest_asyncio

from config.settings import Settings
from memory import MemoryManager
from risk.controller import RiskController
from tools.execution_tools import PaperTradingEngine
from schemas.execution import PortfolioState
from schemas.market_data import MarketSnapshot, OHLCVBar, TechnicalIndicators
from schemas.news import NewsItem
from schemas.rewards import BacktestResult
from schemas.strategy import MarketRegime, Position, PositionAction, RiskMetrics, Strategy

pytest_plugins = ["tests.report_plugin"]


@pytest.fixture
def valid_position() -> Position:
    return Position(
        asset="AAPL",
        action=PositionAction.LONG,
        size_pct=10.0,
        entry_price_target=180.0,
        stop_loss_pct=4.0,
        take_profit_pct=8.0,
        time_horizon_days=10,
        confidence=0.7,
    )


@pytest.fixture
def valid_strategy() -> Strategy:
    return Strategy(
        market_regime=MarketRegime.BULL,
        positions=[
            Position(
                asset="AAPL",
                action=PositionAction.LONG,
                size_pct=10.0,
                entry_price_target=180.0,
                stop_loss_pct=4.0,
                take_profit_pct=8.0,
                time_horizon_days=10,
                confidence=0.7,
            ),
            Position(
                asset="MSFT",
                action=PositionAction.LONG,
                size_pct=8.0,
                entry_price_target=420.0,
                stop_loss_pct=3.0,
                take_profit_pct=7.0,
                time_horizon_days=15,
                confidence=0.65,
            ),
        ],
        rationale="Trend and sentiment aligned for mega-cap tech.",
        data_sources_used=["agent1_news", "agent2_market"],
        risk_metrics=RiskMetrics(
            portfolio_var_95=0.02,
            max_drawdown_estimate=0.08,
            correlation_to_existing=0.45,
            total_exposure_pct=18.0,
            sector_concentrations={"Technology": 18.0},
        ),
        metadata={"source": "test_fixture"},
    )


@pytest.fixture
def valid_news_item() -> NewsItem:
    return NewsItem(
        news_id="news-001",
        source="newsapi",
        title="Apple reports stronger than expected guidance",
        summary="Apple raised guidance and expects robust services growth.",
        full_text="Detailed article body.",
        url="https://example.com/apple-guidance",
        published_at=datetime.now(timezone.utc) - timedelta(minutes=10),
        tickers_mentioned=["AAPL"],
        sentiment_score=0.6,
        relevance_score=0.9,
        category="earnings",
        metadata={"provider": "fixture"},
    )


@pytest.fixture
def valid_market_snapshot() -> MarketSnapshot:
    ts = datetime.now(timezone.utc)
    bar = OHLCVBar(
        timestamp=ts,
        open=100.0,
        high=102.0,
        low=99.5,
        close=101.0,
        volume=1_200_000,
        asset="SPY",
    )
    indicators = TechnicalIndicators(
        asset="SPY",
        timestamp=ts,
        rsi_14=55.0,
        sma_20=100.5,
        sma_50=98.8,
        sma_200=94.2,
        macd={"line": 1.2, "signal": 0.8, "histogram": 0.4},
        bollinger_bands={"upper": 103.0, "middle": 100.5, "lower": 98.0},
    )
    return MarketSnapshot(
        timestamp=ts,
        assets={"SPY": bar},
        indicators={"SPY": indicators},
        market_breadth={"regime": "bull", "advance_decline": 1.6},
        vix=16.4,
        sector_performance={"Technology": 1.2, "Healthcare": 0.4},
    )


@pytest.fixture
def valid_portfolio_state() -> PortfolioState:
    return PortfolioState(
        timestamp=datetime.now(timezone.utc),
        cash=75_000.0,
        total_value=100_000.0,
        positions={
            "AAPL": {"quantity": 50, "avg_cost": 180.0, "market_value": 9_250.0},
            "MSFT": {"quantity": 20, "avg_cost": 410.0, "market_value": 8_400.0},
        },
        daily_pnl=250.0,
        total_pnl=1_200.0,
    )


@pytest.fixture
def valid_backtest_result() -> BacktestResult:
    return BacktestResult(
        strategy_id="strat-001",
        backtest_start=datetime.now(timezone.utc) - timedelta(days=30),
        backtest_end=datetime.now(timezone.utc),
        total_return=0.12,
        sharpe_ratio=1.4,
        max_drawdown=0.08,
        win_rate=0.62,
        profit_factor=1.6,
        total_trades=12,
        avg_trade_return=0.01,
        volatility=0.15,
        partial_data=False,
    )


@pytest.fixture
def mock_llm_client(valid_strategy: Strategy):
    """Configurable AsyncMock for LLM client responses."""

    class MockLLMClient:
        def __init__(self, response: str):
            self._response = response
            self.arun = AsyncMock(side_effect=self._arun)

        async def _arun(self, prompt: str):
            return SimpleNamespace(content=self._response)

        def set_response(self, response: str):
            self._response = response

    return MockLLMClient(valid_strategy.model_dump_json())


@pytest.fixture
def mock_news_api():
    """Realistic canned NewsAPI + Finnhub response payloads."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "newsapi": {
            "articles": [
                {"title": "AAPL beats earnings", "description": "Strong growth", "url": "https://n1", "publishedAt": now},
                {"title": "MSFT cloud accelerates", "description": "AI tailwind", "url": "https://n2", "publishedAt": now},
            ]
        },
        "finnhub": [
            {"title": "TSLA delivery miss", "summary": "Demand concerns", "url": "https://f1", "datetime": now},
            {"title": "Fed minutes hawkish", "summary": "Rates higher", "url": "https://f2", "datetime": now},
        ],
    }


@pytest.fixture
def mock_market_data():
    """Realistic OHLCV bars and indicator references."""
    idx = [datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=i) for i in range(60)]
    prices = [100 + i * 0.4 + 2 * math.sin(i / 5) for i in range(60)]
    bars = [
        {
            "timestamp": t.replace(tzinfo=None),
            "open": p - 0.2,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p,
            "volume": 1_000_000 + i * 1_000,
            "asset": "AAPL",
        }
        for i, (t, p) in enumerate(zip(idx, prices))
    ]
    return {
        "bars": bars,
        "indicators": {"rsi_14": 55.2, "sma_20": 118.3, "sma_50": 112.1},
    }


@pytest_asyncio.fixture
async def memory_manager(tmp_path):
    """MemoryManager with real working/perceptual and fallback episodic/semantic."""
    settings = Settings(
        _env_file=None,
        postgres_url="postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid",
        chroma_persist_dir=str(tmp_path / "chroma"),
    )
    mm = MemoryManager(settings)
    await mm.initialize()
    try:
        yield mm
    finally:
        await mm.shutdown()


@pytest.fixture
def paper_engine() -> PaperTradingEngine:
    return PaperTradingEngine(initial_capital=100_000.0)


@pytest.fixture
def risk_controller() -> RiskController:
    return RiskController(Settings(_env_file=None))


@pytest_asyncio.fixture
async def integrated_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Full TradingPipeline with external APIs/LLM mocked."""
    from orchestrator.pipeline import TradingPipeline
    from schemas.news import SentimentDigest

    class _DummyAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def arun(self, prompt: str):
            payload = {
                "strategy_id": "int-pipeline-strategy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "market_regime": "bull",
                "positions": [
                    {
                        "asset": "AAPL",
                        "action": "long",
                        "size_pct": 8.0,
                        "entry_price_target": None,
                        "stop_loss_pct": 3.0,
                        "take_profit_pct": 8.0,
                        "time_horizon_days": 10,
                        "confidence": 0.7,
                    }
                ],
                "rationale": "Integrated fixture response",
                "data_sources_used": ["market_snapshot", "news_digest"],
                "risk_metrics": {"total_exposure_pct": 8.0},
                "metadata": {"fixture": True},
            }
            return SimpleNamespace(content=json.dumps(payload))

    monkeypatch.setattr("agents.agent1_news.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent2_market.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent3_strategy.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent4_executor.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent6_backtest.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent1_news.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent2_market.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent3_strategy.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent4_executor.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent6_backtest.build_agno_model", lambda s: object())

    async def fake_fetch_news_feed(settings, tickers=None, lookback_hours=24, max_items=50, rss_urls=None):
        return [
            NewsItem(
                news_id="fixture-news",
                source="mock",
                title="AAPL strength continues",
                summary="positive trend",
                published_at=datetime.now(timezone.utc).replace(tzinfo=None),
                tickers_mentioned=["AAPL"],
            )
        ]

    async def fake_generate_digest(items):
        return SentimentDigest(
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
            overall_market_sentiment=0.5,
            sector_sentiments={"Technology": 0.5},
            top_positive_events=items,
            top_negative_events=[],
            trending_tickers=["AAPL"],
            macro_summary="fixture digest",
        )

    monkeypatch.setattr("agents.agent1_news.fetch_news_feed", fake_fetch_news_feed)
    monkeypatch.setattr("agents.agent1_news.generate_sentiment_digest", fake_generate_digest)

    from schemas.market_data import OHLCVBar

    async def fake_fetch_price_data(tickers, start_date, end_date, interval="1d"):
        out = {}
        for t in tickers:
            bars = []
            for i in range(260):
                p = 100 + i * 0.3 + math.sin(i / 5)
                if t == "^VIX":
                    p = 16.0
                bars.append(
                    OHLCVBar(
                        timestamp=(datetime(2025, 1, 1) + timedelta(days=i)),
                        open=p - 0.1,
                        high=p + 1.0,
                        low=p - 1.0,
                        close=p,
                        volume=1_000_000,
                        asset=t,
                    )
                )
            out[t] = bars
        return out

    monkeypatch.setattr("tools.market_tools.fetch_price_data", fake_fetch_price_data)
    monkeypatch.setattr("agents.agent4_executor.fetch_price_data", fake_fetch_price_data)

    settings = Settings(
        _env_file=None,
        postgres_url="postgresql+asyncpg://invalid:invalid@127.0.0.1:1/invalid",
        chroma_persist_dir=str(tmp_path / "chroma"),
        paper_trading=True,
        llm_api_key="test-key",
    )
    pipeline = TradingPipeline(settings)
    await pipeline.initialize()
    try:
        yield pipeline
    finally:
        await pipeline.shutdown()
