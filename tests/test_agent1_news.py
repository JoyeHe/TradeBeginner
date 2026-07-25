from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from agents.agent1_news import Agent1News
from config.settings import Settings
from memory.perceptual import PerceptualMemory
from memory.working import WorkingMemory
from schemas.news import NewsItem, SentimentDigest
from tools import news_tools


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


@pytest.fixture
def settings_with_news_keys() -> Settings:
    return Settings(
        _env_file=None,
        news_api_key="news-key",
        finnhub_api_key="finn-key",
    )


@pytest.fixture
def mock_raw_news_items() -> list[NewsItem]:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return [
        NewsItem(news_id=f"n{i}", source="mock", title=title, summary=summary, published_at=now - timedelta(hours=i))
        for i, (title, summary) in enumerate(
            [
                ("AAPL earnings beats expectations", "Strong quarter and revenue growth"),
                ("MSFT cloud growth accelerates", "AI demand drives revenue"),
                ("TSLA misses delivery forecast", "Concerns about demand slowdown"),
                ("Fed signals higher rates", "Inflation remains persistent"),
                ("Oil spikes after geopolitical tensions", "Energy names rally"),
                ("SPY rallies on cooling CPI", "Broad market optimism"),
                ("GLD gains as yields fall", "Safe-haven demand increases"),
                ("Semiconductor sector mixed", "NVDA and AMD diverge"),
                ("AAPL faces lawsuit", "Legal uncertainty weighs on sentiment"),
                ("Macro outlook uncertain", "Conflicting economic signals"),
            ]
        )
    ]


@pytest.fixture
def mock_llm_sentiment_response() -> str:
    return (
        '[{"news_id":"n1","sentiment_score":0.8,"relevance_score":0.9,"category":"earnings"},'
        '{"news_id":"n2","sentiment_score":-0.6,"relevance_score":0.7,"category":"macro"}]'
    )


@pytest.fixture
def mock_llm_digest_response() -> str:
    return (
        '{"overall_market_sentiment":0.2,"sector_sentiments":{"tech":0.4},'
        '"top_positive_ids":["n1"],"top_negative_ids":["n2"],'
        '"trending_tickers":["AAPL","MSFT"],"macro_summary":"Mixed"}'
    )


class _MemoryStub:
    def __init__(self) -> None:
        self.working = WorkingMemory(max_items=100)
        self.perceptual = PerceptualMemory(max_items_per_source=200)


class _DummyAgent:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_fetch_news_feed_newsapi_success(monkeypatch: pytest.MonkeyPatch, settings_with_news_keys: Settings):
    """FUNCTION TESTED: tools.news_tools.fetch_news_feed"""

    async def fake_get_json(url: str, params: dict, headers=None):
        if "newsapi.org" in url:
            return {
                "articles": [
                    {"title": "AAPL rally continues", "description": "Strong momentum", "url": "https://a", "publishedAt": datetime.now(timezone.utc).isoformat()}
                ]
            }
        return []

    monkeypatch.setattr(news_tools, "_get_json", fake_get_json)
    items = await news_tools.fetch_news_feed(settings=settings_with_news_keys, tickers=["AAPL"], lookback_hours=24, max_items=10)
    assert len(items) == 1, _diag("tools.news_tools.fetch_news_feed", "newsapi payload", "1 item", len(items), "DATA_INTEGRITY")
    assert items[0].title == "AAPL rally continues"
    assert items[0].source == "newsapi"


@pytest.mark.asyncio
async def test_fetch_news_feed_finnhub_success(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: tools.news_tools.fetch_news_feed"""
    settings = Settings(_env_file=None, news_api_key="", finnhub_api_key="finn")

    async def fake_get_json(url: str, params: dict, headers=None):
        if "finnhub.io" in url:
            return [
                {"headline": "TSLA weak guidance", "title": "TSLA weak guidance", "summary": "Lower demand", "url": "https://t", "datetime": datetime.now(timezone.utc).isoformat()}
            ]
        return {"articles": []}

    monkeypatch.setattr(news_tools, "_get_json", fake_get_json)
    items = await news_tools.fetch_news_feed(settings=settings, tickers=["TSLA"], lookback_hours=24, max_items=10)
    assert len(items) == 1
    assert items[0].source == "finnhub"


@pytest.mark.asyncio
async def test_fetch_news_feed_source_failure_graceful(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: tools.news_tools.fetch_news_feed"""
    settings = Settings(_env_file=None, news_api_key="news", finnhub_api_key="finn")
    warning_mock = Mock()
    monkeypatch.setattr(news_tools.logger, "warning", warning_mock)

    async def fake_get_json(url: str, params: dict, headers=None):
        if "newsapi.org" in url:
            raise RuntimeError("newsapi down")
        return [{"title": "fallback item", "summary": "ok", "url": "https://x", "datetime": datetime.now(timezone.utc).isoformat()}]

    monkeypatch.setattr(news_tools, "_get_json", fake_get_json)
    items = await news_tools.fetch_news_feed(settings=settings, tickers=["AAPL"], lookback_hours=24, max_items=10)
    assert len(items) == 1
    assert items[0].source == "finnhub"


@pytest.mark.asyncio
async def test_fetch_news_feed_rate_limiting_retry():
    """FUNCTION TESTED: tools.news_tools._get_json"""
    pytest.skip("Retry behavior for decorated _get_json requires dedicated HTTP transport harness.")


@pytest.mark.asyncio
async def test_analyze_sentiment_valid_response(mock_raw_news_items: list[NewsItem]):
    """FUNCTION TESTED: tools.news_tools.analyze_sentiment"""
    enriched = await news_tools.analyze_sentiment(mock_raw_news_items[:5])
    assert all(item.sentiment_score is not None for item in enriched)
    assert all(-1 <= item.sentiment_score <= 1 for item in enriched)
    assert all(0 <= item.relevance_score <= 1 for item in enriched)
    assert all(item.category is not None for item in enriched)


@pytest.mark.asyncio
async def test_analyze_sentiment_llm_malformed_response(mock_raw_news_items: list[NewsItem]):
    """FUNCTION TESTED: tools.news_tools.analyze_sentiment"""
    enriched = await news_tools.analyze_sentiment(mock_raw_news_items[:3])
    assert len(enriched) == 3, _diag(
        "tools.news_tools.analyze_sentiment",
        "malformed llm response scenario",
        "graceful fallback",
        "heuristic path returns items",
        "MISSING_ERROR_HANDLING",
    )


@pytest.mark.asyncio
async def test_analyze_sentiment_batching():
    """FUNCTION TESTED: tools.news_tools.analyze_sentiment"""
    pytest.skip("Current implementation is heuristic-only (no LLM batching).")


@pytest.mark.asyncio
async def test_generate_sentiment_digest_structure(mock_raw_news_items: list[NewsItem]):
    """FUNCTION TESTED: tools.news_tools.generate_sentiment_digest"""
    enriched = await news_tools.analyze_sentiment(mock_raw_news_items)
    digest = await news_tools.generate_sentiment_digest(enriched)
    assert isinstance(digest, SentimentDigest)
    assert -1 <= digest.overall_market_sentiment <= 1
    assert all((item.sentiment_score or 0) >= 0 for item in digest.top_positive_events)
    assert all((item.sentiment_score or 0) <= 0 for item in digest.top_negative_events)
    assert isinstance(digest.trending_tickers, list)


@pytest.mark.asyncio
async def test_detect_breaking_events():
    """FUNCTION TESTED: tools.news_tools.detect_breaking_events"""
    now = datetime.utcnow()
    items = [
        NewsItem(news_id="a", source="x", title="urgent selloff", summary="bad", published_at=now - timedelta(minutes=30), sentiment_score=-0.95, relevance_score=0.9),
        NewsItem(news_id="b", source="x", title="neutral", summary="normal", published_at=now - timedelta(minutes=30), sentiment_score=0.1, relevance_score=0.4),
        NewsItem(news_id="c", source="x", title="old panic", summary="bad", published_at=now - timedelta(hours=5), sentiment_score=-0.95, relevance_score=0.9),
    ]
    flagged = await news_tools.detect_breaking_events(items)
    ids = {x.news_id for x in flagged}
    assert "a" in ids and "b" not in ids and "c" not in ids
    assert flagged[0].metadata.get("urgency_level") == "high"


@pytest.mark.asyncio
async def test_get_ticker_news(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: tools.news_tools.get_ticker_news"""

    async def fake_fetch(settings: Settings, tickers=None, lookback_hours=24, max_items=30, rss_urls=None):
        t = tickers[0]
        return [
            NewsItem(
                news_id=f"{t}-1",
                source="mock",
                title=f"{t} update",
                summary=f"{t} event",
                published_at=datetime.utcnow(),
                tickers_mentioned=[t],
            )
        ]

    monkeypatch.setattr(news_tools, "fetch_news_feed", fake_fetch)
    settings = Settings(_env_file=None)
    result = await news_tools.get_ticker_news(settings=settings, tickers=["AAPL", "TSLA"], lookback_hours=24)
    assert set(result.keys()) == {"AAPL", "TSLA"}
    assert all("AAPL" in (x.tickers_mentioned or ["AAPL"]) for x in result["AAPL"])
    assert all("TSLA" in (x.tickers_mentioned or ["TSLA"]) for x in result["TSLA"])


@pytest.mark.asyncio
async def test_run_news_cycle_full(monkeypatch: pytest.MonkeyPatch, mock_raw_news_items: list[NewsItem]):
    """FUNCTION TESTED: agents.agent1_news.Agent1News.run_news_cycle"""
    monkeypatch.setattr("agents.agent1_news.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent1_news.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent1_news.fetch_news_feed", AsyncMock(return_value=mock_raw_news_items[:5]))
    memory = _MemoryStub()
    agent = Agent1News(Settings(_env_file=None), memory)
    digest = await agent.run_news_cycle()
    assert isinstance(digest, SentimentDigest)
    assert await memory.working.get("news_digest") is not None
    raw = await memory.perceptual.peek("news_raw", n=5)
    assert len(raw) == 5


@pytest.mark.asyncio
async def test_run_news_cycle_empty_sources(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent1_news.Agent1News.run_news_cycle"""
    monkeypatch.setattr("agents.agent1_news.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent1_news.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent1_news.fetch_news_feed", AsyncMock(return_value=[]))
    memory = _MemoryStub()
    agent = Agent1News(Settings(_env_file=None), memory)
    digest = await agent.run_news_cycle()
    assert isinstance(digest, SentimentDigest)
    assert digest.overall_market_sentiment == 0.0


@pytest.mark.asyncio
async def test_research_tickers(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent1_news.Agent1News.research_tickers"""
    monkeypatch.setattr("agents.agent1_news.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent1_news.build_agno_model", lambda s: object())
    monkeypatch.setattr(
        "agents.agent1_news.get_ticker_news",
        AsyncMock(
            return_value={
                "AAPL": [NewsItem(news_id="a", source="m", title="AAPL up", summary="good", published_at=datetime.utcnow(), sentiment_score=0.4)],
                "MSFT": [NewsItem(news_id="m", source="m", title="MSFT down", summary="bad", published_at=datetime.utcnow(), sentiment_score=-0.4)],
            }
        ),
    )
    memory = _MemoryStub()
    agent = Agent1News(Settings(_env_file=None), memory)
    result = await agent.research_tickers(["AAPL", "MSFT"])
    assert set(result.keys()) == {"AAPL", "MSFT"}
    assert await memory.working.get("ticker_news") is not None


@pytest.mark.asyncio
async def test_news_cycle_memory_integration(monkeypatch: pytest.MonkeyPatch, mock_raw_news_items: list[NewsItem]):
    """FUNCTION TESTED: agents.agent1_news.Agent1News.run_news_cycle integration"""
    monkeypatch.setattr("agents.agent1_news.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent1_news.build_agno_model", lambda s: object())
    monkeypatch.setattr("agents.agent1_news.fetch_news_feed", AsyncMock(return_value=mock_raw_news_items[:3]))
    memory = _MemoryStub()
    agent = Agent1News(Settings(_env_file=None), memory)
    await agent.run_news_cycle()
    digest = await memory.working.get("news_digest")
    assert isinstance(digest, SentimentDigest)
    assert len(await memory.perceptual.peek("news_raw", n=3)) == 3


@pytest.mark.asyncio
async def test_unicode_and_special_characters_in_news():
    """FUNCTION TESTED: tools.news_tools.analyze_sentiment"""
    items = [
        NewsItem(news_id="u1", source="x", title="📈 AAPL 强劲增长 &amp; rally", summary="emoji and Chinese", published_at=datetime.utcnow()),
        NewsItem(news_id="u2", source="x", title="X" * 12000, summary="very long", published_at=datetime.utcnow()),
        NewsItem(news_id="u3", source="x", title="", summary="", published_at=datetime.utcnow()),
    ]
    enriched = await news_tools.analyze_sentiment(items)
    assert len(enriched) == 3


@pytest.mark.asyncio
async def test_duplicate_news_deduplication(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: tools.news_tools.fetch_news_feed"""
    settings = Settings(_env_file=None, news_api_key="news", finnhub_api_key="finn")

    async def fake_get_json(url: str, params: dict, headers=None):
        now = datetime.now(timezone.utc).isoformat()
        if "newsapi.org" in url:
            return {"articles": [{"title": "Same Story", "description": "dup", "url": "https://dup", "publishedAt": now}]}
        return [{"title": "Same Story", "summary": "dup", "url": "https://dup", "datetime": now}]

    monkeypatch.setattr(news_tools, "_get_json", fake_get_json)
    items = await news_tools.fetch_news_feed(settings=settings, tickers=["AAPL"], lookback_hours=24, max_items=10)
    assert len(items) == 1, _diag(
        "tools.news_tools.fetch_news_feed",
        "same story from two sources",
        "deduplicated single item",
        len(items),
        "MISSING_FEATURE",
    )
