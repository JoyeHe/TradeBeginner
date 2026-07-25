"""Agno @tool wrappers for TradeBeginner data tools."""

from __future__ import annotations

import json

from agno.tools import Toolkit, tool

from config.settings import Settings
from tools.market_tools import build_market_snapshot, user_market_lookup
from tools.news_tools import fetch_news_feed, get_ticker_news, search_news_with_llm


class NewsAgentTools(Toolkit):
    """News and sentiment tools exposed to Agno Agent 1."""

    def __init__(self, settings: Settings, **kwargs):
        self._settings = settings
        super().__init__(name="news_agent_tools", tools=self._build_tools(), **kwargs)

    def _build_tools(self) -> list:
        settings = self._settings

        @tool(
            name="fetch_news_feed",
            description="Fetch market news from NewsAPI, Finnhub, TrendRadar, and RSS feeds.",
        )
        async def fetch_news(lookback_hours: int = 24, max_items: int = 50) -> str:
            items = await fetch_news_feed(settings, lookback_hours=lookback_hours, max_items=max_items)
            return json.dumps([i.model_dump(mode="json") for i in items], default=str)

        @tool(
            name="search_news_with_llm",
            description="Natural-language news search with TrendRadar, APIs, and LLM relevance ranking.",
        )
        async def search_news(query: str) -> str:
            result = await search_news_with_llm(settings, user_query=query)
            return json.dumps(result, default=str)

        @tool(
            name="get_ticker_news",
            description="Fetch recent news for specific ticker symbols.",
        )
        async def ticker_news(tickers: str, lookback_hours: int = 72) -> str:
            symbols = [t.strip().upper() for t in tickers.split(",") if t.strip()]
            result = await get_ticker_news(settings, tickers=symbols, lookback_hours=lookback_hours)
            payload = {k: [i.model_dump(mode="json") for i in v] for k, v in result.items()}
            return json.dumps(payload, default=str)

        return [fetch_news, search_news, ticker_news]


class MarketAgentTools(Toolkit):
    """Market data tools exposed to Agno Agent 2."""

    def __init__(self, settings: Settings, **kwargs):
        self._settings = settings
        super().__init__(name="market_agent_tools", tools=self._build_tools(), **kwargs)

    def _build_tools(self) -> list:
        settings = self._settings

        @tool(
            name="user_market_lookup",
            description="Look up OHLCV bars and technical indicators for tickers in a natural-language query.",
        )
        async def market_lookup(
            query: str,
            start_date: str | None = None,
            end_date: str | None = None,
            interval: str = "1d",
        ) -> str:
            result = await user_market_lookup(
                query,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
            )
            return json.dumps(result, default=str)

        @tool(
            name="build_market_snapshot",
            description="Build a market snapshot with indicators for a comma-separated watchlist.",
        )
        async def market_snapshot(watchlist: str) -> str:
            tickers = [t.strip().upper() for t in watchlist.split(",") if t.strip()]
            snap = await build_market_snapshot(tickers)
            return json.dumps(snap.model_dump(mode="json"), default=str)

        return [market_lookup, market_snapshot]


class StrategyAgentTools(Toolkit):
    """Optional research tools for Agno Agent 3 (delegates to other agents via pipeline)."""

    def __init__(self, **kwargs):
        super().__init__(name="strategy_agent_tools", tools=[], **kwargs)
