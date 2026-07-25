"""Agent 2: Market data and technical analysis."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta

import structlog
from agno.agent import Agent

from agents.common import build_agno_model
from config.settings import Settings
from memory import MemoryManager
from schemas.market_data import MarketSnapshot
from tools.agno_tools import MarketAgentTools
from tools.market_tools import build_market_snapshot, compute_technical_indicators, fetch_price_data, get_historical_data_for_backtest, user_market_lookup

logger = structlog.get_logger(__name__)


class Agent2Market:
    """Maintains market snapshot and technical context."""

    def __init__(self, settings: Settings, memory: MemoryManager):
        self.settings = settings
        self.memory = memory
        self.market_tools = MarketAgentTools(settings)
        self.agent = Agent(
            name="agent2_market",
            model=build_agno_model(settings),
            tools=[self.market_tools],
            instructions=[
                "Maintain up-to-date market snapshot.",
                "Compute technical indicators for watchlist.",
                "Detect market regime evidence.",
                "Provide data for strategy and backtesting.",
            ],
            markdown=True,
        )

    async def run_market_update(self, watchlist: list[str]) -> MarketSnapshot:
        """Run full update cycle and store in working memory."""
        started = datetime.utcnow()
        snapshot = await build_market_snapshot(watchlist)
        await self.memory.working.set("market_snapshot", snapshot, category="market")
        logger.info("agent2_market_update_completed", assets=len(snapshot.assets), ms=(datetime.utcnow() - started).total_seconds() * 1000)
        return snapshot

    async def analyze_ticker(self, ticker: str) -> dict:
        """Return deep analysis details for one ticker."""
        bars_map = await fetch_price_data([ticker], date.today() - timedelta(days=300), date.today())
        bars = bars_map.get(ticker, [])
        if not bars:
            return {"ticker": ticker, "error": "no_data"}
        indicators = await compute_technical_indicators(bars)
        return {"ticker": ticker, "last_bar": bars[-1].model_dump(mode="json"), "indicators": indicators.model_dump(mode="json")}

    async def get_backtest_data(self, tickers: list[str], start_date: date, end_date: date) -> dict:
        """Fetch historical data in backtester format."""
        return await get_historical_data_for_backtest(tickers, start_date=start_date, end_date=end_date)

    async def user_lookup(
        self,
        query: str,
        start_date: str | None = None,
        end_date: str | None = None,
        interval: str = "1d",
    ) -> dict:
        """Handle a user's free-form market data lookup request.

        Extracts tickers, fetches price data + indicators for the specified
        date window and interval, and stores the result in working memory.
        """
        result = await user_market_lookup(query, start_date=start_date, end_date=end_date, interval=interval)
        await self.memory.working.set("user_market_search_latest", result, category="market")
        logger.info("agent2_user_lookup_completed", query=query, tickers=result.get("tickers"), interval=interval)
        return result


async def market_monitoring_loop(agent2: Agent2Market, watchlist: list[str], interval_minutes: int = 5) -> None:
    """Periodic market monitoring loop."""
    while True:
        try:
            await agent2.run_market_update(watchlist=watchlist)
        except Exception as exc:
            logger.error("agent2_loop_error", error=str(exc))
        await asyncio.sleep(interval_minutes * 60)

