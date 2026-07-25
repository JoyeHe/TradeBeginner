"""Agent 1: News and sentiment intelligence."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

import structlog
from agno.agent import Agent

from agents.common import build_agno_model
from config.settings import Settings
from memory import MemoryManager
from schemas.news import NewsItem, SentimentDigest
from tools.agno_tools import NewsAgentTools
from tools.news_tools import analyze_sentiment, detect_breaking_events, fetch_news_feed, generate_sentiment_digest, get_ticker_news, search_news_with_llm

logger = structlog.get_logger(__name__)


class Agent1News:
    """Monitors and synthesizes market news for downstream agents."""

    def __init__(self, settings: Settings, memory: MemoryManager):
        self.settings = settings
        self.memory = memory
        self.news_tools = NewsAgentTools(settings)
        self.agent = Agent(
            name="agent1_news",
            model=build_agno_model(settings),
            tools=[self.news_tools],
            instructions=[
                "Fetch market news from configured sources.",
                "Assign sentiment and relevance scores.",
                "Detect urgent market-moving events.",
                "Persist digest to working memory.",
                "Store raw payloads in perceptual memory.",
            ],
            markdown=True,
        )

    async def run_news_cycle(self) -> SentimentDigest:
        """Execute full news cycle and write outputs to memory."""
        started = datetime.utcnow()
        raw_items = await fetch_news_feed(self.settings)
        for item in raw_items:
            await self.memory.perceptual.ingest("news_raw", item.model_dump(mode="json"), ttl_seconds=900)
        enriched = await analyze_sentiment(raw_items)
        digest = await generate_sentiment_digest(enriched)
        breaking = await detect_breaking_events(enriched)
        await self.memory.working.set("news_digest", digest, category="news")
        await self.memory.working.set("breaking_events", breaking, category="news")
        logger.info("agent1_news_cycle_completed", item_count=len(enriched), breaking_count=len(breaking), ms=(datetime.utcnow() - started).total_seconds() * 1000)
        return digest

    async def research_tickers(self, tickers: list[str]) -> dict[str, list[NewsItem]]:
        """Run ticker-focused research used by strategy generation."""
        result = await get_ticker_news(self.settings, tickers=tickers, lookback_hours=72, max_items=50)
        await self.memory.working.set("ticker_news", result, category="news")
        return result

    async def _llm_summarize(self, prompt: str) -> str:
        """Ask the LLM for a natural-language summary via direct SDK call."""
        from agents.common import llm_chat

        return await llm_chat(
            self.settings,
            system_prompt="You are a financial news analyst. Provide concise, actionable summaries.",
            user_prompt=prompt,
        )

    async def user_search(self, query: str) -> dict:
        """Handle a user's free-form news search request.

        Interprets the query, fetches matching news via TrendRadar (primary)
        and Finnhub/NewsAPI (secondary), enriches with sentiment, generates
        a natural-language summary via LLM, and records the interaction into
        all applicable memory tiers.
        """
        started = datetime.utcnow()
        result = await search_news_with_llm(settings=self.settings, user_query=query)

        top_titles = [item.get("title", "") for item in result.get("items", [])[:8]]
        digest_data = result.get("digest", {})
        summary_prompt = (
            f"The user searched for: \"{query}\"\n\n"
            f"We found {result['item_count']} articles from sources: {result.get('source_breakdown', {})}.\n"
            f"Overall market sentiment: {digest_data.get('overall_market_sentiment', 'N/A')}\n"
            f"Top headlines:\n" + "\n".join(f"- {t}" for t in top_titles) + "\n\n"
            "Write a concise 3-5 sentence summary interpreting these results for the user. "
            "Highlight key themes, sentiment direction, and any actionable takeaways."
        )
        result["llm_summary"] = await self._llm_summarize(summary_prompt)

        await self.memory.working.set("user_news_search_latest", result, category="news")
        await self.memory.perceptual.ingest("user_news_search", result, ttl_seconds=1800)
        try:
            from schemas.memory import EpisodicTrace
            import uuid

            trace = EpisodicTrace(
                trace_id=str(uuid.uuid4()),
                strategy={"type": "user_news_search", "query": query},
                user_modification=None,
                execution_result={"item_count": result["item_count"]},
                reward=None,
                context={"source": "user_search", "query": query},
            )
            await self.memory.episodic.store_episode(trace)
        except Exception as exc:
            logger.warning("user_search_episodic_store_failed", error=str(exc))
        elapsed = (datetime.utcnow() - started).total_seconds() * 1000
        logger.info("agent1_user_search_completed", query=query, item_count=result["item_count"], ms=elapsed)
        return result


async def news_monitoring_loop(agent1: Agent1News, interval_minutes: int = 15) -> None:
    """Periodic monitoring loop for Agent 1."""
    while True:
        try:
            await agent1.run_news_cycle()
        except Exception as exc:
            logger.error("agent1_loop_error", error=str(exc))
        await asyncio.sleep(interval_minutes * 60)

