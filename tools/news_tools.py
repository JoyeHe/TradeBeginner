"""News tools with TrendRadar-aware search grammar handling."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
from typing import Any, Optional

import feedparser
import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import Settings
from schemas.news import NewsItem, SentimentDigest

logger = structlog.get_logger(__name__)

SENTIMENT_ANALYSIS_PROMPT = """You are a financial sentiment classifier.
Return strict JSON array with: news_id, sentiment_score(-1..1), relevance_score(0..1), category, tickers_mentioned.
"""

DIGEST_SYNTHESIS_PROMPT = """Create a market sentiment digest from enriched items.
Return strict JSON object: overall_market_sentiment, sector_sentiments, top_positive_ids, top_negative_ids, trending_tickers, macro_summary.
"""

BREAKING_EVENT_PROMPT = """Assess urgency for each event. Return urgency_level in {low,medium,high,critical} with reasoning."""


def parse_trendradar_search_request(user_query: str) -> dict[str, Any]:
    """Interpret user text into TrendRadar search parameters."""
    q = user_query.strip()
    mode = "keyword"
    if q.lower().startswith("entity:"):
        mode, q = "entity", q.split(":", 1)[1].strip()
    elif q.lower().startswith("fuzzy:"):
        mode, q = "fuzzy", q.split(":", 1)[1].strip()
    return {
        "query": q,
        "search_mode": mode,
        "sort_by": "relevance",
        "include_url": False,
    }


def _to_news_item(raw: dict[str, Any], source: str) -> NewsItem:
    title = (raw.get("title") or "").strip()
    summary = (raw.get("description") or raw.get("summary") or title).strip()
    url = raw.get("url") or raw.get("link")
    published = raw.get("publishedAt") or raw.get("datetime") or raw.get("published")
    if isinstance(published, str):
        try:
            published_at = datetime.fromisoformat(published.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            published_at = parsedate_to_datetime(published).astimezone(timezone.utc)
    else:
        published_at = datetime.now(tz=timezone.utc)
    news_id = hashlib.sha1(f"{source}:{title}:{url}".encode("utf-8")).hexdigest()[:16]
    return NewsItem(
        news_id=news_id,
        source=source,
        title=title,
        summary=summary,
        full_text=raw.get("content"),
        url=url,
        published_at=published_at.replace(tzinfo=None),
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def _get_json(url: str, params: dict, headers: Optional[dict] = None) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        return response.json()


async def fetch_news_feed(
    settings: Settings,
    tickers: Optional[list[str]] = None,
    lookback_hours: int = 24,
    max_items: int = 50,
    rss_urls: Optional[list[str]] = None,
) -> list[NewsItem]:
    """Fetch news from NewsAPI, Finnhub, and RSS with graceful degradation."""
    items: list[NewsItem] = []
    since = datetime.now(tz=timezone.utc) - timedelta(hours=lookback_hours)

    if settings.news_api_key:
        try:
            query = " OR ".join(tickers) if tickers else "stock market OR crypto OR earnings"
            payload = await _get_json(
                "https://newsapi.org/v2/everything",
                {"q": query, "from": since.isoformat(), "sortBy": "publishedAt", "pageSize": max_items},
                headers={"X-Api-Key": settings.news_api_key},
            )
            items.extend(_to_news_item(article, "newsapi") for article in payload.get("articles", []))
        except Exception as exc:
            logger.warning("newsapi_fetch_failed", error=str(exc))

    if settings.finnhub_api_key:
        try:
            for ticker in (tickers or ["AAPL", "SPY"])[:5]:
                payload = await _get_json(
                    "https://finnhub.io/api/v1/company-news",
                    {
                        "symbol": ticker,
                        "from": (since.date().isoformat()),
                        "to": datetime.now(tz=timezone.utc).date().isoformat(),
                        "token": settings.finnhub_api_key,
                    },
                )
                items.extend(_to_news_item(article, "finnhub") for article in payload)
        except Exception as exc:
            logger.warning("finnhub_fetch_failed", error=str(exc))

    for rss_url in (rss_urls or []):
        try:
            parsed = await asyncio.to_thread(feedparser.parse, rss_url)
            for entry in parsed.entries[: max_items // max(1, len(rss_urls or [1]))]:
                items.append(
                    _to_news_item(
                        {"title": entry.get("title"), "summary": entry.get("summary"), "link": entry.get("link"), "published": entry.get("published")},
                        "rss",
                    )
                )
        except Exception as exc:
            logger.warning("rss_fetch_failed", rss_url=rss_url, error=str(exc))

    try:
        tr_query = " OR ".join(tickers) if tickers else "stock market OR earnings OR 股市 OR 股票"
        tr_items = await fetch_from_trendradar(
            {
                "query": tr_query,
                "search_mode": "fuzzy",
                "sort_by": "relevance",
                "include_url": True,
            },
            max_items=max_items,
        )
        items.extend(tr_items)
    except Exception as exc:
        logger.warning("trendradar_background_fetch_failed", error=str(exc))

    # Cross-source dedup should not rely on source-prefixed news_id.
    dedup: dict[tuple[str, str], NewsItem] = {}
    for item in items:
        normalized_title = (item.title or "").strip().lower()
        normalized_url = (item.url or "").strip().lower()
        key = (normalized_title, normalized_url)
        existing = dedup.get(key)
        if existing is None or item.published_at > existing.published_at:
            dedup[key] = item

    filtered = [item for item in dedup.values() if item.published_at >= since.replace(tzinfo=None)]
    return sorted(filtered, key=lambda i: i.published_at, reverse=True)[:max_items]


async def analyze_sentiment(items: list[NewsItem]) -> list[NewsItem]:
    """Assign sentiment/relevance/category with lightweight heuristic baseline."""
    positive = ("beats", "growth", "surge", "strong", "upgrade", "rally")
    negative = ("miss", "drop", "downgrade", "lawsuit", "war", "recession")
    for item in items:
        text = f"{item.title} {item.summary}".lower()
        pos_hits = sum(1 for w in positive if w in text)
        neg_hits = sum(1 for w in negative if w in text)
        item.sentiment_score = max(-1.0, min(1.0, (pos_hits - neg_hits) / 3.0))
        item.relevance_score = max(0.1, min(1.0, 0.2 + (pos_hits + neg_hits) * 0.2))
        if "earnings" in text:
            item.category = "earnings"
        elif any(k in text for k in ("fed", "inflation", "rate")):
            item.category = "macro"
        elif any(k in text for k in ("oil", "war", "sanction")):
            item.category = "geopolitical"
        else:
            item.category = "company-specific"
        item.tickers_mentioned = [tok for tok in item.title.split() if tok.isupper() and 1 < len(tok) <= 5][:4]
    return items


async def generate_sentiment_digest(items: list[NewsItem]) -> SentimentDigest:
    """Synthesize a digest used by strategy generation."""
    if not items:
        return SentimentDigest(timestamp=datetime.utcnow(), overall_market_sentiment=0.0, macro_summary="No news available.")
    overall = sum((x.sentiment_score or 0.0) * (x.relevance_score or 1.0) for x in items) / len(items)
    positives = sorted(items, key=lambda x: (x.sentiment_score or 0.0), reverse=True)[:5]
    negatives = sorted(items, key=lambda x: (x.sentiment_score or 0.0))[:5]
    sectors: dict[str, float] = {}
    for item in items:
        sector = item.category or "other"
        sectors.setdefault(sector, 0.0)
        sectors[sector] += item.sentiment_score or 0.0
    trending = sorted({t for i in items for t in i.tickers_mentioned})
    return SentimentDigest(
        timestamp=datetime.utcnow(),
        overall_market_sentiment=max(-1.0, min(1.0, overall)),
        sector_sentiments=sectors,
        top_positive_events=positives,
        top_negative_events=negatives,
        trending_tickers=trending[:20],
        macro_summary="Automated digest generated from blended source sentiment.",
    )


async def detect_breaking_events(items: list[NewsItem]) -> list[NewsItem]:
    """Detect high urgency events by recency and sentiment extremes."""
    now = datetime.utcnow()
    out: list[NewsItem] = []
    for item in items:
        age_minutes = (now - item.published_at).total_seconds() / 60.0
        score = abs(item.sentiment_score or 0.0)
        if age_minutes <= 60 and score >= 0.8:
            item.metadata["urgency_level"] = "high"
            out.append(item)
    return out


async def get_ticker_news(
    settings: Settings, tickers: list[str], lookback_hours: int = 24, max_items: int = 30
) -> dict[str, list[NewsItem]]:
    """Get ticker-specific news map."""
    result: dict[str, list[NewsItem]] = {}
    for ticker in tickers:
        batch = await fetch_news_feed(settings=settings, tickers=[ticker], lookback_hours=lookback_hours, max_items=max_items)
        result[ticker] = await analyze_sentiment(batch)
    return result


_TRENDRADAR_ROOT: Optional[str] = None


def _find_trendradar_root() -> Optional[str]:
    """Locate the TrendRadar project root on disk."""
    global _TRENDRADAR_ROOT
    if _TRENDRADAR_ROOT is not None:
        return _TRENDRADAR_ROOT
    import pathlib
    candidates = [
        pathlib.Path(__file__).resolve().parent.parent.parent / "TrendRadar",
        pathlib.Path("/app/TrendRadar"),
        pathlib.Path.home() / "TrendRadar",
    ]
    for c in candidates:
        if (c / "mcp_server" / "tools" / "search_tools.py").exists():
            _TRENDRADAR_ROOT = str(c)
            return _TRENDRADAR_ROOT
    return None


async def fetch_from_trendradar(
    trendradar_params: dict[str, Any],
    max_items: int = 20,
) -> list[NewsItem]:
    """Call TrendRadar SearchTools directly (in-process) and convert results to NewsItem."""
    root = _find_trendradar_root()
    if root is None:
        logger.warning("trendradar_not_found", msg="TrendRadar project directory not located")
        return []

    try:
        items, rss_direct_count = await asyncio.to_thread(
            _fetch_from_trendradar_sync, root, trendradar_params, max_items
        )
    except Exception as exc:
        logger.warning("trendradar_search_failed", error=str(exc))
        return []

    logger.info(
        "trendradar_fetch_completed",
        items=len(items),
        hot=len(items) - rss_direct_count,
        rss=rss_direct_count,
    )
    return items


def _fetch_from_trendradar_sync(root: str, trendradar_params: dict[str, Any], max_items: int) -> tuple[list[NewsItem], int]:
    import sys

    if root not in sys.path:
        sys.path.insert(0, root)
    from mcp_server.tools.search_tools import SearchTools

    tools = SearchTools(project_root=root)
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    three_days_ago = (datetime.utcnow() - timedelta(days=3)).strftime("%Y-%m-%d")
    data = tools.search_news_unified(
        query=trendradar_params.get("query", ""),
        search_mode=trendradar_params.get("search_mode", "fuzzy"),
        date_range={"start": three_days_ago, "end": today_str},
        sort_by=trendradar_params.get("sort_by", "relevance"),
        threshold=0.2,
        include_url=trendradar_params.get("include_url", True),
        include_rss=True,
        limit=max_items,
        rss_limit=max_items,
    )

    raw_results = []
    if isinstance(data, dict):
        raw_results = data.get("data", data.get("results", data.get("items", data.get("hot_results", []))))
        rss = data.get("rss", data.get("rss_results", []))
        if isinstance(rss, list):
            raw_results = list(raw_results) + rss
    elif isinstance(data, list):
        raw_results = data

    items: list[NewsItem] = []
    for entry in raw_results[:max_items]:
        title = (entry.get("title") or "").strip()
        if not title:
            continue
        published_str = entry.get("published") or entry.get("date") or entry.get("time")
        if isinstance(published_str, str):
            try:
                pub = datetime.fromisoformat(published_str.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                pub = datetime.utcnow()
        else:
            pub = datetime.utcnow()
        news_id = hashlib.sha1(f"trendradar:{title}".encode("utf-8")).hexdigest()[:16]
        items.append(
            NewsItem(
                news_id=news_id,
                source="trendradar",
                title=title,
                summary=(entry.get("summary") or entry.get("description") or title).strip(),
                full_text=entry.get("content"),
                url=entry.get("url") or entry.get("link"),
                published_at=pub,
                metadata={"platform": entry.get("platform", "unknown"), "weight": entry.get("weight")},
            )
        )
    rss_direct = _search_trendradar_rss_direct(root, trendradar_params.get("query", ""), max_items)
    seen_ids = {i.news_id for i in items}
    items.extend(i for i in rss_direct if i.news_id not in seen_ids)
    return items, len(rss_direct)


def _search_trendradar_rss_direct(root: str, query: str, max_items: int) -> list[NewsItem]:
    """Direct SQLite search on TrendRadar RSS DB for reliable keyword matching."""
    import sqlite3
    import pathlib

    rss_dir = pathlib.Path(root) / "output" / "rss"
    if not rss_dir.exists():
        return []

    db_files = sorted(rss_dir.glob("*.db"), reverse=True)[:3]
    items: list[NewsItem] = []
    keywords = [w.strip().lower().lstrip("+") for w in query.split() if len(w.strip()) > 1 and not w.startswith("!")]
    if not keywords:
        return []

    for db_path in db_files:
        try:
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("SELECT id, title, feed_id, url, published_at, summary, created_at FROM rss_items ORDER BY id DESC").fetchall()
            for row in rows:
                row_title = row[1] or ""
                title_lower = row_title.lower()
                if any(kw in title_lower for kw in keywords):
                    pub = datetime.utcnow()
                    try:
                        pub_str = row[4] or row[6]
                        if pub_str:
                            pub = datetime.fromisoformat(str(pub_str).replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        pass
                    nid = hashlib.sha1(f"trendradar-rss:{row_title}".encode("utf-8")).hexdigest()[:16]
                    feed_id = row[2] or "?"
                    items.append(
                        NewsItem(
                            news_id=nid,
                            source="trendradar",
                            title=row_title,
                            summary=row[5] or row_title,
                            url=row[3],
                            published_at=pub,
                            metadata={"platform": f"rss:{feed_id}", "weight": None},
                        )
                    )
                    if len(items) >= max_items:
                        break
            conn.close()
        except Exception as exc:
            logger.warning("trendradar_rss_direct_failed", db=str(db_path), error=str(exc))
            continue
        if len(items) >= max_items:
            break
    return items[:max_items]


TRENDRADAR_GRAMMAR_PROMPT = """You are an expert at converting natural-language news queries into TrendRadar search parameters.

CRITICAL: TrendRadar crawls news from BOTH English AND Chinese sources (Hacker News, Zhihu, Weibo, Douyin, Baidu, Bilibili, Toutiao, etc).
You MUST generate BILINGUAL keywords — include both English AND Chinese terms so the search matches headlines in either language.

TrendRadar search supports these modes:
- keyword: substring match (default). Use for general topics.
- fuzzy: similarity search. Use when the query is vague or broad.
- entity: exact name match. Use for specific company/person/brand names.

TrendRadar keyword grammar for the query field:
- Plain words are OR-matched against headlines (any word match = hit).
- +word means the word is REQUIRED.
- !word means EXCLUDE headlines containing this word.
- /regex/ for regex patterns (e.g. /\\bAI\\b/i for exact "AI").
- Separate multiple terms with spaces.
- For bilingual search, list BOTH English and Chinese synonyms as OR terms.

Your job: given a user's natural language query, output a JSON object with:
{
  "query": "<TrendRadar grammar with BOTH English AND Chinese keywords>",
  "search_mode": "keyword" | "fuzzy" | "entity",
  "tickers": ["<any stock tickers mentioned, e.g. AAPL, BTC-USD>"],
  "sort_by": "relevance" | "date" | "weight"
}

Bilingual keyword mapping (always apply these):
- BTC/Bitcoin → 比特币
- ETH/Ethereum → 以太坊
- SOL/Solana → 索拉纳
- stock market → 股市 股票 A股
- crypto/cryptocurrency → 加密货币 虚拟货币 数字货币
- AI/artificial intelligence → 人工智能 大模型
- Apple/AAPL → 苹果
- Tesla/TSLA → 特斯拉
- NVIDIA → 英伟达
- Microsoft → 微软
- Google → 谷歌
- Chinese market → 中国市场 A股 中概股 港股
- earnings → 财报 业绩
- Federal Reserve/Fed → 美联储 央行
- inflation → 通胀 通货膨胀
- interest rate → 利率 加息 降息

Examples:
- User: "latest news about BTC" → {"query": "BTC Bitcoin 比特币 crypto 加密货币", "search_mode": "fuzzy", "tickers": ["BTC-USD"], "sort_by": "date"}
- User: "stock performs best in Chinese Market" → {"query": "stock 股票 A股 Chinese market 中国市场 港股 中概股", "search_mode": "fuzzy", "tickers": [], "sort_by": "relevance"}
- User: "NVIDIA earnings report" → {"query": "NVIDIA 英伟达 earnings 财报 业绩", "search_mode": "fuzzy", "tickers": ["NVDA"], "sort_by": "date"}
- User: "AI technology trends" → {"query": "AI 人工智能 大模型 technology 科技", "search_mode": "fuzzy", "tickers": [], "sort_by": "relevance"}
- User: "Fed interest rate decision" → {"query": "Fed 美联储 interest rate 利率 加息 降息", "search_mode": "fuzzy", "tickers": [], "sort_by": "date"}
- User: "Apple stock price" → {"query": "Apple 苹果 AAPL stock 股票", "search_mode": "fuzzy", "tickers": ["AAPL"], "sort_by": "date"}

Return ONLY the JSON object, no explanation."""


async def _llm_interpret_query(settings: Settings, user_query: str) -> dict[str, Any]:
    """Use LLM to interpret a user query into TrendRadar search params + tickers."""
    import json as _json

    from agents.common import llm_chat

    raw = await llm_chat(
        settings,
        system_prompt=TRENDRADAR_GRAMMAR_PROMPT,
        user_prompt=f"User query: \"{user_query}\"",
        temperature=0.3,
    )
    if not raw:
        return {"query": user_query, "search_mode": "fuzzy", "tickers": [], "sort_by": "relevance"}
    try:
        import re as _re
        json_match = _re.search(r"\{[\s\S]*\}", raw)
        if json_match:
            return _json.loads(json_match.group())
    except Exception:
        pass
    return {"query": user_query, "search_mode": "fuzzy", "tickers": [], "sort_by": "relevance"}


async def _llm_rank_relevance(
    settings: Settings,
    user_query: str,
    items: list[NewsItem],
    max_keep: int = 30,
) -> list[NewsItem]:
    """Use LLM to score each item's relevance to the user query (0-10).

    Removes items scored below 3 and sorts by relevance descending.
    Falls back to returning all items if LLM fails.
    """
    if not items:
        return items

    from agents.common import llm_chat
    import json as _json

    batch_size = 40
    scored: list[tuple[NewsItem, float]] = []

    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        titles_block = "\n".join(f"{i}: {item.title}" for i, item in enumerate(batch))
        raw = await llm_chat(
            settings,
            system_prompt=(
                "You are a relevance judge. Given a user query and a list of news headlines, "
                "rate each headline's relevance to the query on a scale of 0-10.\n"
                "0 = completely irrelevant, 10 = directly about the query topic.\n"
                "Return ONLY a JSON array of integers, one score per headline, same order.\n"
                "Example: [8, 2, 0, 9, 1]"
            ),
            user_prompt=f"User query: \"{user_query}\"\n\nHeadlines:\n{titles_block}",
            temperature=0.1,
        )
        try:
            import re as _re
            m = _re.search(r"\[[\d\s,]+\]", raw)
            if m:
                scores = _json.loads(m.group())
                for item, score in zip(batch, scores):
                    scored.append((item, float(score)))
                continue
        except Exception:
            pass
        for item in batch:
            scored.append((item, 5.0))

    relevant = [(item, s) for item, s in scored if s >= 3]
    relevant.sort(key=lambda x: x[1], reverse=True)

    result: list[NewsItem] = []
    for item, score in relevant[:max_keep]:
        item.relevance_score = round(score / 10.0, 2)
        result.append(item)

    logger.info("llm_rank_relevance", query=user_query, input_count=len(items), output_count=len(result))
    return result


async def _fetch_live_rss(query_keywords: list[str], max_items: int = 30) -> list[NewsItem]:
    """Fetch RSS feeds LIVE and filter by user's keywords at fetch time."""
    RSS_FEEDS = [
        ("yahoo-finance", "https://finance.yahoo.com/news/rssindex"),
        ("yahoo-finance-stocks", "https://finance.yahoo.com/rss/topstories"),
        ("cnbc-top", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"),
        ("cnbc-finance", "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664"),
        ("coindesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("cointelegraph", "https://cointelegraph.com/rss"),
        ("marketwatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories"),
        ("wsj-markets", "https://feeds.a.dj.com/rss/RSSMarketsMain.xml"),
        ("seekingalpha", "https://seekingalpha.com/market_currents.xml"),
    ]
    kw_lower = [k.lower() for k in query_keywords if len(k) > 1]
    items: list[NewsItem] = []
    for feed_id, url in RSS_FEEDS:
        try:
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:50]:
                title = (entry.get("title") or "").strip()
                if not title:
                    continue
                title_lower = title.lower()
                if not any(kw in title_lower for kw in kw_lower):
                    continue
                pub = datetime.utcnow()
                pub_str = entry.get("published")
                if isinstance(pub_str, str):
                    try:
                        pub = parsedate_to_datetime(pub_str).replace(tzinfo=None)
                    except Exception:
                        pass
                nid = hashlib.sha1(f"live-rss:{feed_id}:{title}".encode("utf-8")).hexdigest()[:16]
                items.append(
                    NewsItem(
                        news_id=nid,
                        source="trendradar",
                        title=title,
                        summary=(entry.get("summary") or title)[:300],
                        url=entry.get("link"),
                        published_at=pub,
                        metadata={"platform": f"rss:{feed_id}", "weight": None},
                    )
                )
        except Exception as exc:
            logger.warning("live_rss_fetch_failed", feed=feed_id, error=str(exc))
    items.sort(key=lambda x: x.published_at, reverse=True)
    logger.info("live_rss_fetch_completed", keywords=kw_lower[:5], items=len(items))
    return items[:max_items]


async def search_news_with_llm(
    settings: Settings,
    user_query: str,
    lookback_hours: int = 72,
    max_items: int = 30,
) -> dict[str, Any]:
    """User-triggered news search. Every search is fresh and targeted.

    Pipeline:
    1. LLM interprets query → bilingual keywords + tickers
    2. NewsAPI: search user's query text directly (live)
    3. Finnhub: fetch news for extracted tickers specifically (live)
    4. TrendRadar DB: search crawled hot-rankings for bilingual keywords
    5. Live RSS: fetch finance RSS feeds and filter by keywords (live)
    6. LLM ranks TrendRadar/RSS results for relevance
    7. Sentiment enrichment on all results
    """
    interpreted = await _llm_interpret_query(settings, user_query)
    trendradar_params = {
        "query": interpreted.get("query", user_query),
        "search_mode": interpreted.get("search_mode", "keyword"),
        "sort_by": interpreted.get("sort_by", "relevance"),
        "include_url": True,
    }
    tickers = interpreted.get("tickers", [])
    bilingual_keywords = [w for w in interpreted.get("query", user_query).split() if len(w) > 1]

    # --- Source 1 & 2: Finnhub + NewsAPI (live, targeted to user query) ---
    api_items = await fetch_news_feed(
        settings=settings,
        tickers=tickers or None,
        lookback_hours=lookback_hours,
        max_items=max_items,
    )
    if settings.news_api_key and not tickers:
        try:
            payload = await _get_json(
                "https://newsapi.org/v2/everything",
                {"q": user_query, "sortBy": "publishedAt", "pageSize": max_items},
                headers={"X-Api-Key": settings.news_api_key},
            )
            seen = {i.news_id for i in api_items}
            for article in payload.get("articles", []):
                item = _to_news_item(article, "newsapi")
                if item.news_id not in seen:
                    api_items.append(item)
        except Exception as exc:
            logger.warning("newsapi_direct_query_failed", error=str(exc))

    # --- Source 3: TrendRadar DB (hot rankings) ---
    fetch_limit = max(max_items * 3, 100)
    trendradar_items = await fetch_from_trendradar(trendradar_params, max_items=fetch_limit)

    # --- Source 4: Live RSS feeds (fresh, keyword-filtered) ---
    live_rss = await _fetch_live_rss(bilingual_keywords, max_items=max_items)
    seen_tr = {i.news_id for i in trendradar_items}
    trendradar_items.extend(i for i in live_rss if i.news_id not in seen_tr)

    # --- LLM relevance ranking on TrendRadar + RSS results ---
    if trendradar_items:
        trendradar_items = await _llm_rank_relevance(settings, user_query, trendradar_items, max_keep=max_items)

    # --- Dedup across all sources ---
    dedup: dict[tuple[str, str], NewsItem] = {}
    for item in trendradar_items + api_items:
        key = ((item.title or "").strip().lower(), (item.url or "").strip().lower())
        existing = dedup.get(key)
        if existing is None or item.published_at > existing.published_at:
            dedup[key] = item

    tr_final = [i for i in dedup.values() if i.source == "trendradar"]
    api_final = [i for i in dedup.values() if i.source != "trendradar"]
    tr_final.sort(key=lambda x: x.relevance_score or 0, reverse=True)
    api_final.sort(key=lambda x: x.published_at, reverse=True)
    merged = tr_final[:max_items] + api_final[:max_items]

    enriched = await analyze_sentiment(merged)
    digest = await generate_sentiment_digest(enriched)

    source_counts: dict[str, int] = {}
    for item in enriched:
        source_counts[item.source] = source_counts.get(item.source, 0) + 1

    tr_enriched = [i.model_dump(mode="json") for i in enriched if i.source == "trendradar"]
    api_enriched = [i.model_dump(mode="json") for i in enriched if i.source != "trendradar"]

    return {
        "query": user_query,
        "llm_interpretation": interpreted,
        "trendradar_params": trendradar_params,
        "tickers_detected": tickers,
        "items": [item.model_dump(mode="json") for item in enriched],
        "trendradar_items": tr_enriched,
        "api_items": api_enriched,
        "item_count": len(enriched),
        "source_breakdown": source_counts,
        "digest": digest.model_dump(mode="json"),
    }

