"""Helpers to build analysis inputs and heuristic fallbacks."""

from __future__ import annotations

from typing import Any, Optional

from schemas.analysis import FlowType, MarketAnalysis, MarketOutlook, OutlookDirection, SentimentView
from schemas.news import NewsItem, SentimentDigest


def build_news_bundle_from_search(result: dict) -> dict:
    """Normalize Agent1 user_search output for analysis."""
    items = result.get("items") or []
    headlines = []
    for i, item in enumerate(items[:20]):
        nid = item.get("news_id") or f"h{i}"
        headlines.append(
            {
                "news_id": nid,
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "sentiment_score": item.get("sentiment_score"),
                "category": item.get("category"),
                "url": item.get("url"),
            }
        )
    digest = result.get("digest") or {}
    return {
        "item_count": result.get("item_count", len(items)),
        "source_breakdown": result.get("source_breakdown", {}),
        "llm_summary": result.get("llm_summary", ""),
        "overall_sentiment": digest.get("overall_market_sentiment", 0.0),
        "headlines": headlines,
        "key_themes": _extract_themes(headlines),
    }


def build_news_bundle_from_digest(digest: SentimentDigest, breaking: list[NewsItem] | None = None) -> dict:
    headlines = []
    for ev in (digest.top_positive_events or [])[:5] + (digest.top_negative_events or [])[:5]:
        headlines.append(
            {
                "news_id": ev.news_id,
                "title": ev.title,
                "source": ev.source,
                "sentiment_score": ev.sentiment_score,
                "category": ev.category,
                "url": ev.url,
            }
        )
    return {
        "item_count": len(headlines),
        "source_breakdown": {"background_cycle": len(headlines)},
        "llm_summary": digest.macro_summary,
        "overall_sentiment": digest.overall_market_sentiment,
        "headlines": headlines,
        "key_themes": digest.trending_tickers[:5] if digest.trending_tickers else _extract_themes(headlines),
    }


def build_market_evidence_from_lookup(market_data: dict) -> dict:
    """Build evidence dict from Agent2 user_lookup / market snapshot data."""
    evidence: dict[str, Any] = {"tickers": {}, "regime": "uncertain"}
    data = market_data.get("data") if isinstance(market_data, dict) and "data" in market_data else market_data
    if isinstance(data, dict):
        for ticker, payload in data.items():
            if not isinstance(payload, dict):
                continue
            ind = payload.get("indicators") or {}
            bar = payload.get("last_bar") or {}
            evidence["tickers"][ticker] = {
                "close": bar.get("close"),
                "rsi_14": ind.get("rsi_14"),
                "sma_20": ind.get("sma_20"),
                "macd": ind.get("macd"),
                "adx": ind.get("adx"),
                "volume": bar.get("volume"),
            }
    return evidence


def build_market_evidence_from_snapshot(snapshot) -> dict:
    if snapshot is None:
        return {"tickers": {}, "regime": "uncertain"}
    dumped = snapshot.model_dump(mode="json") if hasattr(snapshot, "model_dump") else snapshot
    regime = (dumped.get("market_breadth") or {}).get("regime", "uncertain")
    tickers = {}
    assets = dumped.get("assets") or {}
    indicators = dumped.get("indicators") or {}
    for tk, bar in assets.items():
        ind = indicators.get(tk) or {}
        tickers[tk] = {
            "close": bar.get("close") if isinstance(bar, dict) else getattr(bar, "close", None),
            "rsi_14": ind.get("rsi_14") if isinstance(ind, dict) else getattr(ind, "rsi_14", None),
            "volume": bar.get("volume") if isinstance(bar, dict) else getattr(bar, "volume", None),
        }
    return {"tickers": tickers, "regime": regime, "vix": dumped.get("vix")}


def heuristic_market_analysis(
    query: str,
    news_bundle: dict,
    market_evidence: dict,
    user_id: str = "default",
    flow_type: FlowType = FlowType.BASELINE,
    parent_analysis_id: Optional[str] = None,
) -> MarketAnalysis:
    """Deterministic analysis when LLM is unavailable."""
    score = float(news_bundle.get("overall_sentiment") or 0.0)
    score = max(-1.0, min(1.0, score))
    if score > 0.15:
        direction = OutlookDirection.BULLISH
    elif score < -0.15:
        direction = OutlookDirection.BEARISH
    else:
        direction = OutlookDirection.NEUTRAL

    themes = news_bundle.get("key_themes") or []
    headlines = news_bundle.get("headlines") or []
    headline_ids = [h.get("news_id", "") for h in headlines[:8] if h.get("news_id")]
    tickers = list((market_evidence.get("tickers") or {}).keys())[:6]
    if not tickers:
        tickers = ["SPY"]

    narrative = news_bundle.get("llm_summary") or (
        f"News sentiment is {score:+.2f}. Market evidence suggests a {direction.value} bias over the near term."
    )
    risks = ["Macro uncertainty", "News sentiment may be lagging price action"]
    if abs(score) > 0.5:
        risks.append("Extreme sentiment — potential reversal risk")

    return MarketAnalysis(
        user_id=user_id,
        query_context=query,
        news_bundle=news_bundle,
        market_evidence=market_evidence,
        sentiment=SentimentView(
            overall_score=score,
            sector_sentiments={},
            key_themes=themes,
            supporting_headline_ids=headline_ids,
        ),
        outlook=MarketOutlook(
            direction=direction,
            horizon_days=10,
            confidence=0.55 if flow_type == FlowType.BASELINE else 0.65,
            affected_assets=tickers,
            narrative=narrative,
        ),
        risks=risks,
        data_sources=["news_bundle", "market_evidence", "heuristic"],
        flow_type=flow_type,
        parent_analysis_id=parent_analysis_id,
        metadata={"heuristic": True},
    )


def apply_feedback_heuristic(analysis: MarketAnalysis, correction_text: str) -> MarketAnalysis:
    """Lightweight revision without LLM."""
    text = correction_text.lower()
    direction = analysis.outlook.direction
    score = analysis.sentiment.overall_score

    if any(w in text for w in ("bear", "short", "down", "空", "跌")):
        direction = OutlookDirection.BEARISH
        score = min(score, -0.2)
    if any(w in text for w in ("bull", "long", "up", "涨", "多")):
        direction = OutlookDirection.BULLISH
        score = max(score, 0.2)
    if "neutral" in text or "观望" in text:
        direction = OutlookDirection.NEUTRAL

    revised = analysis.model_copy(deep=True)
    revised.flow_type = FlowType.REVISED
    revised.parent_analysis_id = analysis.analysis_id
    revised.analysis_id = str(__import__("uuid").uuid4())
    revised.outlook.direction = direction
    revised.sentiment.overall_score = score
    revised.outlook.narrative = f"{analysis.outlook.narrative}\n\n[User revision] {correction_text}"
    revised.outlook.confidence = min(0.95, analysis.outlook.confidence + 0.1)
    revised.metadata["revised_from"] = analysis.analysis_id
    revised.metadata["heuristic_revision"] = True
    return revised


def _extract_themes(headlines: list[dict]) -> list[str]:
    themes: list[str] = []
    for h in headlines:
        cat = h.get("category")
        if cat and cat not in themes:
            themes.append(str(cat))
    return themes[:8]
