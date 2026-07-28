"""REST API for TradeBeginner pipeline."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from api.auth import require_api_key
from orchestrator.pipeline import TradingPipeline
from schemas.api_settings import RiskSettingsUpdate, WatchlistUpdate

app = FastAPI(title="TradeBeginner API", version="0.1.0")
_UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"


async def get_pipeline() -> TradingPipeline:
    pipeline = getattr(app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@app.get("/", response_class=HTMLResponse)
async def ui_home():
    if not _UI_PATH.exists():
        raise HTTPException(status_code=404, detail={"error": "ui_not_found", "detail": "UI assets missing"})
    return _UI_PATH.read_text(encoding="utf-8")


@app.post("/strategy/request")
async def request_strategy(
    payload: dict | None = None,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    try:
        return await pipeline.request_strategy()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "strategy_generation_failed", "detail": str(exc)}) from exc


@app.get("/strategy/{strategy_id}")
async def get_strategy(strategy_id: str, pipeline: TradingPipeline = Depends(get_pipeline)):
    return await pipeline.get_strategy_result(strategy_id)


@app.post("/strategy/{strategy_id}/approve")
async def approve_strategy(
    strategy_id: str,
    payload: dict | None = None,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    return await pipeline.user_approve(strategy_id, modifications=(payload or {}).get("modifications"))


@app.post("/strategy/{strategy_id}/reject")
async def reject_strategy(
    strategy_id: str,
    payload: dict | None = None,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    return await pipeline.user_reject(strategy_id, reason=(payload or {}).get("reason"))


@app.post("/strategy/{strategy_id}/refine")
async def refine_strategy(
    strategy_id: str,
    payload: dict,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    feedback = payload.get("feedback")
    if not feedback:
        raise HTTPException(status_code=400, detail={"error": "missing_feedback", "detail": "feedback is required"})
    return await pipeline.user_request_refinement(strategy_id, feedback=feedback)


@app.get("/portfolio")
async def get_portfolio(pipeline: TradingPipeline = Depends(get_pipeline)):
    return (await pipeline.get_portfolio()).model_dump(mode="json")


@app.get("/news/digest")
async def get_news_digest(pipeline: TradingPipeline = Depends(get_pipeline)):
    digest = await pipeline.memory.working.get("news_digest")
    return {} if digest is None else digest.model_dump(mode="json")


@app.get("/system/status")
async def get_system_status(pipeline: TradingPipeline = Depends(get_pipeline)):
    return await pipeline.get_system_status()


@app.get("/history/episodes")
async def get_episode_history(n: int = 20, pipeline: TradingPipeline = Depends(get_pipeline)):
    episodes = await pipeline.memory.episodic.retrieve_recent(n=n)
    return [x.model_dump(mode="json") for x in episodes]


@app.get("/behavior/stats")
async def get_behavior_stats(pipeline: TradingPipeline = Depends(get_pipeline)):
    return await pipeline.agent5.report()


@app.put("/settings/risk")
async def update_risk_settings(
    payload: RiskSettingsUpdate,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    updated = {}
    for key, value in payload.model_dump(exclude_none=True).items():
        setattr(pipeline.settings, key, value)
        updated[key] = value
    return {"status": "ok", "updated": updated}


@app.post("/news/search")
async def news_search(payload: dict, pipeline: TradingPipeline = Depends(get_pipeline)):
    query = (payload or {}).get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail={"error": "missing_query", "detail": "query is required"})
    try:
        return await pipeline.user_news_search(query)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "news_search_failed", "detail": str(exc)}) from exc


@app.post("/market/search")
async def market_search(payload: dict, pipeline: TradingPipeline = Depends(get_pipeline)):
    query = (payload or {}).get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail={"error": "missing_query", "detail": "query is required"})
    try:
        return await pipeline.user_market_search(
            query,
            start_date=(payload or {}).get("start_date"),
            end_date=(payload or {}).get("end_date"),
            interval=(payload or {}).get("interval", "1d"),
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "market_search_failed", "detail": str(exc)}) from exc


@app.post("/strategy/request-with-summary")
async def request_strategy_with_summary(
    payload: dict | None = None,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    try:
        return await pipeline.request_strategy_with_summary()
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "strategy_generation_failed", "detail": str(exc)}) from exc


@app.post("/strategy/user-request")
async def user_request_strategy(
    payload: dict,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    request_text = (payload or {}).get("request", "").strip()
    if not request_text:
        raise HTTPException(status_code=400, detail={"error": "missing_request", "detail": "request text is required"})
    try:
        return await pipeline.user_directed_strategy(request_text)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "strategy_request_failed", "detail": str(exc)}) from exc


@app.put("/settings/watchlist")
async def update_watchlist(
    payload: WatchlistUpdate,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    await pipeline.update_watchlist(payload.tickers)
    return {"status": "ok", "watchlist": payload.tickers}


@app.post("/analysis/run-baseline")
async def run_analysis_baseline(
    payload: dict,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    query = (payload or {}).get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail={"error": "missing_query", "detail": "query is required"})
    tickers = (payload or {}).get("tickers")
    user_id = (payload or {}).get("user_id", "default")
    try:
        return await pipeline.run_analysis_baseline(query, tickers=tickers, user_id=user_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "analysis_baseline_failed", "detail": str(exc)}) from exc


@app.get("/analysis/{analysis_id}")
async def get_analysis_session(analysis_id: str, pipeline: TradingPipeline = Depends(get_pipeline)):
    return await pipeline.get_analysis_session(analysis_id)


@app.post("/analysis/{analysis_id}/feedback")
async def submit_analysis_feedback(
    analysis_id: str,
    payload: dict,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    from schemas.analysis_feedback import AnalysisFeedback, DimensionFeedback

    try:
        dims = [DimensionFeedback.model_validate(d) for d in (payload or {}).get("dimension_feedbacks", [])]
        feedback = AnalysisFeedback(
            analysis_id=analysis_id,
            user_id=(payload or {}).get("user_id", "default"),
            overall_verdict=(payload or {}).get("overall_verdict", "partial"),
            dimension_feedbacks=dims,
            free_text=(payload or {}).get("free_text"),
        )
        return await pipeline.submit_analysis_feedback(analysis_id, feedback)
    except KeyError:
        raise HTTPException(status_code=404, detail={"error": "analysis_not_found", "detail": analysis_id}) from None
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "analysis_feedback_failed", "detail": str(exc)}) from exc


@app.get("/analysis/{analysis_id}/compare")
async def compare_analysis(analysis_id: str, pipeline: TradingPipeline = Depends(get_pipeline)):
    return await pipeline.compare_analysis_session(analysis_id)


@app.get("/library/strategies")
async def list_strategy_library(
    user_id: str = "default",
    limit: int = 50,
    tag: str | None = None,
    min_reward: float | None = None,
    pipeline: TradingPipeline = Depends(get_pipeline),
):
    return await pipeline.list_strategy_library(user_id=user_id, limit=limit, tag=tag, min_reward=min_reward)


@app.get("/library/strategies/search")
async def search_strategy_library(
    q: str,
    user_id: str = "default",
    limit: int = 20,
    pipeline: TradingPipeline = Depends(get_pipeline),
):
    return await pipeline.search_strategy_library(query=q, user_id=user_id, limit=limit)


@app.get("/library/strategies/{entry_id}")
async def get_strategy_library_entry(entry_id: str, pipeline: TradingPipeline = Depends(get_pipeline)):
    payload = await pipeline.get_strategy_library_entry(entry_id)
    if payload.get("status") == "not_found":
        raise HTTPException(status_code=404, detail={"error": "library_entry_not_found", "detail": entry_id})
    return payload


@app.post("/library/strategies")
async def create_strategy_library_entry(
    payload: dict,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    from schemas.strategy_library import StrategyLibraryEntry

    entry = StrategyLibraryEntry.model_validate(payload or {})
    saved = await pipeline.analysis_store.upsert_library_entry(entry)
    return saved.model_dump(mode="json")


@app.post("/library/strategies/promote")
async def promote_strategy_library_entry(
    payload: dict,
    pipeline: TradingPipeline = Depends(get_pipeline),
    _: None = Depends(require_api_key),
):
    force = bool((payload or {}).get("force"))
    entry = (payload or {}).get("entry") or payload
    return await pipeline.promote_strategy_to_library(entry, force=force)


@app.get("/library/preferences")
async def get_preference_stats(user_id: str = "default", pipeline: TradingPipeline = Depends(get_pipeline)):
    return await pipeline.get_preference_stats(user_id=user_id)
