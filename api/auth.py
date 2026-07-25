"""Optional API-key authentication for mutating endpoints."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from config.settings import Settings


def get_settings(request: Request) -> Settings:
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        return pipeline.settings
    return Settings()


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests when ATS_API_KEY is set and header does not match."""
    settings = get_settings(request)
    expected = (settings.api_key or "").strip()
    if not expected:
        return
    if (x_api_key or "").strip() != expected:
        raise HTTPException(status_code=401, detail={"error": "unauthorized", "detail": "Invalid or missing X-API-Key"})
