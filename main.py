"""Application entrypoint."""

from __future__ import annotations

import asyncio

import structlog
import uvicorn

from api.server import app
from config.settings import Settings
from orchestrator.pipeline import TradingPipeline

logger = structlog.get_logger(__name__)


async def main() -> None:
    """Initialize pipeline and serve API."""
    settings = Settings()
    pipeline = TradingPipeline(settings)
    await pipeline.initialize()
    await pipeline.start_background_tasks()
    app.state.pipeline = pipeline
    config = uvicorn.Config(app, host=settings.api_host, port=settings.api_port)
    server = uvicorn.Server(config)
    try:
        await server.serve()
    finally:
        await pipeline.shutdown()
        logger.info("pipeline_shutdown_complete")


if __name__ == "__main__":
    asyncio.run(main())

