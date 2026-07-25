# TradeBeginner

TradeBeginner is a multi-agent trading system scaffold that combines:

- **Agno** for agent orchestration.
- **OpenBB** for market data retrieval.
- **TrendRadar-compatible grammar handling** for news/search interpretation.
- **SQLAlchemy async + PostgreSQL** for persistent episodic storage.
- **ChromaDB** for semantic memory retrieval.
- **Agent Lightning-ready trace logging** for future RL integration.

## Quick Start

```bash
cd TradeBeginner
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
cp .env.example .env
python main.py
```

API default: `http://0.0.0.0:8000`

## Required API Keys

Minimum to run core strategy generation:

- `ATS_LLM_API_KEY` (DeepSeek key)

Needed for full feature coverage:

- `ATS_NEWS_API_KEY` (NewsAPI)
- `ATS_FINNHUB_API_KEY` (Finnhub)
- `ATS_OPENBB_PAT` (OpenBB PAT, provider-dependent endpoints)

Needed only for live broker execution (`ATS_PAPER_TRADING=false`):

- `ATS_BROKER_API_KEY`
- `ATS_BROKER_API_SECRET`
