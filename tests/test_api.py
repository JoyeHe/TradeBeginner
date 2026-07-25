from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient
import pytest

from api.server import app, get_pipeline
from schemas.execution import PortfolioState


class _PipeStub:
    def __init__(self):
        self.memory = type("M", (), {})()
        self.memory.working = type("W", (), {})()
        self.memory.working.get = self._get
        self.memory.episodic = type("E", (), {})()
        self.memory.episodic.retrieve_recent = self._episodes
        self.agent5 = type("A5", (), {})()
        self.agent5.report = self._report
        self._digest = {"overall_market_sentiment": 0.2}

    async def _get(self, key: str):
        if key == "news_digest":
            return type("D", (), {"model_dump": lambda self, mode="json": {"overall_market_sentiment": 0.2}})()
        return None

    async def _episodes(self, n=20):
        return [type("Ep", (), {"model_dump": lambda self, mode="json": {"trace_id": "e1"}})()]

    async def _report(self):
        return {"total_records": 1}

    async def request_strategy(self):
        return {"status": "generated", "strategy": {"strategy_id": "s1"}}

    async def get_strategy_result(self, strategy_id: str):
        if strategy_id == "s1":
            return {"status": "ready", "strategy": {"strategy_id": "s1"}, "reward": None}
        return {"status": "not_found"}

    async def user_approve(self, strategy_id: str, modifications=None):
        if strategy_id != "s1":
            return {"error": "strategy_not_found"}
        return {"risk_check": {"result": "pass"}, "execution": []}

    async def user_reject(self, strategy_id: str, reason=None):
        if strategy_id != "s1":
            return {"error": "strategy_not_found"}
        return {"status": "rejected"}

    async def user_request_refinement(self, strategy_id: str, feedback: str):
        if strategy_id != "s1":
            return {"error": "strategy_not_found"}
        return {"status": "generated", "strategy": {"strategy_id": "s2"}}

    async def get_portfolio(self):
        return PortfolioState(
            timestamp=datetime.utcnow(),
            cash=100_000,
            total_value=100_000,
            positions={},
            daily_pnl=0,
            total_pnl=0,
        )

    async def get_system_status(self):
        return {"news_fresh": True, "market_fresh": True, "open_orders": 0, "pending_strategies": 0}

    async def update_watchlist(self, tickers):
        return None


@pytest.fixture
def client():
    pipe = _PipeStub()
    app.dependency_overrides[get_pipeline] = lambda: pipe
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def test_api_generate_strategy(client: TestClient):
    """FUNCTION TESTED: api.server.request_strategy endpoint"""
    r = client.post("/strategy/request", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "generated"
    assert "strategy" in body


def test_api_approve_strategy(client: TestClient):
    """FUNCTION TESTED: api.server.approve_strategy endpoint"""
    r = client.post("/strategy/s1/approve", json={})
    assert r.status_code == 200
    assert "risk_check" in r.json()


def test_api_reject_strategy(client: TestClient):
    """FUNCTION TESTED: api.server.reject_strategy endpoint"""
    r = client.post("/strategy/s1/reject", json={"reason": "Too risky"})
    assert r.status_code == 200
    assert r.json()["status"] == "rejected"


def test_api_get_portfolio(client: TestClient):
    """FUNCTION TESTED: api.server.get_portfolio endpoint"""
    r = client.get("/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert "cash" in body and "total_value" in body


def test_api_get_news_digest(client: TestClient):
    """FUNCTION TESTED: api.server.get_news_digest endpoint"""
    r = client.get("/news/digest")
    assert r.status_code == 200
    assert "overall_market_sentiment" in r.json()


def test_api_nonexistent_strategy(client: TestClient):
    """FUNCTION TESTED: api.server.approve_strategy invalid strategy handling"""
    r = client.post("/strategy/invalid-uuid/approve", json={})
    assert r.status_code == 200
    assert r.json().get("error") == "strategy_not_found"


def test_api_error_format_consistency(client: TestClient):
    """FUNCTION TESTED: api.server.error format"""
    r = client.post("/strategy/s1/refine", json={})
    assert r.status_code == 400
    detail = r.json()["detail"]
    assert set(detail.keys()) == {"error", "detail"}


def test_api_system_status(client: TestClient):
    """FUNCTION TESTED: api.server.get_system_status endpoint"""
    r = client.get("/system/status")
    assert r.status_code == 200
    assert "news_fresh" in r.json()
