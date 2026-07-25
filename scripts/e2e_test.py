#!/usr/bin/env python3
"""End-to-end smoke test against a running TradeBeginner API (http://127.0.0.1:8000)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

BASE = "http://127.0.0.1:8000"
POLL_INTERVAL = 2.0
STRATEGY_TIMEOUT = 120.0
ANALYSIS_TIMEOUT = 180.0


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def _request(method: str, path: str, body: dict | None = None, timeout: float = 60.0) -> tuple[int, Any]:
    url = f"{BASE}{path}"
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        method=method,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            return resp.status, json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            parsed = json.loads(raw) if raw else {"detail": exc.reason}
        except json.JSONDecodeError:
            parsed = {"detail": raw.decode(errors="replace")}
        return exc.code, parsed


def _poll_strategy(strategy_id: str, timeout: float) -> tuple[bool, dict]:
    deadline = time.time() + timeout
    last: dict = {}
    while time.time() < deadline:
        code, body = _request("GET", f"/strategy/{strategy_id}", timeout=30)
        last = body if isinstance(body, dict) else {"raw": body}
        if code != 200:
            time.sleep(POLL_INTERVAL)
            continue
        status = last.get("status")
        if status in {"ready", "evaluating"} and last.get("reward") is not None:
            return True, last
        if status == "ready" and last.get("strategy"):
            return True, last
        time.sleep(POLL_INTERVAL)
    return False, last


def run_e2e() -> list[StepResult]:
    results: list[StepResult] = []

    code, status = _request("GET", "/system/status")
    ok = code == 200 and status.get("episodic_db_connected") is True
    results.append(
        StepResult(
            "1. System status",
            ok,
            f"HTTP {code} news_fresh={status.get('news_fresh')} market_fresh={status.get('market_fresh')} db={status.get('episodic_db_connected')}",
            status if isinstance(status, dict) else {},
        )
    )

    code, digest = _request("GET", "/news/digest")
    ok = code == 200 and bool(digest)
    results.append(
        StepResult(
            "2. News digest",
            ok,
            f"HTTP {code} sentiment={digest.get('overall_market_sentiment') if isinstance(digest, dict) else None}",
            digest if isinstance(digest, dict) else {},
        )
    )

    code, market = _request("POST", "/market/search", {"query": "AAPL price last 30 days", "interval": "1d"}, timeout=90)
    tickers = market.get("tickers", []) if isinstance(market, dict) else []
    bars = 0
    if isinstance(market, dict) and market.get("data"):
        for t in market["data"].values():
            bars = max(bars, t.get("bars_count", 0))
    ok = code == 200 and bars > 0
    results.append(StepResult("3. Market search (OpenBB/CBOE)", ok, f"HTTP {code} tickers={tickers} max_bars={bars}"))

    code, news = _request("POST", "/news/search", {"query": "Apple earnings"}, timeout=90)
    items = news.get("item_count", 0) if isinstance(news, dict) else 0
    results.append(StepResult("4. News search", code == 200 and items >= 0, f"HTTP {code} items={items}"))

    code, analysis = _request(
        "POST",
        "/analysis/run-baseline",
        {"query": "AAPL tech outlook", "tickers": ["AAPL"], "user_id": "e2e"},
        timeout=ANALYSIS_TIMEOUT,
    )
    analysis_id = analysis.get("analysis_id") if isinstance(analysis, dict) else None
    baseline_sid = None
    if isinstance(analysis, dict) and analysis.get("baseline_strategy"):
        baseline_sid = analysis["baseline_strategy"].get("strategy_id")
    ok = code == 200 and analysis_id and baseline_sid
    results.append(
        StepResult(
            "5. Analysis baseline (LLM + backtest)",
            ok,
            f"HTTP {code} analysis_id={analysis_id} strategy_id={baseline_sid}",
            {"analysis_id": analysis_id},
        )
    )

    if analysis_id:
        code, session = _request("GET", f"/analysis/{analysis_id}")
        ok = code == 200 and session.get("analysis_id") == analysis_id
        results.append(StepResult("6. Get analysis session", ok, f"HTTP {code} flow={session.get('flow_type') if isinstance(session, dict) else None}"))

        code, fb = _request(
            "POST",
            f"/analysis/{analysis_id}/feedback",
            {
                "user_id": "e2e",
                "overall_verdict": "partial",
                "dimension_feedbacks": [
                    {"dimension": "market_outlook", "verdict": "disagree", "correction": "More neutral near-term"},
                ],
                "free_text": "E2E feedback smoke test",
            },
            timeout=ANALYSIS_TIMEOUT,
        )
        revised_sid = None
        if isinstance(fb, dict) and fb.get("feedback_strategy"):
            revised_sid = fb["feedback_strategy"].get("strategy_id")
        results.append(StepResult("7. Analysis feedback → revised strategy", code == 200 and bool(revised_sid), f"HTTP {code} feedback_strategy={revised_sid}"))

        code, compare = _request("GET", f"/analysis/{analysis_id}/compare")
        results.append(StepResult("8. Analysis compare", code == 200, f"HTTP {code} keys={list(compare.keys()) if isinstance(compare, dict) else []}"))

    code, lib = _request("GET", "/library/strategies?user_id=e2e&limit=5")
    lib_count = len(lib) if isinstance(lib, list) else 0
    results.append(StepResult("9. Strategy library", code == 200, f"HTTP {code} entries={lib_count}"))

    code, gen = _request("POST", "/strategy/request", {}, timeout=ANALYSIS_TIMEOUT)
    sid = gen.get("strategy", {}).get("strategy_id") if isinstance(gen, dict) else None
    results.append(StepResult("10. Strategy request (auto)", code == 200 and bool(sid), f"HTTP {code} strategy_id={sid}"))

    if sid:
        ready, strat = _poll_strategy(sid, STRATEGY_TIMEOUT)
        reward = strat.get("reward") if isinstance(strat, dict) else None
        results.append(
            StepResult(
                "11. Strategy evaluation poll",
                ready,
                f"status={strat.get('status') if isinstance(strat, dict) else None} reward={'yes' if reward else 'no'}",
            )
        )

        code, approved = _request("POST", f"/strategy/{sid}/approve", {})
        exec_ok = code == 200 and approved.get("execution") is not None
        risk = approved.get("risk_check", {}) if isinstance(approved, dict) else {}
        results.append(
            StepResult(
                "12. Strategy approve (paper)",
                exec_ok or (code == 200 and risk.get("result") == "block"),
                f"HTTP {code} risk={risk.get('result')} executed={approved.get('execution') is not None if isinstance(approved, dict) else False}",
            )
        )

    code, portfolio = _request("GET", "/portfolio")
    ok = code == 200 and "total_value" in (portfolio if isinstance(portfolio, dict) else {})
    results.append(StepResult("13. Portfolio", ok, f"HTTP {code} total={portfolio.get('total_value') if isinstance(portfolio, dict) else None}"))

    code, episodes = _request("GET", "/history/episodes?n=5")
    ep_count = len(episodes) if isinstance(episodes, list) else 0
    results.append(StepResult("14. Episode history", code == 200, f"HTTP {code} episodes={ep_count}"))

    code, behavior = _request("GET", "/behavior/stats")
    results.append(
        StepResult(
            "15. Behavior stats",
            code == 200,
            f"HTTP {code} records={behavior.get('total_records') if isinstance(behavior, dict) else None}",
        )
    )

    return results


def main() -> int:
    print("TradeBeginner E2E test →", BASE)
    print("=" * 60)
    results = run_e2e()
    passed = sum(1 for r in results if r.ok)
    for r in results:
        mark = "PASS" if r.ok else "FAIL"
        print(f"[{mark}] {r.name}: {r.detail}")
    print("=" * 60)
    print(f"Result: {passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
