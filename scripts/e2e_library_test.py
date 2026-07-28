#!/usr/bin/env python3
"""E2E validation for strategy library + weighted backtest + promote path."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from typing import Any

BASE = "http://127.0.0.1:8000"


def _request(method: str, path: str, body: dict | None = None, timeout: float = 180.0) -> tuple[int, Any]:
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
    except Exception as exc:  # noqa: BLE001
        return 0, {"detail": str(exc)}


def main() -> int:
    failures: list[str] = []

    code, status = _request("GET", "/system/status", timeout=30)
    if code != 200:
        print("FAIL system status", code, status)
        return 1
    print("OK system status")

    code, lib = _request("GET", "/library/strategies?limit=100", timeout=30)
    if code != 200 or not isinstance(lib, list) or len(lib) < 20:
        failures.append(f"library seed count expected >=20 got {None if not isinstance(lib, list) else len(lib)}")
    else:
        seeds = [e for e in lib if e.get("user_id") == "*" or e.get("origin") == "seed"]
        print(f"OK library entries={len(lib)} seeds≈{len(seeds)}")
        sample = lib[0]
        print("  sample:", sample.get("name"), sample.get("template_id"), "alphas", len(sample.get("alphas") or []))

    code, searched = _request("GET", "/library/strategies/search?q=momentum+roc", timeout=30)
    if code != 200 or not isinstance(searched, list) or len(searched) < 1:
        failures.append(f"library search failed: {code} {searched}")
    else:
        print(f"OK library search hits={len(searched)}")

    code, baseline = _request(
        "POST",
        "/analysis/run-baseline",
        {"query": "QQQ swing using SMA20 and ROC20 from library templates", "tickers": ["QQQ"], "user_id": "e2e_library"},
        timeout=240,
    )
    if code != 200 or not isinstance(baseline, dict):
        failures.append(f"baseline failed: {code} {baseline}")
        print("FAILURES:", failures)
        return 1

    strat = baseline.get("baseline_strategy") or {}
    reward = baseline.get("baseline_reward") or {}
    bt = (reward.get("backtest_result") or {})
    meta = strat.get("metadata") or {}
    print(
        "OK baseline",
        "analysis_id=",
        baseline.get("analysis_id"),
        "fallback=",
        meta.get("fallback"),
        "template_id=",
        meta.get("template_id"),
        "trades=",
        bt.get("total_trades"),
        "return=",
        bt.get("total_return"),
        "reward=",
        reward.get("terminal_reward"),
        "dd=",
        bt.get("max_drawdown"),
    )
    if meta.get("fallback") is True:
        failures.append("baseline strategy fell back")
    if bt.get("total_trades", 0) >= 1 and (bt.get("total_return") or 0) < 0 and (bt.get("max_drawdown") or 0) <= 0:
        failures.append("max_drawdown still zero on losing trade")
    # size weighting sanity: positions sizes should be constrained
    for p in strat.get("positions") or []:
        if float(p.get("size_pct") or 0) > 15.01:
            failures.append(f"size over cap: {p}")

    # feedback path
    aid = baseline.get("analysis_id")
    code, fb = _request(
        "POST",
        f"/analysis/{aid}/feedback",
        {
            "user_id": "e2e_library",
            "overall_verdict": "partial",
            "free_text": "Please keep QQQ but reduce size to 10% and use SMA/ROC template discipline.",
        },
        timeout=240,
    )
    if code != 200:
        failures.append(f"feedback failed: {code} {fb}")
    else:
        print(
            "OK feedback material_change=",
            fb.get("material_change"),
            "reward=",
            (fb.get("feedback_reward") or {}).get("terminal_reward"),
        )

    code, prefs = _request("GET", "/library/preferences?user_id=e2e_library", timeout=30)
    if code == 200:
        print("OK prefs", {k: prefs.get(k) for k in ("library_entries", "seed_library_count", "total_preference_records")})
    else:
        failures.append(f"prefs failed {code}")

    if failures:
        print("E2E FAILURES:")
        for f in failures:
            print(" -", f)
        return 1
    print("E2E PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
