#!/usr/bin/env python3
"""Frontend E2E: tab routing, UX states, and frontend↔backend API alignment."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from playwright.sync_api import Page, sync_playwright, expect

BASE = "http://127.0.0.1:8000"
TABS = ["news", "market", "analysis", "strategy", "system"]

# Frontend fetch() paths → backend routes (from api/static/index.html)
EXPECTED_API_MAP = {
    "POST /news/search": r"/news/search",
    "POST /market/search": r"/market/search",
    "POST /analysis/run-baseline": r"/analysis/run-baseline",
    "POST /analysis/.+/feedback": r"/analysis/[^/]+/feedback",
    "GET /analysis/.+/compare": r"/analysis/[^/]+/compare",
    "GET /library/strategies": r"/library/strategies",
    "GET /library/preferences": r"/library/preferences",
    "POST /strategy/user-request": r"/strategy/user-request",
    "POST /strategy/request-with-summary": r"/strategy/request-with-summary",
    "GET /strategy/.+": r"/strategy/[^/]+$",
    "POST /strategy/.+/approve": r"/strategy/[^/]+/approve",
    "POST /strategy/.+/reject": r"/strategy/[^/]+/reject",
    "POST /strategy/.+/refine": r"/strategy/[^/]+/refine",
    "GET /system/status": r"/system/status",
    "GET /portfolio": r"/portfolio",
    "GET /news/digest": r"/news/digest",
    "GET /history/episodes": r"/history/episodes",
    "GET /behavior/stats": r"/behavior/stats",
}


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str = ""
    severity: str = "error"  # error | warn | info


@dataclass
class E2EReport:
    steps: list[StepResult] = field(default_factory=list)
    api_calls: list[dict[str, Any]] = field(default_factory=list)
    console_errors: list[str] = field(default_factory=list)

    def add(self, name: str, ok: bool, detail: str = "", severity: str = "error") -> None:
        self.steps.append(StepResult(name, ok, detail, severity))


def _switch_tab(page: Page, tab: str) -> None:
    page.locator(f'nav.tabs button[data-tab="{tab}"]').click()
    expect(page.locator(f"#tab-{tab}")).to_have_class(re.compile(r"\bactive\b"))


def _wait_no_spinner(page: Page, spinner_id: str, timeout_ms: int = 120_000) -> None:
    page.locator(f"#{spinner_id}").wait_for(state="hidden", timeout=timeout_ms)


def run_frontend_e2e() -> E2EReport:
    report = E2EReport()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        def on_response(resp):
            url = resp.url
            if BASE not in url or "/docs" in url:
                return
            path = urlparse(url).path
            if path == "/":
                return
            report.api_calls.append({"method": resp.request.method, "path": path, "status": resp.status})

        page.on("response", on_response)
        page.on("console", lambda msg: report.console_errors.append(msg.text) if msg.type == "error" else None)

        page.goto(BASE, wait_until="networkidle", timeout=60_000)

        # ── Initial load UX ──
        title = page.locator("header h1").inner_text()
        report.add("Page title", title == "TradeBeginner", title)

        status_pill = page.locator("#statusPill").inner_text()
        report.add("Status pill on load", status_pill in {"Healthy", "Stale"}, status_pill)

        mode_pill = page.locator("#modePill").inner_text()
        report.add("Mode pill shows paper/live", mode_pill in {"Paper Trading", "Live Trading"}, mode_pill)

        status_box = page.locator("#statusBox").inner_text()
        report.add("System status auto-loaded", "news_fresh" in status_box, "JSON present" if "news_fresh" in status_box else status_box[:80])

        portfolio_box = page.locator("#portfolioBox").inner_text()
        report.add("Portfolio auto-loaded", "total_value" in portfolio_box, "JSON present" if "total_value" in portfolio_box else portfolio_box[:80])

        # ── Tab routing ──
        for tab in TABS:
            _switch_tab(page, tab)
            active_btn = page.locator("nav.tabs button.active").get_attribute("data-tab")
            panel_visible = page.locator(f"#tab-{tab}").is_visible()
            others_hidden = all(
                not page.locator(f"#tab-{other}").evaluate("el => el.classList.contains('active')")
                for other in TABS if other != tab
            )
            report.add(f"Tab route: {tab}", active_btn == tab and panel_visible and others_hidden, f"active={active_btn} visible={panel_visible}")

        # Default tab should be news
        _switch_tab(page, "news")
        report.add("Default tab is News", page.locator('nav.tabs button[data-tab="news"]').evaluate("el => el.classList.contains('active')"), "news")

        # No URL hash routing (UX note)
        page.locator('nav.tabs button[data-tab="market"]').click()
        hash_empty = page.evaluate("() => !window.location.hash || window.location.hash === ''")
        report.add("URL hash routing", False, "Tabs do not update location.hash — no deep links / back-button support", severity="warn")

        # ── News tab UX + API ──
        _switch_tab(page, "news")
        page.locator("#newsQuery").fill("Apple stock")
        page.locator("#btnNewsSearch").click()
        _wait_no_spinner(page, "newsSpinner", 90_000)
        summary = page.locator("#newsSummary").inner_text()
        has_summary = summary and summary != "Search results will appear here with an AI-generated interpretation." and summary != "Searching..."
        report.add("News search UX: AI summary", has_summary, summary[:120] if summary else "empty")

        kpi = page.locator("#newsKpi").inner_text()
        report.add("News search UX: KPI row", "Total:" in kpi, kpi[:100])

        tr_count = page.locator("#trCount").inner_text()
        report.add("News search UX: TrendRadar count pill", tr_count.isdigit(), f"count={tr_count}")

        # ── Market tab UX + API ──
        _switch_tab(page, "market")
        page.locator("#marketQuery").fill("AAPL")
        page.locator("#btnMarketSearch").click()
        _wait_no_spinner(page, "marketSpinner", 90_000)
        kpi_m = page.locator("#marketKpi").inner_text()
        report.add("Market search UX: tickers KPI", "AAPL" in kpi_m, kpi_m[:100])

        bars_html = page.locator("#marketBarsContainer").inner_html()
        report.add("Market search UX: OHLCV table", "bars-table" in bars_html and "Close" in bars_html, "table rendered" if "bars-table" in bars_html else "missing")

        indicators = page.locator("#marketIndicators").inner_text()
        report.add("Market search UX: indicators JSON", "rsi" in indicators.lower() or "close" in indicators.lower(), indicators[:80])

        # Date defaults
        d_from = page.locator("#mktDateFrom").input_value()
        d_to = page.locator("#mktDateTo").input_value()
        report.add("Market UX: date defaults set", bool(d_from) and bool(d_to), f"{d_from} → {d_to}")

        # ── Analysis tab (baseline + feedback + compare) ──
        _switch_tab(page, "analysis")
        page.locator("#analysisQuery").fill("AAPL outlook")
        page.locator("#analysisTickers").fill("AAPL")
        page.locator("#btnRunBaseline").click()
        _wait_no_spinner(page, "analysisSpinner", 180_000)

        aid = page.locator("#currentAnalysisId").inner_text()
        report.add("Analysis baseline: analysis_id set", aid != "None" and len(aid) > 8, aid)

        baseline_summary = page.locator("#baselineSummary").inner_text()
        report.add("Analysis baseline UX: summary text", len(baseline_summary) > 20 and "Run baseline" not in baseline_summary, baseline_summary[:100])

        baseline_json = page.locator("#baselineAnalysis").inner_text()
        report.add("Analysis baseline: JSON panel", "outlook" in baseline_json or "sentiment" in baseline_json, "schema fields present" if "outlook" in baseline_json else baseline_json[:80])

        page.locator("#analysisFeedbackText").fill("Near-term outlook should be more neutral.")
        page.locator("#feedbackVerdict").select_option("partial")
        page.locator("#feedbackDimension").select_option("market_outlook")
        page.locator("#btnSubmitFeedback").click()
        _wait_no_spinner(page, "feedbackSpinner", 180_000)

        revised_summary = page.locator("#revisedSummary").inner_text()
        report.add("Analysis feedback UX: revised summary", "Feedback failed" not in revised_summary and len(revised_summary) > 10, revised_summary[:100])

        page.locator("#btnCompareAnalysis").click()
        page.wait_for_timeout(1500)
        compare = page.locator("#compareResults").inner_text()
        report.add("Analysis compare: baseline vs revised", "baseline" in compare and "revised" in compare, compare[:80])

        page.locator("#btnRefreshLibrary").click()
        page.wait_for_timeout(1500)
        library = page.locator("#libraryResults").inner_text()
        report.add("Strategy library load", library != "Library not loaded." and "[" in library, library[:60])

        # ── Strategy tab ──
        _switch_tab(page, "strategy")
        page.evaluate("() => { document.getElementById('currentStrategy').textContent = 'None'; document.getElementById('strategySummary').textContent = ''; }")
        page.locator("#btnAutoStrategy").click()
        _wait_no_spinner(page, "stratSpinner", 180_000)
        sid = page.locator("#currentStrategy").inner_text()
        report.add("Strategy auto-generate: strategy_id", sid != "None", sid)

        strat_summary = page.locator("#strategySummary").inner_text()
        summary_ok = (
            strat_summary
            and "Auto-generating" not in strat_summary
            and "Generation failed" not in strat_summary
            and len(strat_summary) > 20
        )
        report.add("Strategy UX: summary shown", summary_ok, strat_summary[:100])

        page.locator("#btnGetStrategy").click()
        page.wait_for_timeout(2000)
        strat_json = page.locator("#strategyResults").inner_text()
        report.add("Strategy refresh: JSON panel", "strategy" in strat_json.lower() or "status" in strat_json.lower(), strat_json[:80])

        # ── System tab log ──
        _switch_tab(page, "system")
        log_box = page.locator("#logBox").inner_text()
        report.add("System UX: response log populated", "news_search" in log_box or "analysis_baseline" in log_box, "actions logged")

        page.locator("#btnRefreshStatus").click()
        page.wait_for_timeout(1000)
        report.add("System refresh button", "news_fresh" in page.locator("#statusBox").inner_text(), "status refreshed")

        # ── API alignment audit ──
        seen_paths = {(c["method"], c["path"]) for c in report.api_calls}
        failed_status = [c for c in report.api_calls if c["status"] >= 400]
        report.add("API calls: no 4xx/5xx", len(failed_status) == 0, str(failed_status[:5]) if failed_status else f"{len(report.api_calls)} calls OK")

        unmatched = []
        for method, path in seen_paths:
            key = f"{method} {path}"
            matched = any(re.search(pat, path) for pat in EXPECTED_API_MAP.values())
            if not matched:
                unmatched.append(key)
        report.add("API alignment: all calls map to backend", len(unmatched) == 0, str(unmatched[:5]) if unmatched else "aligned")

        # Frontend fields vs backend — news response keys used in UI
        news_calls = [c for c in report.api_calls if c["path"] == "/news/search" and c["status"] == 200]
        report.add("API alignment: /news/search called", len(news_calls) >= 1, f"calls={len(news_calls)}")

        browser.close()

    js_errors = [e for e in report.console_errors if "favicon" not in e.lower()]
    report.add("Browser console errors", len(js_errors) == 0, "; ".join(js_errors[:3]) if js_errors else "none", severity="warn")

    return report


def main() -> int:
    print("TradeBeginner Frontend E2E →", BASE)
    print("=" * 70)
    report = run_frontend_e2e()

    passed = failed = warned = 0
    for s in report.steps:
        if s.ok:
            mark, passed = "PASS", passed + 1
        elif s.severity == "warn":
            mark, warned = "WARN", warned + 1
        else:
            mark, failed = "FAIL", failed + 1
        print(f"[{mark}] {s.name}: {s.detail}")

    print("=" * 70)
    print(f"Result: {passed} passed, {failed} failed, {warned} warnings")
    print(f"API calls captured: {len(report.api_calls)}")
    if report.api_calls:
        uniq = sorted({f"{c['method']} {c['path']} → {c['status']}" for c in report.api_calls})
        print("Endpoints hit:")
        for line in uniq:
            print(f"  • {line}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
