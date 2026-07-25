from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

import pytest


FUNC_RE = re.compile(r"FUNCTION TESTED:\s*([^\n]+)")
DIAG_RE = re.compile(r"DIAGNOSIS=([A-Z_]+)")


@dataclass
class Row:
    test: str
    function_tested: str
    module: str
    problem_location: str = ""
    technical_issue: str = ""
    diagnosis_category: str = ""


def _parse_function_tested(doc: str | None) -> str:
    if not doc:
        return "unknown"
    m = FUNC_RE.search(doc)
    return m.group(1).strip() if m else "unknown"


def _module_from_function(function_tested: str) -> str:
    if function_tested == "unknown":
        return "unknown"
    parts = function_tested.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else parts[0]


def _extract_issue(longrepr: str) -> tuple[str, str]:
    diagnosis = "UNKNOWN"
    d = DIAG_RE.search(longrepr)
    if d:
        diagnosis = d.group(1)
    issue = ""
    for line in longrepr.splitlines():
        line = line.strip()
        if line.startswith("AssertionError:"):
            issue = line.replace("AssertionError:", "").strip()
            break
    if not issue:
        issue = "test failed (no assertion message captured)"
    return issue, diagnosis


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def pytest_addoption(parser):
    parser.addoption(
        "--report",
        action="store_true",
        default=False,
        help="Print structured benchmark test summary report.",
    )


def pytest_configure(config):
    config._benchmark_report_data = {
        "function_map": {},
        "passed": [],
        "failed": [],
        "warnings": [],
    }


def pytest_collection_modifyitems(config, items):
    function_map = config._benchmark_report_data["function_map"]
    for item in items:
        doc = inspect.getdoc(getattr(item, "obj", None))
        function_map[item.nodeid] = _parse_function_tested(doc)


def pytest_warning_recorded(warning_message, when, nodeid, location):
    # Hook signature required by pytest. We keep this no-op placeholder so
    # warning capture can be expanded without changing plugin contract.
    _ = warning_message, when, nodeid, location


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when != "call":
        return

    data = item.config._benchmark_report_data
    function_tested = data["function_map"].get(item.nodeid, "unknown")
    module = _module_from_function(function_tested)

    if report.passed:
        data["passed"].append(Row(test=item.name, function_tested=function_tested, module=module))
    elif report.failed:
        longrepr = str(report.longrepr)
        issue, diagnosis = _extract_issue(longrepr)
        data["failed"].append(
            Row(
                test=item.name,
                function_tested=function_tested,
                module=module,
                problem_location=f"{report.location[0]}:{report.location[1] + 1}",
                technical_issue=issue,
                diagnosis_category=diagnosis,
            )
        )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not config.getoption("--report"):
        return

    data = config._benchmark_report_data
    terminalreporter.write_sep("=", "Test Summary Report")

    passed: list[Row] = data["passed"]
    failed: list[Row] = data["failed"]
    warnings: list[Row] = data["warnings"]

    terminalreporter.write_line(f"\n### PASSED ({len(passed)} tests)")
    if passed:
        rows = [[r.test, r.function_tested, r.module] for r in passed]
        for line in _md_table(["Test", "Function Tested", "Module"], rows):
            terminalreporter.write_line(line)
    else:
        terminalreporter.write_line("(none)")

    terminalreporter.write_line(f"\n### FAILED ({len(failed)} tests)")
    if failed:
        rows = [
            [r.test, r.function_tested, r.problem_location, r.technical_issue, r.diagnosis_category]
            for r in failed
        ]
        for line in _md_table(
            ["Test", "Function Tested", "Problem Location", "Technical Issue", "Diagnosis Category"],
            rows,
        ):
            terminalreporter.write_line(line)
    else:
        terminalreporter.write_line("(none)")

    terminalreporter.write_line(f"\n### WARNINGS ({len(warnings)} tests)")
    if warnings:
        rows = [[r.test, r.technical_issue] for r in warnings]
        for line in _md_table(["Test", "Concern"], rows):
            terminalreporter.write_line(line)
    else:
        terminalreporter.write_line("(none)")

    _ = exitstatus
