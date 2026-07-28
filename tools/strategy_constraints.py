"""Strategy constraint validation and repair."""

from __future__ import annotations

from typing import Any, Optional

import structlog

from config.settings import Settings
from schemas.strategy import Position, PositionAction, Strategy

logger = structlog.get_logger(__name__)

MEGA_CAP = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"}
CRYPTO_SUFFIXES = ("-USD",)
CRYPTO_TICKERS = {"BTC", "ETH", "BTC-USD", "ETH-USD"}


def _is_crypto(asset: str) -> bool:
    a = asset.upper()
    return a in CRYPTO_TICKERS or a.endswith("-USD")


def apply_strategy_constraints(
    strategy: Strategy,
    settings: Settings,
    requested_tickers: Optional[list[str]] = None,
) -> tuple[Strategy, list[str]]:
    """Validate/repair size, exposure, crypto caps; note missing requested tickers.

    Returns (possibly repaired strategy, list of violation/repair notes).
    """
    notes: list[str] = []
    repaired = strategy.model_copy(deep=True)
    max_single = float(getattr(settings, "max_position_size_pct", 15.0))
    max_gross = float(getattr(settings, "max_total_exposure_pct", 80.0))
    max_crypto = float(getattr(settings, "max_crypto_pct", 3.0))

    active: list[Position] = []
    for pos in repaired.positions:
        if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
            active.append(pos)
            continue
        p = pos.model_copy(deep=True)
        if p.size_pct > max_single:
            notes.append(f"scaled {p.asset} size {p.size_pct:.1f}% → {max_single:.1f}%")
            p.size_pct = max_single
        if _is_crypto(p.asset) and p.size_pct > max_crypto:
            notes.append(f"scaled crypto {p.asset} size → {max_crypto:.1f}%")
            p.size_pct = max_crypto
        if p.action == PositionAction.SHORT and p.asset.upper() in MEGA_CAP:
            notes.append(f"mega-cap naked short warning: {p.asset}")
        active.append(p)
    repaired.positions = active

    gross = sum(
        p.size_pct
        for p in repaired.positions
        if p.action not in (PositionAction.CLOSE, PositionAction.REDUCE)
    )
    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        notes.append(f"scaled gross exposure {gross:.1f}% → {max_gross:.1f}%")
        new_positions = []
        for p in repaired.positions:
            if p.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                new_positions.append(p)
            else:
                new_positions.append(p.model_copy(update={"size_pct": round(p.size_pct * scale, 4)}))
        repaired.positions = new_positions
        gross = sum(
            p.size_pct
            for p in repaired.positions
            if p.action not in (PositionAction.CLOSE, PositionAction.REDUCE)
        )

    repaired.risk_metrics.total_exposure_pct = gross

    if requested_tickers:
        have = {p.asset.upper() for p in repaired.positions}
        missing = [t for t in requested_tickers if t.upper() not in have]
        if missing:
            notes.append(f"missing_requested_tickers: {missing}")
            meta = dict(repaired.metadata or {})
            meta["skipped_tickers"] = missing
            repaired.metadata = meta

    if notes:
        meta = dict(repaired.metadata or {})
        meta["constraint_notes"] = notes
        repaired.metadata = meta
        logger.info("strategy_constraints_applied", notes=notes, strategy_id=repaired.strategy_id)
    return repaired, notes


def strategies_materially_different(a: Strategy | None, b: Strategy | None, size_eps: float = 1.0) -> bool:
    """True if tickers/actions/sizes/stops/horizons differ meaningfully."""
    if a is None or b is None:
        return a is not b

    def keyset(s: Strategy) -> set[tuple]:
        out = set()
        for p in s.positions:
            out.add(
                (
                    p.asset.upper(),
                    p.action.value,
                    round(p.size_pct / size_eps) * size_eps,
                    round(p.stop_loss_pct, 2),
                    p.time_horizon_days,
                )
            )
        return out

    return keyset(a) != keyset(b)
