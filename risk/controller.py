"""Deterministic risk gatekeeper."""

from __future__ import annotations

from typing import Optional

from config.settings import Settings
from schemas.execution import PortfolioState
from schemas.risk import RiskAssessment, RiskCheckResult, RiskViolation
from schemas.strategy import PositionAction, Strategy


SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer",
    "TSLA": "Consumer",
    "JPM": "Financial",
    "XOM": "Energy",
    "SPY": "Index",
    "QQQ": "Index",
}


class RiskController:
    """Rule-based strategy risk validator."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._sector_cache: dict[str, str] = {}

    def check_position_size(self, strategy: Strategy) -> list[RiskViolation]:
        out = []
        for pos in strategy.positions:
            if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                continue
            if pos.size_pct > self.settings.max_position_size_pct:
                out.append(
                    RiskViolation(
                        rule_name="max_position_size",
                        description=f"{pos.asset} exceeds single-position limit",
                        severity="hard",
                        current_value=pos.size_pct,
                        limit_value=self.settings.max_position_size_pct,
                    )
                )
        return out

    def check_total_exposure(self, strategy: Strategy) -> Optional[RiskViolation]:
        exposure = sum(pos.size_pct for pos in strategy.positions if pos.action not in (PositionAction.CLOSE, PositionAction.REDUCE))
        if exposure > self.settings.max_total_exposure_pct:
            return RiskViolation(
                rule_name="max_total_exposure",
                description="Total exposure exceeds configured max",
                severity="hard",
                current_value=exposure,
                limit_value=self.settings.max_total_exposure_pct,
            )
        return None

    def check_stop_losses(self, strategy: Strategy) -> list[RiskViolation]:
        out = []
        for pos in strategy.positions:
            if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                continue
            if pos.stop_loss_pct <= 0:
                out.append(
                    RiskViolation(
                        rule_name="mandatory_stop_loss",
                        description=f"{pos.asset} missing stop loss",
                        severity="hard",
                        current_value=0.0,
                        limit_value=0.01,
                    )
                )
            if pos.stop_loss_pct > self.settings.max_stop_loss_pct:
                out.append(
                    RiskViolation(
                        rule_name="max_stop_loss_pct",
                        description=f"{pos.asset} stop loss too wide",
                        severity="hard",
                        current_value=pos.stop_loss_pct,
                        limit_value=self.settings.max_stop_loss_pct,
                    )
                )
        return out

    def check_concurrent_positions(self, strategy: Strategy) -> Optional[RiskViolation]:
        count = sum(1 for pos in strategy.positions if pos.action not in (PositionAction.CLOSE, PositionAction.REDUCE))
        if count > self.settings.max_concurrent_positions:
            return RiskViolation(
                rule_name="max_concurrent_positions",
                description="Too many concurrent positions",
                severity="hard",
                current_value=float(count),
                limit_value=float(self.settings.max_concurrent_positions),
            )
        return None

    def _sector(self, ticker: str) -> str:
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]
        sector = SECTOR_MAP.get(ticker, "Unknown")
        self._sector_cache[ticker] = sector
        return sector

    def check_sector_concentration(self, strategy: Strategy) -> list[RiskViolation]:
        agg: dict[str, float] = {}
        for pos in strategy.positions:
            if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                continue
            sector = self._sector(pos.asset)
            agg[sector] = agg.get(sector, 0.0) + pos.size_pct
        out = []
        for sector, exposure in agg.items():
            if exposure > self.settings.max_sector_concentration_pct:
                out.append(
                    RiskViolation(
                        rule_name="max_sector_concentration",
                        description=f"{sector} exposure too high",
                        severity="soft",
                        current_value=exposure,
                        limit_value=self.settings.max_sector_concentration_pct,
                    )
                )
        return out

    def check_correlation(self, strategy: Strategy) -> list[RiskViolation]:
        """SOFT RULE: warn if >3 positions in the same sector."""
        sectors: dict[str, int] = {}
        for pos in strategy.positions:
            if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                continue
            sector = self._sector(pos.asset)
            sectors[sector] = sectors.get(sector, 0) + 1
        out = []
        for sector, count in sectors.items():
            if count > 3:
                out.append(
                    RiskViolation(
                        rule_name="correlation_warning",
                        description=f">3 positions in {sector} sector",
                        severity="soft",
                        current_value=float(count),
                        limit_value=3.0,
                    )
                )
        return out

    def check_drawdown_limit(self, portfolio: PortfolioState) -> Optional[RiskViolation]:
        if portfolio.total_value <= 0:
            return None
        approx_drawdown = max(0.0, min(100.0, -portfolio.total_pnl / max(portfolio.total_value, 1e-9) * 100))
        if approx_drawdown > self.settings.max_portfolio_drawdown_pct:
            return RiskViolation(
                rule_name="max_portfolio_drawdown_pct",
                description="Portfolio drawdown circuit breaker triggered",
                severity="hard",
                current_value=approx_drawdown,
                limit_value=self.settings.max_portfolio_drawdown_pct,
            )
        return None

    async def auto_adjust(self, strategy: Strategy, violations: list[RiskViolation]) -> Strategy:
        """Auto-fix soft violations by scaling positions and tightening stops."""
        adjusted = strategy.model_copy(deep=True)
        if any(v.rule_name == "max_position_size" for v in violations):
            for pos in adjusted.positions:
                pos.size_pct = min(pos.size_pct, self.settings.max_position_size_pct)
        if any(v.rule_name == "max_stop_loss_pct" for v in violations):
            for pos in adjusted.positions:
                pos.stop_loss_pct = min(pos.stop_loss_pct, self.settings.max_stop_loss_pct)
        if any(v.rule_name == "max_sector_concentration" for v in violations):
            sector_alloc: dict[str, float] = {}
            for pos in adjusted.positions:
                if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                    continue
                sector = self._sector(pos.asset)
                sector_alloc[sector] = sector_alloc.get(sector, 0.0) + pos.size_pct
            for pos in adjusted.positions:
                sector = self._sector(pos.asset)
                total = sector_alloc.get(sector, 0.0)
                if total > self.settings.max_sector_concentration_pct and total > 0:
                    scale = self.settings.max_sector_concentration_pct / total
                    pos.size_pct = round(pos.size_pct * scale, 2)
        return adjusted

    def check_min_avg_volume(self, strategy: Strategy, asset_volumes: dict[str, int] | None = None) -> list[RiskViolation]:
        """Reject new long exposure when average volume is below configured minimum."""
        if not asset_volumes:
            return []
        out: list[RiskViolation] = []
        for pos in strategy.positions:
            if pos.action in (PositionAction.CLOSE, PositionAction.REDUCE):
                continue
            volume = asset_volumes.get(pos.asset)
            if volume is not None and volume < self.settings.min_avg_volume:
                out.append(
                    RiskViolation(
                        rule_name="min_avg_volume",
                        description=f"{pos.asset} average volume below minimum",
                        severity="hard",
                        current_value=float(volume),
                        limit_value=float(self.settings.min_avg_volume),
                    )
                )
        return out

    async def assess(
        self,
        strategy: Strategy,
        portfolio: PortfolioState,
        asset_volumes: dict[str, int] | None = None,
    ) -> RiskAssessment:
        violations = []
        violations.extend(self.check_position_size(strategy))
        total = self.check_total_exposure(strategy)
        if total:
            violations.append(total)
        violations.extend(self.check_stop_losses(strategy))
        conc = self.check_concurrent_positions(strategy)
        if conc:
            violations.append(conc)
        violations.extend(self.check_sector_concentration(strategy))
        violations.extend(self.check_correlation(strategy))
        violations.extend(self.check_min_avg_volume(strategy, asset_volumes=asset_volumes))
        draw = self.check_drawdown_limit(portfolio)
        if draw:
            violations.append(draw)

        hard = [v for v in violations if v.severity == "hard"]
        if hard:
            return RiskAssessment(result=RiskCheckResult.FAIL, violations=violations, summary="Execution blocked by hard risk rules.")
        if violations:
            adjusted = await self.auto_adjust(strategy, violations)
            return RiskAssessment(result=RiskCheckResult.WARN, violations=violations, adjusted_strategy=adjusted, warnings=[v.description for v in violations], summary="Warnings found; execution allowed with caution.")
        return RiskAssessment(result=RiskCheckResult.PASS, summary="All risk checks passed.")

