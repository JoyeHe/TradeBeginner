"""Agent 4: Trade execution (paper-first)."""

from __future__ import annotations

from datetime import date, timedelta

import structlog
from agno.agent import Agent

from agents.common import build_agno_model
from config.settings import Settings
from memory import MemoryManager
from schemas.execution import Order, OrderStatus, PortfolioState
from schemas.risk import RiskAssessment, RiskCheckResult
from schemas.strategy import Strategy
from tools.execution_tools import PaperTradingEngine, submit_orders_live, submit_orders_paper, translate_strategy_to_orders
from tools.market_tools import fetch_price_data

logger = structlog.get_logger(__name__)


class Agent4Executor:
    """Translates approved strategies into broker orders."""

    def __init__(self, settings: Settings, memory: MemoryManager):
        self.settings = settings
        self.memory = memory
        self.paper = PaperTradingEngine()
        self.open_orders: dict[str, Order] = {}
        self.agent = Agent(
            name="agent4_executor",
            model=build_agno_model(settings),
            instructions=["Execute strategies safely. Paper trading is default."],
            markdown=True,
        )

    async def _price_map(self, symbols: list[str]) -> dict[str, float]:
        bars = await fetch_price_data(symbols, start_date=date.today() - timedelta(days=7), end_date=date.today())
        return {s: b[-1].close for s, b in bars.items() if b}

    async def execute_strategy(self, strategy: Strategy, risk_assessment: RiskAssessment) -> list[Order]:
        """Run strategy execution workflow after risk checks."""
        if risk_assessment.result == RiskCheckResult.FAIL:
            raise RuntimeError("Risk assessment failed; execution blocked.")
        prices = await self._price_map([p.asset for p in strategy.positions])
        portfolio = await self.paper.get_portfolio_state(prices)
        orders = await translate_strategy_to_orders(strategy, portfolio.total_value, prices)
        submitted = (
            await submit_orders_paper(self.paper, orders, prices) if self.settings.paper_trading else await submit_orders_live(self.settings, orders)
        )
        for order in submitted:
            if order.status in {OrderStatus.SUBMITTED, OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}:
                self.open_orders[order.order_id] = order
        await self.memory.working.set("latest_orders", submitted, category="execution")
        await self.memory.working.set("portfolio_state", await self.paper.get_portfolio_state(prices), category="execution")
        return submitted

    async def monitor_open_orders(self) -> list[Order]:
        """Update status of pending open orders."""
        if not self.open_orders:
            return []
        prices = await self._price_map(sorted({x.asset for x in self.open_orders.values()}))
        updated = []
        for oid, order in list(self.open_orders.items()):
            if self.settings.paper_trading:
                order = await self.paper.submit_order(order, prices.get(order.asset, order.limit_price or 0.0))
            updated.append(order)
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                self.open_orders.pop(oid, None)
        return updated

    async def get_current_portfolio(self) -> PortfolioState:
        """Return current paper/live portfolio state."""
        prices = await self._price_map(sorted(self.paper.positions.keys()) or ["SPY"])
        return await self.paper.get_portfolio_state(prices)

