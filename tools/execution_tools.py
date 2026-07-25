"""Execution tools including paper trading engine and broker adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

import structlog

from config.settings import Settings
from schemas.execution import Order, OrderStatus, OrderType, PortfolioState
from schemas.strategy import PositionAction, Strategy

logger = structlog.get_logger(__name__)


class PaperTradingEngine:
    """Simulated order execution for safe default mode."""

    def __init__(self, initial_capital: float = 100000.0):
        self.cash = initial_capital
        self.positions: dict[str, dict] = {}
        self.orders: list[Order] = []
        self.trade_history: list[dict] = []

    async def submit_order(self, order: Order, current_price: float) -> Order:
        if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            return order
        if order.submitted_at is None:
            order.submitted_at = datetime.utcnow()
        order.status = OrderStatus.SUBMITTED
        fill = False
        if order.order_type == OrderType.MARKET:
            fill = True
        elif order.order_type == OrderType.LIMIT and order.limit_price is not None:
            fill = (order.side == "buy" and current_price <= order.limit_price) or (
                order.side == "sell" and current_price >= order.limit_price
            )
        if fill:
            order.status = OrderStatus.FILLED
            order.filled_price = current_price if order.order_type == OrderType.MARKET else order.limit_price
            order.filled_at = datetime.utcnow()
            cost = order.quantity * (order.filled_price or current_price)
            if order.side == "buy":
                if self.cash < cost:
                    order.status = OrderStatus.REJECTED
                    return order
                self.cash -= cost
                pos = self.positions.setdefault(order.asset, {"quantity": 0.0, "avg_cost": 0.0})
                old_qty = pos["quantity"]
                new_qty = old_qty + order.quantity
                pos["avg_cost"] = ((pos["avg_cost"] * old_qty) + cost) / max(new_qty, 1e-9)
                pos["quantity"] = new_qty
            else:
                self.cash += cost
                pos = self.positions.setdefault(order.asset, {"quantity": 0.0, "avg_cost": current_price})
                pos["quantity"] -= order.quantity
            self.trade_history.append(order.model_dump(mode="json"))
        if not any(o.order_id == order.order_id for o in self.orders):
            self.orders.append(order)
        return order

    async def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        market_value = 0.0
        for ticker, pos in self.positions.items():
            market_value += pos["quantity"] * current_prices.get(ticker, pos["avg_cost"])
        return self.cash + market_value

    async def get_portfolio_state(self, current_prices: dict[str, float]) -> PortfolioState:
        positions = {}
        total_pnl = 0.0
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos["avg_cost"])
            value = pos["quantity"] * price
            unreal = (price - pos["avg_cost"]) * pos["quantity"]
            total_pnl += unreal
            positions[ticker] = {"quantity": pos["quantity"], "avg_cost": pos["avg_cost"], "market_value": value, "unrealized_pnl": unreal}
        total = await self.get_portfolio_value(current_prices)
        return PortfolioState(timestamp=datetime.utcnow(), cash=self.cash, total_value=total, positions=positions, daily_pnl=0.0, total_pnl=total_pnl)


async def translate_strategy_to_orders(strategy: Strategy, portfolio_value: float, prices: dict[str, float]) -> list[Order]:
    """Translate strategy positions to normalized orders."""
    orders: list[Order] = []
    for pos in strategy.positions:
        price = prices.get(pos.asset)
        if price is None or price <= 0:
            continue
        side = "buy"
        if pos.action in (PositionAction.SHORT, PositionAction.CLOSE, PositionAction.REDUCE):
            side = "sell"
        quantity = max((portfolio_value * (pos.size_pct / 100.0)) / price, 0.0)
        order_type = OrderType.MARKET if pos.entry_price_target is None else OrderType.LIMIT
        orders.append(
            Order(
                order_id=str(uuid.uuid4()),
                strategy_id=strategy.strategy_id,
                asset=pos.asset,
                side=side,
                order_type=order_type,
                quantity=quantity,
                limit_price=pos.entry_price_target,
            )
        )
    return orders


async def submit_orders_paper(engine: PaperTradingEngine, orders: list[Order], prices: dict[str, float]) -> list[Order]:
    """Submit orders to paper engine."""
    return [await engine.submit_order(o, prices.get(o.asset, o.limit_price or 0.0)) for o in orders]


async def submit_orders_live(settings: Settings, orders: list[Order]) -> list[Order]:
    """Submit orders to Alpaca when live trading is enabled."""
    if settings.paper_trading:
        raise RuntimeError("Live submission blocked while paper_trading=True")
    try:
        from alpaca.trading.client import TradingClient  # type: ignore
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest
    except Exception as exc:
        raise RuntimeError(f"alpaca dependency unavailable: {exc}") from exc

    use_paper = settings.paper_trading or "paper" in (settings.broker_base_url or "").lower()
    client = TradingClient(settings.broker_api_key, settings.broker_api_secret, paper=use_paper)
    submitted: list[Order] = []
    for order in orders:
        try:
            req = (
                MarketOrderRequest(symbol=order.asset, qty=order.quantity, side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL, time_in_force=TimeInForce.DAY)
                if order.order_type == OrderType.MARKET
                else LimitOrderRequest(
                    symbol=order.asset,
                    qty=order.quantity,
                    side=OrderSide.BUY if order.side == "buy" else OrderSide.SELL,
                    limit_price=order.limit_price or 0.0,
                    time_in_force=TimeInForce.DAY,
                )
            )
            resp = client.submit_order(req)
            order.order_id = str(resp.id)
            order.status = OrderStatus.SUBMITTED
            order.submitted_at = datetime.utcnow()
        except Exception as exc:
            logger.error("live_order_failed", order_id=order.order_id, error=str(exc))
            order.status = OrderStatus.REJECTED
        submitted.append(order)
    return submitted


async def check_order_status_paper(engine: PaperTradingEngine, order: Order, current_price: float) -> Order:
    """Check and update paper order status."""
    if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
        return order
    return await engine.submit_order(order, current_price=current_price)


async def cancel_order_paper(order: Order) -> Order:
    """Cancel pending paper order."""
    if order.status in {OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED}:
        order.status = OrderStatus.CANCELLED
    return order


async def get_portfolio_state_paper(engine: PaperTradingEngine, prices: dict[str, float]) -> PortfolioState:
    """Read simulated paper portfolio."""
    return await engine.get_portfolio_state(prices)


