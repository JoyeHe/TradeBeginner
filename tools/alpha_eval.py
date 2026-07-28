"""Safe boolean evaluation of alpha expressions against indicator maps."""

from __future__ import annotations

import ast
import operator
from typing import Any, Mapping, Optional

import structlog

logger = structlog.get_logger(__name__)

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
}
_CMPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ALLOWED_NAMES = {
    "close",
    "sma_20",
    "sma_50",
    "sma_200",
    "ema_12",
    "ema_26",
    "rsi_14",
    "momentum_roc_10",
    "momentum_roc_20",
    "adx",
    "atr_14",
    "vix",
    "macd",
    "macd_signal",
    "macd_histogram",
    "bollinger_upper",
    "bollinger_middle",
    "bollinger_lower",
    "true",
    "false",
}


def flatten_indicator_context(
    close: Optional[float],
    indicators: Mapping[str, Any] | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, float | bool]:
    """Build a flat numeric context for alpha expressions."""
    ctx: dict[str, float | bool] = {"true": True, "false": False}
    if close is not None:
        ctx["close"] = float(close)
    ind = dict(indicators or {})
    for key in (
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_12",
        "ema_26",
        "rsi_14",
        "momentum_roc_10",
        "momentum_roc_20",
        "adx",
        "atr_14",
    ):
        val = ind.get(key)
        if val is not None and not isinstance(val, dict):
            try:
                ctx[key] = float(val)
            except (TypeError, ValueError):
                pass
    macd = ind.get("macd")
    if isinstance(macd, dict):
        if macd.get("macd") is not None:
            ctx["macd"] = float(macd["macd"])
        if macd.get("signal") is not None:
            ctx["macd_signal"] = float(macd["signal"])
        if macd.get("histogram") is not None:
            ctx["macd_histogram"] = float(macd["histogram"])
    bb = ind.get("bollinger_bands")
    if isinstance(bb, dict):
        for src, dst in (("upper", "bollinger_upper"), ("middle", "bollinger_middle"), ("lower", "bollinger_lower")):
            if bb.get(src) is not None:
                ctx[dst] = float(bb[src])
    if extra:
        for k, v in extra.items():
            if v is None:
                continue
            try:
                ctx[k] = float(v) if not isinstance(v, bool) else v
            except (TypeError, ValueError):
                pass
    return ctx


def _eval_node(node: ast.AST, ctx: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ctx)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES and node.id not in ctx:
            raise ValueError(f"name not allowed: {node.id}")
        if node.id not in ctx:
            raise ValueError(f"missing input: {node.id}")
        return ctx[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_node(node.operand, ctx))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -float(_eval_node(node.operand, ctx))
    if isinstance(node, ast.BoolOp):
        vals = [_eval_node(v, ctx) for v in node.values]
        if isinstance(node.op, ast.And):
            return all(vals)
        if isinstance(node.op, ast.Or):
            return any(vals)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left, ctx), _eval_node(node.right, ctx))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, ctx)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_node(comparator, ctx)
            fn = _CMPS.get(type(op))
            if fn is None:
                raise ValueError("unsupported compare")
            if not fn(left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsupported expression node: {type(node).__name__}")


def eval_alpha(expression: str, ctx: Mapping[str, Any]) -> bool:
    """Evaluate a restricted boolean expression. Returns False on error/missing inputs."""
    try:
        tree = ast.parse(expression, mode="eval")
        return bool(_eval_node(tree, ctx))
    except Exception as exc:
        logger.debug("alpha_eval_failed", expression=expression, error=str(exc))
        return False
