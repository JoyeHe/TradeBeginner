"""Alpha formula schema for strategy library templates."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class AlphaFormula(BaseModel):
    """Declarative boolean alpha evaluated against indicators + last close."""

    formula_id: str
    name: str
    expression: str = Field(
        ...,
        description='e.g. "(close > sma_20) and (momentum_roc_20 > 0) and (rsi_14 < 70)"',
    )
    inputs: list[str] = Field(default_factory=list)
    direction: Literal["long", "short", "flat"] = "long"
    lookback_hint_days: int = 60
    notes: str = ""
