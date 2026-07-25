"""Risk assessment schemas."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from schemas.strategy import Strategy


class RiskCheckResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"


class RiskViolation(BaseModel):
    rule_name: str
    description: str
    severity: str
    current_value: float
    limit_value: float


class RiskAssessment(BaseModel):
    result: RiskCheckResult
    violations: list[RiskViolation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    adjusted_strategy: Optional[Strategy] = None
    summary: str

