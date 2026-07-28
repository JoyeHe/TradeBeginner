from __future__ import annotations

import math
import time
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from agents.agent2_market import Agent2Market
from config.settings import Settings
from memory.working import WorkingMemory
from schemas.market_data import MarketSnapshot, OHLCVBar, TechnicalIndicators
from schemas.strategy import MarketRegime
from tools import market_tools


def _diag(function_tested: str, input_value, expected, actual, diagnosis: str) -> str:
    return (
        f"FUNCTION TESTED: {function_tested} | INPUT={input_value!r} | "
        f"EXPECTED={expected!r} | ACTUAL={actual!r} | DIAGNOSIS={diagnosis}"
    )


def _bars_from_df(df: pd.DataFrame, ticker: str) -> list[OHLCVBar]:
    bars: list[OHLCVBar] = []
    for idx, row in df.iterrows():
        bars.append(
            OHLCVBar(
                timestamp=pd.Timestamp(idx).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row["volume"]),
                asset=ticker,
            )
        )
    return bars


@pytest.fixture(autouse=True)
def clear_market_cache():
    market_tools.market_cache._cache.clear()
    yield
    market_tools.market_cache._cache.clear()


@pytest.fixture
def synthetic_ohlcv_df() -> pd.DataFrame:
    periods = 260
    idx = pd.date_range("2024-01-01", periods=periods, freq="D")
    x = np.arange(periods, dtype=float)
    close = 100 + 0.2 * x + 3 * np.sin(x / 6)
    open_ = close - 0.3
    high = close + 1.2
    low = close - 1.2
    volume = 1_000_000 + (x * 1000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx)


@pytest.mark.asyncio
async def test_fetch_price_data_success(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.fetch_price_data"""

    class _Res:
        def to_dataframe(self):
            return synthetic_ohlcv_df.copy()

    class _Price:
        @staticmethod
        def historical(*args, **kwargs):
            return _Res()

    class _Equity:
        price = _Price()

    monkeypatch.setattr(market_tools, "obb", SimpleNamespace(equity=_Equity()))

    result = await market_tools.fetch_price_data(["AAPL"], date(2024, 1, 1), date(2024, 9, 30), "1d")
    assert "AAPL" in result
    bars = result["AAPL"]
    assert len(bars) == len(synthetic_ohlcv_df)
    assert all(b.high >= b.low for b in bars), _diag(
        "tools.market_tools.fetch_price_data",
        "mapped bars",
        "high >= low",
        "violation found",
        "DATA_INTEGRITY",
    )
    assert all(b.volume >= 0 for b in bars)
    assert bars == sorted(bars, key=lambda b: b.timestamp)


@pytest.mark.asyncio
async def test_fetch_price_data_openbb_fallback_to_yfinance(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.fetch_price_data"""

    class _BrokenPrice:
        @staticmethod
        def historical(*args, **kwargs):
            raise RuntimeError("openbb down")

    class _BrokenEquity:
        price = _BrokenPrice()

    class _YF:
        def history(self, **kwargs):
            return synthetic_ohlcv_df.copy().rename(columns=str.title)

    monkeypatch.setattr(market_tools, "obb", SimpleNamespace(equity=_BrokenEquity()))
    monkeypatch.setattr(market_tools.yf, "Ticker", lambda ticker: _YF())
    warn_mock = AsyncMock()
    monkeypatch.setattr(market_tools.logger, "warning", lambda *args, **kwargs: None)

    result = await market_tools.fetch_price_data(["AAPL"], date(2024, 1, 1), date(2024, 9, 30), "1d")
    assert "AAPL" in result and len(result["AAPL"]) > 0
    assert warn_mock.await_count == 0


@pytest.mark.asyncio
async def test_fetch_price_data_caching(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.fetch_price_data caching"""
    calls = {"history": 0}

    class _YF:
        def history(self, **kwargs):
            calls["history"] += 1
            return synthetic_ohlcv_df.copy().rename(columns=str.title)

    monkeypatch.setattr(market_tools, "obb", None)
    monkeypatch.setattr(market_tools.yf, "Ticker", lambda ticker: _YF())

    args = (["AAPL"], date(2024, 1, 1), date(2024, 9, 30), "1d")
    await market_tools.fetch_price_data(*args)
    await market_tools.fetch_price_data(*args)
    assert calls["history"] == 1, _diag(
        "tools.market_tools.fetch_price_data",
        "same cache key twice",
        "1 provider call",
        calls["history"],
        "MISSING_FEATURE",
    )


@pytest.mark.asyncio
async def test_fetch_price_data_missing_ticker(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.fetch_price_data missing ticker handling"""

    class _YF:
        def __init__(self, ticker: str):
            self.ticker = ticker

        def history(self, **kwargs):
            if self.ticker == "INVALIDXYZ":
                return pd.DataFrame()
            return synthetic_ohlcv_df.copy().rename(columns=str.title)

    monkeypatch.setattr(market_tools, "obb", None)
    monkeypatch.setattr(market_tools.yf, "Ticker", lambda ticker: _YF(ticker))
    result = await market_tools.fetch_price_data(["AAPL", "INVALIDXYZ"], date(2024, 1, 1), date(2024, 9, 30), "1d")
    assert "AAPL" in result
    assert "INVALIDXYZ" not in result


@pytest.mark.asyncio
async def test_compute_technical_indicators_rsi(synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators RSI"""
    bars = _bars_from_df(synthetic_ohlcv_df, "TEST")
    ind = await market_tools.compute_technical_indicators(bars)

    close = synthetic_ohlcv_df["close"]
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    expected = float((100 - (100 / (1 + rs))).iloc[-1])
    assert ind.rsi_14 is not None
    assert abs(ind.rsi_14 - expected) <= 0.5, _diag(
        "tools.market_tools.compute_technical_indicators",
        "RSI-14",
        expected,
        ind.rsi_14,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_compute_technical_indicators_macd(synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators MACD"""
    bars = _bars_from_df(synthetic_ohlcv_df, "TEST")
    ind = await market_tools.compute_technical_indicators(bars)
    close = synthetic_ohlcv_df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    assert abs(ind.macd["macd"] - float(macd.iloc[-1])) < 1e-6
    assert abs(ind.macd["signal"] - float(signal.iloc[-1])) < 1e-6
    assert abs(ind.macd["histogram"] - float(hist.iloc[-1])) < 1e-6


@pytest.mark.asyncio
async def test_compute_technical_indicators_sma(synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators SMA"""
    bars = _bars_from_df(synthetic_ohlcv_df, "TEST")
    ind = await market_tools.compute_technical_indicators(bars)
    close = synthetic_ohlcv_df["close"]
    assert abs(ind.sma_20 - float(close.rolling(20).mean().iloc[-1])) < 1e-6
    assert abs(ind.sma_50 - float(close.rolling(50).mean().iloc[-1])) < 1e-6
    assert abs(ind.sma_200 - float(close.rolling(200).mean().iloc[-1])) < 1e-6


@pytest.mark.asyncio
async def test_compute_technical_indicators_momentum_roc(synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators momentum ROC"""
    bars = _bars_from_df(synthetic_ohlcv_df, "TEST")
    ind = await market_tools.compute_technical_indicators(bars)
    close = synthetic_ohlcv_df["close"]
    roc10 = float((close.iloc[-1] / close.iloc[-11] - 1.0) * 100.0)
    roc20 = float((close.iloc[-1] / close.iloc[-21] - 1.0) * 100.0)
    assert ind.momentum_roc_10 is not None
    assert ind.momentum_roc_20 is not None
    assert abs(ind.momentum_roc_10 - roc10) < 1e-6
    assert abs(ind.momentum_roc_20 - roc20) < 1e-6


@pytest.mark.asyncio
async def test_compute_technical_indicators_bollinger_bands(synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators Bollinger"""
    bars = _bars_from_df(synthetic_ohlcv_df, "TEST")
    ind = await market_tools.compute_technical_indicators(bars)
    upper = ind.bollinger_bands["upper"]
    middle = ind.bollinger_bands["middle"]
    lower = ind.bollinger_bands["lower"]
    assert upper > middle > lower, _diag(
        "tools.market_tools.compute_technical_indicators",
        "bollinger ordering",
        "upper > middle > lower",
        (upper, middle, lower),
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_compute_technical_indicators_insufficient_data():
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators insufficient data"""
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame(
        {"open": np.arange(10) + 100, "high": np.arange(10) + 101, "low": np.arange(10) + 99, "close": np.arange(10) + 100, "volume": np.full(10, 1000)},
        index=idx,
    )
    bars = _bars_from_df(df, "TEST")
    ind = await market_tools.compute_technical_indicators(bars)
    assert ind.sma_200 is None
    assert ind.sma_50 is None
    assert ind.rsi_14 is None
    assert ind.sma_20 is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rsi", "sma50", "sma200", "spy_close", "vix_close", "expected"),
    [
        (55.0, 105.0, 100.0, 110.0, 15.0, MarketRegime.BULL),
        (45.0, 95.0, 100.0, 90.0, 28.0, MarketRegime.BEAR),
        (49.0, 101.0, 100.0, 102.0, 18.0, MarketRegime.SIDEWAYS),
        (40.0, 105.0, 100.0, 110.0, 22.0, MarketRegime.UNCERTAIN),
    ],
)
async def test_detect_market_regime_cases(
    monkeypatch: pytest.MonkeyPatch,
    rsi: float,
    sma50: float,
    sma200: float,
    spy_close: float,
    vix_close: float,
    expected: MarketRegime,
):
    """FUNCTION TESTED: tools.market_tools.detect_market_regime"""
    spy_bars = [OHLCVBar(timestamp=datetime.now(), open=100, high=101, low=99, close=spy_close, volume=1_000_000, asset="SPY")] * 220
    vix_bars = [OHLCVBar(timestamp=datetime.now(), open=20, high=21, low=19, close=vix_close, volume=1_000_000, asset="^VIX")]

    async def fake_fetch(tickers, start_date, end_date, interval="1d"):
        return {"SPY": spy_bars, "^VIX": vix_bars}

    async def fake_ind(_):
        return TechnicalIndicators(asset="SPY", timestamp=datetime.now(), rsi_14=rsi, sma_50=sma50, sma_200=sma200, adx=10.0 if expected == MarketRegime.SIDEWAYS else 30.0)

    monkeypatch.setattr(market_tools, "fetch_price_data", fake_fetch)
    monkeypatch.setattr(market_tools, "compute_technical_indicators", fake_ind)
    regime, _, _ = await market_tools.detect_market_regime()
    assert regime == expected, _diag(
        "tools.market_tools.detect_market_regime",
        (rsi, sma50, sma200, spy_close, vix_close),
        expected.value,
        regime.value,
        "WRONG_CALCULATION",
    )


@pytest.mark.asyncio
async def test_build_market_snapshot(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.build_market_snapshot"""
    bars = _bars_from_df(synthetic_ohlcv_df, "AAPL")

    async def fake_fetch(tickers, start_date, end_date, interval="1d"):
        out = {}
        for t in tickers:
            if t == "^VIX":
                out[t] = [OHLCVBar(timestamp=datetime.now(), open=18, high=19, low=17, close=18.5, volume=1000, asset=t)]
            else:
                out[t] = bars
        return out

    async def fake_detect():
        return (MarketRegime.BULL, 0.8, {})

    monkeypatch.setattr(market_tools, "fetch_price_data", fake_fetch)
    monkeypatch.setattr(market_tools, "detect_market_regime", fake_detect)
    snap = await market_tools.build_market_snapshot(["AAPL", "MSFT", "TSLA"])
    assert isinstance(snap, MarketSnapshot)
    assert {"AAPL", "MSFT", "TSLA"} <= set(snap.assets.keys())
    assert {"AAPL", "MSFT", "TSLA"} <= set(snap.indicators.keys())
    assert snap.vix is not None
    assert isinstance(snap.sector_performance, dict)


@pytest.mark.asyncio
async def test_get_historical_data_for_backtest(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: tools.market_tools.get_historical_data_for_backtest"""
    bars = _bars_from_df(synthetic_ohlcv_df, "AAPL")

    async def fake_fetch(tickers, start_date, end_date, interval="1d"):
        return {"AAPL": bars}

    monkeypatch.setattr(market_tools, "fetch_price_data", fake_fetch)
    out = await market_tools.get_historical_data_for_backtest(["AAPL"], date(2024, 1, 1), date(2024, 9, 30), "1d")
    df = out["AAPL"]
    assert set(["open", "high", "low", "close", "volume", "asset"]) <= set(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert not df[["close", "volume"]].isna().any().any()


@pytest.mark.asyncio
async def test_screen_stocks(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: tools.market_tools.screen_stocks"""
    bars_ok = [
        OHLCVBar(timestamp=datetime.now(), open=100, high=101, low=99, close=102, volume=2_000_000, asset="AAPL")
        for _ in range(300)
    ]
    bars_bad = [
        OHLCVBar(timestamp=datetime.now(), open=100, high=101, low=99, close=95, volume=100_000, asset="TSLA")
        for _ in range(300)
    ]

    async def fake_fetch(tickers, start_date, end_date, interval="1d"):
        return {t: (bars_ok if t == "AAPL" else bars_bad) for t in tickers}

    async def fake_ind(bars):
        ticker = bars[-1].asset
        if ticker == "AAPL":
            return TechnicalIndicators(asset=ticker, timestamp=datetime.now(), rsi_14=25, sma_200=100)
        return TechnicalIndicators(asset=ticker, timestamp=datetime.now(), rsi_14=45, sma_200=100)

    monkeypatch.setattr(market_tools, "fetch_price_data", fake_fetch)
    monkeypatch.setattr(market_tools, "compute_technical_indicators", fake_ind)
    result = await market_tools.screen_stocks({"universe": ["AAPL", "TSLA"], "rsi_14_below": 30, "above_sma_200": True})
    assert result == ["AAPL"], _diag("tools.market_tools.screen_stocks", "criteria filter", ["AAPL"], result, "WRONG_CALCULATION")


class _MemoryStub:
    def __init__(self) -> None:
        self.working = WorkingMemory(max_items=100)


class _DummyAgent:
    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.mark.asyncio
async def test_run_market_update(monkeypatch: pytest.MonkeyPatch):
    """FUNCTION TESTED: agents.agent2_market.Agent2Market.run_market_update"""
    monkeypatch.setattr("agents.agent2_market.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent2_market.build_agno_model", lambda s: object())
    snap = MarketSnapshot(timestamp=datetime.now(), assets={}, indicators={}, market_breadth={"regime": "bull"}, vix=16.0, sector_performance={})
    monkeypatch.setattr("agents.agent2_market.build_market_snapshot", AsyncMock(return_value=snap))
    agent = Agent2Market(Settings(_env_file=None), _MemoryStub())
    out = await agent.run_market_update(["AAPL", "MSFT"])
    assert isinstance(out, MarketSnapshot)
    stored = await agent.memory.working.get("market_snapshot")
    assert isinstance(stored, MarketSnapshot)


@pytest.mark.asyncio
async def test_analyze_ticker(monkeypatch: pytest.MonkeyPatch, synthetic_ohlcv_df: pd.DataFrame):
    """FUNCTION TESTED: agents.agent2_market.Agent2Market.analyze_ticker"""
    monkeypatch.setattr("agents.agent2_market.Agent", _DummyAgent)
    monkeypatch.setattr("agents.agent2_market.build_agno_model", lambda s: object())
    bars = _bars_from_df(synthetic_ohlcv_df, "AAPL")
    monkeypatch.setattr("agents.agent2_market.fetch_price_data", AsyncMock(return_value={"AAPL": bars}))
    monkeypatch.setattr("agents.agent2_market.compute_technical_indicators", AsyncMock(return_value=TechnicalIndicators(asset="AAPL", timestamp=datetime.now(), rsi_14=48)))
    agent = Agent2Market(Settings(_env_file=None), _MemoryStub())
    result = await agent.analyze_ticker("AAPL")
    assert result["ticker"] == "AAPL"
    assert "indicators" in result


@pytest.mark.asyncio
@pytest.mark.slow
async def test_indicator_computation_performance():
    """FUNCTION TESTED: tools.market_tools.compute_technical_indicators performance"""
    periods = 1000
    idx = pd.date_range("2022-01-01", periods=periods, freq="D")
    close = np.linspace(100, 200, periods)
    df = pd.DataFrame(
        {"open": close - 0.2, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": np.full(periods, 1_000_000)},
        index=idx,
    )
    bars = _bars_from_df(df, "TEST")
    durations = []
    for _ in range(50):
        start = time.perf_counter()
        await market_tools.compute_technical_indicators(bars)
        durations.append((time.perf_counter() - start) * 1000)
    avg_ms = float(np.mean(durations))
    p95 = float(np.percentile(durations, 95))
    p99 = float(np.percentile(durations, 99))
    assert avg_ms < 100.0, _diag(
        "tools.market_tools.compute_technical_indicators",
        {"avg_ms": avg_ms, "p95": p95, "p99": p99},
        "<100ms avg",
        avg_ms,
        "PERFORMANCE",
    )
