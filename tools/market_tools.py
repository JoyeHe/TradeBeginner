"""Market tools using OpenBB first, yfinance fallback."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Optional

import numpy as np
import pandas as pd
import structlog
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from schemas.market_data import MarketSnapshot, OHLCVBar, TechnicalIndicators
from schemas.strategy import MarketRegime

logger = structlog.get_logger(__name__)

try:
    from openbb import obb  # type: ignore
except Exception:  # pragma: no cover
    obb = None


@dataclass
class CacheEntry:
    expires_at: datetime
    value: Any


class MarketDataCache:
    """Simple in-memory TTL cache keyed by deterministic string."""

    def __init__(self) -> None:
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Any:
        entry = self._cache.get(key)
        if entry is None or datetime.utcnow() > entry.expires_at:
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        self._cache[key] = CacheEntry(expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds), value=value)


market_cache = MarketDataCache()


def _interval_ttl(interval: str) -> int:
    return 4 * 3600 if interval.endswith("d") else 5 * 60


def _apply_openbb_credentials() -> None:
    """Sync TradeBeginner env vars into OpenBB credential env names."""
    pat = os.getenv("ATS_OPENBB_PAT") or os.getenv("OPENBB_PAT")
    if pat:
        if pat.startswith("obb_mcp_"):
            logger.warning(
                "openbb_mcp_token_ignored",
                hint="Workspace MCP tokens (obb_mcp_*) do not work for SDK/API. Use ~/.openbb_platform/.env API keys instead.",
            )
        else:
            os.environ["OPENBB_PAT"] = pat

    for cred_key, env_names in (
        ("FMP_API_KEY", ("ATS_FMP_API_KEY", "FMP_API_KEY")),
        ("FRED_API_KEY", ("ATS_FRED_API_KEY", "FRED_API_KEY")),
        ("ALPHA_VANTAGE_API_KEY", ("ATS_ALPHA_VANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY")),
    ):
        if os.getenv(cred_key):
            continue
        for name in env_names:
            value = os.getenv(name)
            if value:
                os.environ[cred_key] = value
                break


def _openbb_price_providers(ticker: str) -> tuple[str, ...]:
    """Pick OpenBB providers; CBOE works for US equities/ETFs/VIX without API keys."""
    if ticker.endswith("-USD"):
        return ("fmp", "yfinance")
    return ("cboe", "fmp", "yfinance")


def _fetch_openbb_df(ticker: str, start_date: date, end_date: date, interval: str) -> pd.DataFrame | None:
    if obb is None:
        return None
    _apply_openbb_credentials()
    last_error: str | None = None
    for provider in _openbb_price_providers(ticker):
        try:
            res = obb.equity.price.historical(
                ticker,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
                interval=interval,
                provider=provider,
            )
            df = res.to_dataframe()
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            last_error = str(exc)
            logger.debug("openbb_provider_failed", ticker=ticker, provider=provider, error=last_error)
    if last_error:
        logger.warning("openbb_fetch_failed", ticker=ticker, error=last_error)
    return None


def _fetch_yfinance_df(ticker: str, start_date: date, end_date: date, interval: str) -> pd.DataFrame | None:
    try:
        hist = yf.Ticker(ticker).history(start=start_date.isoformat(), end=end_date.isoformat(), interval=interval)
        if hist.empty:
            return None
        return hist.rename(columns=str.lower)
    except Exception as exc:
        logger.warning("yfinance_fetch_failed", ticker=ticker, error=str(exc))
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
async def fetch_price_data(tickers: list[str], start_date: date, end_date: date, interval: str = "1d") -> dict[str, list[OHLCVBar]]:
    """Fetch OHLCV bars for tickers with caching and fallback."""
    output: dict[str, list[OHLCVBar]] = {}
    for ticker in tickers:
        key = f"{ticker}:{start_date}:{end_date}:{interval}"
        cached = market_cache.get(key)
        if cached is not None:
            output[ticker] = cached
            continue

        df: Optional[pd.DataFrame] = await asyncio.to_thread(_fetch_openbb_df, ticker, start_date, end_date, interval)
        if df is None or df.empty:
            df = await asyncio.to_thread(_fetch_yfinance_df, ticker, start_date, end_date, interval)

        if df is None or df.empty:
            logger.warning("ticker_data_missing", ticker=ticker)
            continue

        bars = [
            OHLCVBar(
                timestamp=pd.Timestamp(idx).to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                asset=ticker,
            )
            for idx, row in df.iterrows()
        ]
        market_cache.set(key, bars, _interval_ttl(interval))
        output[ticker] = bars
    return output


def _series_from_bars(bars: list[OHLCVBar], field: str) -> pd.Series:
    return pd.Series([getattr(x, field) for x in bars], dtype=float)


async def compute_technical_indicators(ohlcv_data: list[OHLCVBar]) -> TechnicalIndicators:
    """Compute latest technical indicator values."""
    asset = ohlcv_data[-1].asset
    ts = ohlcv_data[-1].timestamp
    close = _series_from_bars(ohlcv_data, "close")
    high = _series_from_bars(ohlcv_data, "high")
    low = _series_from_bars(ohlcv_data, "low")
    volume = _series_from_bars(ohlcv_data, "volume")

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).iloc[-1] if len(close) >= 14 else np.nan

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()

    obv = (np.sign(close.diff().fillna(0)) * volume).cumsum()
    plus_dm = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr14 = tr.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).sum() / tr14.replace(0, np.nan))
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx = dx.rolling(14).mean()

    def v(series: pd.Series) -> Optional[float]:
        x = series.iloc[-1] if not series.empty else np.nan
        return None if pd.isna(x) else float(x)

    return TechnicalIndicators(
        asset=asset,
        timestamp=ts,
        rsi_14=None if pd.isna(rsi) else float(rsi),
        macd={"macd": v(macd_line), "signal": v(signal), "histogram": v(macd_line - signal)},
        sma_20=v(close.rolling(20).mean()),
        sma_50=v(close.rolling(50).mean()),
        sma_200=v(close.rolling(200).mean()),
        ema_12=v(ema12),
        ema_26=v(ema26),
        bollinger_bands={"upper": v(bb_mid + 2 * bb_std), "middle": v(bb_mid), "lower": v(bb_mid - 2 * bb_std)},
        atr_14=v(atr14),
        volume_sma_20=v(volume.rolling(20).mean()),
        obv=v(obv),
        adx=v(adx),
    )


async def detect_market_regime() -> tuple[MarketRegime, float, dict]:
    """Classify current regime using SPY and VIX proxies."""
    today = date.today()
    bars = await fetch_price_data(["SPY", "^VIX"], today - timedelta(days=280), today, interval="1d")
    spy = bars.get("SPY", [])
    vix = bars.get("^VIX", [])
    if len(spy) < 210:
        return (MarketRegime.UNCERTAIN, 0.2, {"reason": "insufficient_data"})
    ind = await compute_technical_indicators(spy)
    vix_close = vix[-1].close if vix else None
    evidence = {"spy_close": spy[-1].close, "sma200": ind.sma_200, "sma50": ind.sma_50, "rsi": ind.rsi_14, "vix": vix_close}
    if ind.sma_200 and ind.sma_50 and ind.rsi_14 and vix_close is not None:
        if spy[-1].close > ind.sma_200 and ind.sma_50 > ind.sma_200 and ind.rsi_14 > 50 and vix_close < 20:
            return (MarketRegime.BULL, 0.8, evidence)
        if spy[-1].close < ind.sma_200 and ind.sma_50 < ind.sma_200 and ind.rsi_14 < 50 and vix_close > 25:
            return (MarketRegime.BEAR, 0.8, evidence)
        if (ind.adx or 30) < 20:
            return (MarketRegime.SIDEWAYS, 0.6, evidence)
        if vix_close > 30:
            return (MarketRegime.VOLATILE, 0.8, evidence)
    return (MarketRegime.UNCERTAIN, 0.4, evidence)


async def build_market_snapshot(watchlist: list[str]) -> MarketSnapshot:
    """Build complete snapshot including indicators and sector ETFs."""
    today = date.today()
    price = await fetch_price_data(watchlist, today - timedelta(days=300), today, interval="1d")
    assets: dict[str, OHLCVBar] = {}
    indicators: dict[str, TechnicalIndicators] = {}
    for ticker, bars in price.items():
        if not bars:
            continue
        assets[ticker] = bars[-1]
        indicators[ticker] = await compute_technical_indicators(bars)

    regime, _, _ = await detect_market_regime()
    sector_etfs = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB"]
    sector_data = await fetch_price_data(sector_etfs, today - timedelta(days=7), today)
    sector_perf = {}
    for etf, bars in sector_data.items():
        if len(bars) >= 2:
            sector_perf[etf] = (bars[-1].close / bars[-2].close) - 1
    vix_data = await fetch_price_data(["^VIX"], today - timedelta(days=7), today)
    vix = vix_data.get("^VIX", [None])[-1]
    return MarketSnapshot(
        timestamp=datetime.utcnow(),
        assets=assets,
        indicators=indicators,
        market_breadth={"regime": regime.value},
        vix=None if vix is None else vix.close,
        sector_performance=sector_perf,
    )


async def get_historical_data_for_backtest(
    tickers: list[str], start_date: date, end_date: date, interval: str = "1d"
) -> dict[str, pd.DataFrame]:
    """Return historical bars as DataFrame for backtest engine."""
    bars = await fetch_price_data(tickers, start_date=start_date, end_date=end_date, interval=interval)
    out: dict[str, pd.DataFrame] = {}
    for ticker, series in bars.items():
        out[ticker] = pd.DataFrame([x.model_dump() for x in series]).set_index("timestamp")
    return out


CRYPTO_ALIASES = {
    "BTC": "BTC-USD", "BITCOIN": "BTC-USD",
    "ETH": "ETH-USD", "ETHEREUM": "ETH-USD",
    "SOL": "SOL-USD", "SOLANA": "SOL-USD",
    "BNB": "BNB-USD",
    "XRP": "XRP-USD", "RIPPLE": "XRP-USD",
    "DOGE": "DOGE-USD", "DOGECOIN": "DOGE-USD",
    "ADA": "ADA-USD", "CARDANO": "ADA-USD",
    "DOT": "DOT-USD", "POLKADOT": "DOT-USD",
    "AVAX": "AVAX-USD", "AVALANCHE": "AVAX-USD",
    "MATIC": "MATIC-USD", "POLYGON": "MATIC-USD",
    "LINK": "LINK-USD", "CHAINLINK": "LINK-USD",
    "SHIB": "SHIB-USD",
    "UNI": "UNI-USD", "UNISWAP": "UNI-USD",
    "ATOM": "ATOM-USD", "COSMOS": "ATOM-USD",
    "LTC": "LTC-USD", "LITECOIN": "LTC-USD",
    "NEAR": "NEAR-USD",
    "APT": "APT-USD", "APTOS": "APT-USD",
    "ARB": "ARB-USD", "ARBITRUM": "ARB-USD",
    "OP": "OP-USD", "OPTIMISM": "OP-USD",
    "SUI": "SUI-USD",
    "PEPE": "PEPE-USD",
}

VALID_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk", "1mo"}


def _resolve_tickers(query: str) -> list[str]:
    import re as _re
    STOP = {"THE", "AND", "FOR", "NOT", "BUY", "SELL", "NOW", "WHAT", "HOW", "GET", "PRICE", "LAST", "DAY", "DAYS", "WEEK", "MONTH", "YEAR"}
    candidates = [tok for tok in _re.findall(r"\b[A-Z][A-Z0-9\-]{1,9}\b", query.upper()) if tok not in STOP]
    tickers = [CRYPTO_ALIASES.get(c, c) for c in candidates]
    return tickers if tickers else ["SPY"]


async def user_market_lookup(
    query: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
) -> dict[str, Any]:
    """Fetch price data + indicators for tickers extracted from query.

    Supports user-adjustable date window and interval aligned with OpenBB/yfinance:
    intervals: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo
    """
    tickers = _resolve_tickers(query)

    if interval not in VALID_INTERVALS:
        interval = "1d"

    today = date.today()
    try:
        sd = date.fromisoformat(start_date) if start_date else today - timedelta(days=100)
    except ValueError:
        sd = today - timedelta(days=100)
    try:
        ed = date.fromisoformat(end_date) if end_date else today
    except ValueError:
        ed = today

    bars_map = await fetch_price_data(list(set(tickers)), sd, ed, interval=interval)
    results: dict[str, Any] = {}
    for ticker, series in bars_map.items():
        if not series:
            continue
        ind = await compute_technical_indicators(series)
        last = series[-1]
        bars_list = [
            {
                "date": b.timestamp.isoformat(),
                "open": round(b.open, 4),
                "high": round(b.high, 4),
                "low": round(b.low, 4),
                "close": round(b.close, 4),
                "volume": b.volume,
            }
            for b in series
        ]
        results[ticker] = {
            "last_bar": last.model_dump(mode="json"),
            "bars_count": len(series),
            "bars": bars_list,
            "indicators": {
                "rsi_14": ind.rsi_14,
                "sma_20": ind.sma_20,
                "sma_50": ind.sma_50,
                "macd": ind.macd,
                "bollinger_bands": ind.bollinger_bands,
                "atr_14": ind.atr_14,
                "adx": ind.adx,
            },
        }
    return {
        "query": query,
        "tickers": list(results.keys()),
        "start_date": sd.isoformat(),
        "end_date": ed.isoformat(),
        "interval": interval,
        "data": results,
    }


async def screen_stocks(criteria: dict) -> list[str]:
    """Basic stock screener over a small starter universe."""
    universe = criteria.get("universe", ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "XOM", "BTC-USD"])
    today = date.today()
    bars_map = await fetch_price_data(universe, today - timedelta(days=300), today)
    result = []
    for ticker, bars in bars_map.items():
        if len(bars) < 30:
            continue
        ind = await compute_technical_indicators(bars)
        last = bars[-1]
        if criteria.get("rsi_14_below") is not None and (ind.rsi_14 is None or ind.rsi_14 >= criteria["rsi_14_below"]):
            continue
        if criteria.get("above_sma_200") and (ind.sma_200 is None or last.close <= ind.sma_200):
            continue
        if criteria.get("min_volume") and last.volume < criteria["min_volume"]:
            continue
        result.append(ticker)
    return result

