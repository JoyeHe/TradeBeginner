"""Global settings loaded from environment variables."""

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for TradeBeginner."""

    llm_provider: str = "deepseek"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_base_url: Optional[str] = "https://api.deepseek.com/v1"
    llm_temperature: float = 0.7

    postgres_url: str = "postgresql+asyncpg://trade:trade@localhost:5432/tradebeginner"
    chroma_persist_dir: str = "./data/chroma"

    news_api_key: str = ""
    finnhub_api_key: str = ""
    openbb_pat: str = ""

    broker_name: str = "alpaca"
    broker_api_key: str = ""
    broker_api_secret: str = ""
    broker_base_url: str = "https://paper-api.alpaca.markets"
    paper_trading: bool = True

    max_position_size_pct: float = 15.0
    max_total_exposure_pct: float = 80.0
    max_stop_loss_pct: float = 5.0
    max_concurrent_positions: int = 10
    max_sector_concentration_pct: float = 30.0
    max_portfolio_drawdown_pct: float = 15.0
    max_crypto_pct: float = 3.0
    min_avg_volume: int = 500000

    backtest_window_days: int = 20
    reward_weight_sharpe: float = 0.5
    reward_weight_drawdown: float = 0.3
    reward_weight_winrate: float = 0.2
    min_trades_for_sharpe: int = 5
    library_min_reward: float = 0.55
    library_min_trades: int = 3
    library_top_k: int = 5
    strategy_library_seed_path: str = "./data/strategy_library/seed_strategies.json"
    strategy_library_learned_path: str = "./data/strategy_library/learned.json"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="ATS_")
