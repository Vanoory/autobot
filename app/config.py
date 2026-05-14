from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = ROOT_DIR / "storage"
DB_PATH = STORAGE_DIR / "bot.sqlite3"


@dataclass(slots=True)
class Config:
    bot_token: str
    owner_id: int
    channel_id: str
    timezone: str
    morning_post_time: str
    news_post_time: str
    signal_scan_interval_minutes: int
    price_monitor_interval_minutes: int
    max_open_signals: int
    signal_symbols: list[str]
    market_data_providers: list[str]
    openrouter_api_key: str | None
    openrouter_model: str
    groq_api_key: str | None
    groq_model: str
    gemini_api_key: str | None
    gemini_model: str
    coingecko_api_key: str | None
    coinmarketcal_api_key: str | None
    root_dir: Path
    storage_dir: Path
    db_path: Path


def _required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is required")
    return value


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _parse_csv(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def load_config() -> Config:
    load_dotenv()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    for subdir in ("charts", "news", "knowledge", "tmp"):
        (STORAGE_DIR / subdir).mkdir(parents=True, exist_ok=True)

    return Config(
        bot_token=_required("BOT_TOKEN"),
        owner_id=int(_required("OWNER_ID")),
        channel_id=_required("CHANNEL_ID"),
        timezone=os.getenv("TIMEZONE", "Europe/Kiev").strip(),
        morning_post_time=os.getenv("MORNING_POST_TIME", "09:00").strip(),
        news_post_time=os.getenv("NEWS_POST_TIME", "11:00").strip(),
        signal_scan_interval_minutes=int(os.getenv("SIGNAL_SCAN_INTERVAL_MINUTES", "180")),
        price_monitor_interval_minutes=int(os.getenv("PRICE_MONITOR_INTERVAL_MINUTES", "5")),
        max_open_signals=int(os.getenv("MAX_OPEN_SIGNALS", "3")),
        signal_symbols=_parse_csv(
            os.getenv(
                "SIGNAL_SYMBOLS",
                "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT",
            )
        ),
        market_data_providers=_parse_csv(
            os.getenv(
                "MARKET_DATA_PROVIDERS",
                "BYBIT,OKX,BINANCE",
            )
        ),
        openrouter_api_key=_optional("OPENROUTER_API_KEY"),
        openrouter_model=os.getenv(
            "OPENROUTER_MODEL",
            "deepseek/deepseek-chat-v3-0324:free",
        ).strip(),
        groq_api_key=_optional("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
        gemini_api_key=_optional("GEMINI_API_KEY"),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip(),
        coingecko_api_key=_optional("COINGECKO_API_KEY"),
        coinmarketcal_api_key=_optional("COINMARKETCAL_API_KEY"),
        root_dir=ROOT_DIR,
        storage_dir=STORAGE_DIR,
        db_path=DB_PATH,
    )
