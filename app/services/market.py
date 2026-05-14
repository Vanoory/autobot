from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import mean
from typing import Any

import httpx
import pandas as pd


BINANCE_API = "https://api.binance.com"


@dataclass(slots=True)
class SignalCandidate:
    symbol: str
    side: str
    timeframe: str
    entry: float
    stop: float
    take1: float
    take2: float
    current_price: float
    score: float
    reason: str
    candles: pd.DataFrame
    indicators: dict[str, float]


class MarketService:
    async def fetch_klines(self, symbol: str, interval: str = "1h", limit: int = 220) -> pd.DataFrame:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{BINANCE_API}/api/v3/klines", params=params)
            response.raise_for_status()
            raw = response.json()

        rows: list[dict[str, Any]] = []
        for item in raw:
            rows.append(
                {
                    "open_time": datetime.fromtimestamp(item[0] / 1000, UTC),
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                }
            )
        df = pd.DataFrame(rows)
        return self._add_indicators(df)

    async def fetch_price(self, symbol: str) -> float:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(f"{BINANCE_API}/api/v3/ticker/price", params={"symbol": symbol})
            response.raise_for_status()
            data = response.json()
        return float(data["price"])

    async def best_signal(self, symbols: list[str]) -> SignalCandidate | None:
        candidates: list[SignalCandidate] = []
        for symbol in symbols:
            try:
                df = await self.fetch_klines(symbol)
                candidate = self._build_signal(symbol, df)
            except Exception:
                continue
            if candidate:
                candidates.append(candidate)

        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.score, reverse=True)[0]

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = (-delta.clip(upper=0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, pd.NA)
        df["rsi14"] = 100 - (100 / (1 + rs))

        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr14"] = tr.rolling(window=14).mean()
        df["vol_ma20"] = df["volume"].rolling(window=20).mean()
        return df.dropna().reset_index(drop=True)

    def _build_signal(self, symbol: str, df: pd.DataFrame) -> SignalCandidate | None:
        if len(df) < 80:
            return None

        last = df.iloc[-1]
        prev = df.iloc[-2]
        recent_high = df["high"].tail(24).max()
        recent_low = df["low"].tail(24).min()
        atr = float(last["atr14"])
        price = float(last["close"])
        ema20 = float(last["ema20"])
        ema50 = float(last["ema50"])
        rsi = float(last["rsi14"])
        vol_ratio = float(last["volume"] / max(last["vol_ma20"], 1))

        long_conditions = [
            ema20 > ema50 * 0.998,
            price >= ema20 * 0.985,
            42 <= rsi <= 69,
            price >= prev["close"] * 0.997,
        ]
        short_conditions = [
            ema20 < ema50 * 1.002,
            price <= ema20 * 1.015,
            31 <= rsi <= 58,
            price <= prev["close"] * 1.003,
        ]

        long_score = sum(bool(item) for item in long_conditions)
        short_score = sum(bool(item) for item in short_conditions)

        if long_score >= 3 and ema20 >= ema50:
            stop = min(float(df["low"].tail(10).min()), price - atr * 1.1)
            risk = price - stop
            if risk <= 0:
                return None
            take1 = price + risk * 1.6
            take2 = price + risk * 2.4
            score = self._score_candidate(ema20, ema50, rsi, vol_ratio, price, recent_high, side="long")
            score += long_score * 0.35
            reason = "Тренд выше скользящих, цена удерживается в рабочей зоне и может продолжить движение по импульсу."
            return SignalCandidate(
                symbol=symbol,
                side="long",
                timeframe="1h",
                entry=round(price, 4),
                stop=round(stop, 4),
                take1=round(take1, 4),
                take2=round(take2, 4),
                current_price=round(price, 4),
                score=score,
                reason=reason,
                candles=df.tail(90).reset_index(drop=True),
                indicators={
                    "ema20": round(ema20, 4),
                    "ema50": round(ema50, 4),
                    "rsi14": round(rsi, 2),
                    "atr14": round(atr, 4),
                    "volume_ratio": round(vol_ratio, 2),
                },
            )

        if short_score >= 3 and ema20 <= ema50:
            stop = max(float(df["high"].tail(10).max()), price + atr * 1.1)
            risk = stop - price
            if risk <= 0:
                return None
            take1 = price - risk * 1.6
            take2 = price - risk * 2.4
            score = self._score_candidate(ema20, ema50, rsi, vol_ratio, price, recent_low, side="short")
            score += short_score * 0.35
            reason = "Монета остается под давлением, цена ниже рабочих средних и продавец пока сохраняет контроль."
            return SignalCandidate(
                symbol=symbol,
                side="short",
                timeframe="1h",
                entry=round(price, 4),
                stop=round(stop, 4),
                take1=round(take1, 4),
                take2=round(take2, 4),
                current_price=round(price, 4),
                score=score,
                reason=reason,
                candles=df.tail(90).reset_index(drop=True),
                indicators={
                    "ema20": round(ema20, 4),
                    "ema50": round(ema50, 4),
                    "rsi14": round(rsi, 2),
                    "atr14": round(atr, 4),
                    "volume_ratio": round(vol_ratio, 2),
                },
            )
        return None

    def _score_candidate(
        self,
        ema20: float,
        ema50: float,
        rsi: float,
        vol_ratio: float,
        price: float,
        pivot: float,
        *,
        side: str,
    ) -> float:
        trend_strength = abs(ema20 - ema50) / max(price, 1)
        proximity = 1 - min(abs(price - pivot) / max(price, 1), 1)
        rsi_component = (70 - abs(55 - rsi)) / 100
        if side == "short":
            rsi_component = (70 - abs(45 - rsi)) / 100
        volume_component = min(vol_ratio, 2.0) / 2
        return round(trend_strength * 200 + proximity * 3 + rsi_component * 2 + volume_component, 4)

    async def market_snapshot(self, symbols: list[str]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for symbol in symbols[:5]:
            try:
                df = await self.fetch_klines(symbol, limit=60)
            except Exception:
                continue
            last = df.iloc[-1]
            change = ((last["close"] / df.iloc[-24]["close"]) - 1) * 100 if len(df) >= 24 else 0
            items.append(
                {
                    "symbol": symbol,
                    "price": round(float(last["close"]), 4),
                    "change_24h": round(change, 2),
                    "rsi14": round(float(last["rsi14"]), 1),
                }
            )
        items.sort(key=lambda item: abs(item["change_24h"]), reverse=True)
        return items

    def take1_hit(self, signal: dict[str, Any], price: float) -> bool:
        if signal["side"] == "long":
            return price >= signal["take1"]
        return price <= signal["take1"]

    def stop_hit(self, signal: dict[str, Any], price: float) -> bool:
        if signal["side"] == "long":
            return price <= signal["stop"]
        return price >= signal["stop"]
