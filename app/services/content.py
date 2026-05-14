from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from app.config import Config
from app.database import Database
from app.services.knowledge import KnowledgeBase
from app.services.llm import LLMError, Message, MultiLLM
from app.services.market import MarketService, SignalCandidate
from app.services.rendering import Renderer


class ContentService:
    def __init__(
        self,
        config: Config,
        database: Database,
        llm: MultiLLM,
        knowledge: KnowledgeBase,
        market: MarketService,
        renderer: Renderer,
    ) -> None:
        self.config = config
        self.database = database
        self.llm = llm
        self.knowledge = knowledge
        self.market = market
        self.renderer = renderer
        self.timezone = ZoneInfo(config.timezone)

    async def generate_morning_post(self) -> tuple[str, dict[str, Any]]:
        snapshot = await self.market.market_snapshot(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
        text = self._build_morning_post(snapshot)
        return text, {"kind": "morning", "snapshot": snapshot}

    async def generate_news_post(self) -> tuple[str, str | None, dict[str, Any]]:
        events = await self._fetch_investing_calendar()
        caption = self._build_news_caption(events)

        card_items = []
        for item in events[:4]:
            card_items.append(
                {
                    "time": item["time"],
                    "tag": item["currency"],
                    "headline": item["headline"],
                    "meta": item["meta"],
                }
            )

        image_path = self.renderer.render_news_card("Новости на сегодня", card_items)
        payload = {"kind": "news", "events": events, "source": "investing_economic_calendar"}
        return caption, str(image_path), payload

    async def generate_signal_post(self, symbols: list[str] | None = None) -> tuple[str, str | None, dict[str, Any]] | None:
        candidate = await self.market.best_signal(symbols or self.config.signal_symbols)
        if candidate is None:
            return None

        recent_posts = await self.database.recent_published_texts()
        knowledge = await self.knowledge.retrieve(
            f"анализ графика {candidate.symbol} вход стоп тейк стиль короткий пост"
        )
        chart_path = self.renderer.render_signal_chart(candidate)

        prompt = (
            "Напиши пост для приватного трейдинг-канала на русском языке. "
            "Нужно максимально похоже на обычный человеческий пост, как в закрытом тгк, а не как ответ нейросети.\n\n"
            "Примеры формата:\n"
            "#BTC\n\n"
            "Сейчас цена снимает пул ликвидности на уровне 79,181 и оттуда мы можем увидеть локальный разворот.\n\n"
            "Или:\n"
            "BTC/USDT | 1D\n\n"
            "Ожидаю небольшую коррекцию к зоне, и оттуда уже буду рассматривать лонг.\n\n"
            "Правила:\n"
            "- очень коротко\n"
            "- только заголовок и 1 короткий абзац\n"
            "- без списков\n"
            "- без лишней аналитической воды\n"
            "- не расписывай стоп и тейки текстом, они уже есть на графике\n"
            "- не пиши как робот\n"
            "- не используй фразы вроде 'данный актив', 'в целом', 'с высокой вероятностью'\n\n"
            f"Данные сетапа:\n{self._candidate_summary(candidate)}\n\n"
            f"Последние посты:\n{self._join_or_none(recent_posts)}\n\n"
            f"База знаний:\n{self._join_or_none(knowledge)}"
        )
        fallback = self._fallback_signal(candidate)
        text = await self._generate_text(prompt, fallback=fallback, temperature=0.7)
        payload = {
            "kind": "signal",
            "symbol": candidate.symbol,
            "side": candidate.side,
            "timeframe": candidate.timeframe,
            "entry": candidate.entry,
            "stop": candidate.stop,
            "take1": candidate.take1,
            "take2": candidate.take2,
            "current_price": candidate.current_price,
            "reason": candidate.reason,
            "indicators": candidate.indicators,
        }
        return text, str(chart_path), payload

    async def generate_take1_update(self, signal_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        prompt = (
            "Напиши очень короткий апдейт в трейдинг-канал: первый тейк уже взяли, стоп переводим в б/у. "
            "Максимум 2 короткие строки, живой телеграм-стиль."
        )
        fallback = (
            f"По {signal_payload['symbol']} первый тейк уже забрали.\n\n"
            "Стоп переводим в б/у."
        )
        text = await self._generate_text(prompt, fallback=fallback, temperature=0.6)
        return text, {"kind": "take1", **signal_payload}

    async def _generate_text(self, prompt: str, fallback: str, temperature: float = 0.9) -> str:
        messages = [
            Message(
                role="system",
                content=(
                    "Ты пишешь для закрытого трейдинг-канала. "
                    "Стиль максимально простой, человеческий и короткий. "
                    "Текст должен выглядеть как сообщение от автора канала, а не как ответ ИИ."
                ),
            ),
            Message(role="user", content=prompt),
        ]
        try:
            text = await self.llm.complete(messages, temperature=temperature)
            return self._sanitize_generated_text(text)
        except LLMError:
            return fallback

    async def _fetch_investing_calendar(self) -> list[dict[str, str]]:
        url = "https://ru.investing.com/economic-calendar"
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        }
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            html = response.text

        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html)
        if not match:
            return self._fallback_calendar_events()

        data = json.loads(match.group(1))
        state = data["props"]["pageProps"]["state"]
        store = state["economicCalendarStore"]["calendarEventsByDate"]
        local_date = datetime.now(self.timezone).date().isoformat()
        raw_events = store.get(local_date)
        if raw_events is None and store:
            first_key = sorted(store.keys())[0]
            raw_events = store[first_key]
        if not raw_events:
            return self._fallback_calendar_events()

        items: list[dict[str, str]] = []
        for item in raw_events:
            if item.get("type") != "event":
                continue
            if item.get("currencyFlag") != "US" and item.get("country") != "United States":
                continue
            if item.get("importance") != "3":
                continue
            items.append(
                {
                    "time": self._format_event_time(item.get("time")),
                    "currency": item.get("currency", "USD"),
                    "headline": self._compose_event_title(item),
                    "meta": self._compose_event_meta(item),
                }
            )

        items.sort(key=lambda event: event["time"])
        return items[:6] if items else self._fallback_calendar_events()

    def _format_event_time(self, value: str | None) -> str:
        if not value:
            return "Сегодня"
        event_time = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(self.timezone)
        return event_time.strftime("%H:%M")

    def _compose_event_title(self, item: dict[str, Any]) -> str:
        title = (item.get("eventLong") or item.get("event") or "").strip()
        title = title.replace("в США", "").strip()
        period = (item.get("period") or "").strip()
        if period:
            title = f"{title} {period}"
        return " ".join(title.split())

    def _compose_event_meta(self, item: dict[str, Any]) -> str:
        actual = (item.get("actual") or "").strip() or "-"
        forecast = (item.get("forecast") or "").strip() or "-"
        previous = (item.get("previous") or "").strip() or "-"
        return f"Факт: {actual} | Прог: {forecast} | Пред.: {previous}"

    def _fallback_calendar_events(self) -> list[dict[str, str]]:
        return [
            {
                "time": "15:30",
                "currency": "USD",
                "headline": "Объём розничных продаж (м/м)",
                "meta": "Факт: - | Прог: - | Пред.: -",
            }
        ]

    def _build_news_caption(self, events: list[dict[str, str]]) -> str:
        if not events:
            return "Новости на сегодня скинул выше."
        main_time = events[0]["time"]
        if len({item["time"] for item in events}) == 1:
            return (
                "Новости на сегодня скинул выше.\n\n"
                f"Сегодня в фокусе блок новостей по США в {main_time}."
            )
        return (
            "Новости на сегодня скинул выше.\n\n"
            "Сегодня внимательно слежу за новостным фоном по США."
        )

    def _build_morning_post(self, snapshot: list[dict[str, Any]]) -> str:
        btc_change = 0.0
        for item in snapshot:
            if item["symbol"] == "BTCUSDT":
                btc_change = float(item["change_24h"])
                break

        if abs(btc_change) < 1.2:
            mood = "Сегодня рынок выглядит спокойно."
        elif btc_change > 0:
            mood = "С утра рынок выглядит чуть бодрее."
        else:
            mood = "С утра рынок немного откатился."

        templates = [
            f"Доброе утро 🌞\n\n{mood}\n\nСажусь смотреть рынок, буду искать для вас интересные сетапчики.",
            f"Всем привет ✋\n\n{mood}\n\nСейчас пробегусь по графикам и если будет что-то интересное, сразу скину.",
            f"Доброе утро 👋\n\n{mood}\n\nБуду смотреть рынок и искать хорошие входы.",
        ]
        day_index = datetime.now(self.timezone).timetuple().tm_yday % len(templates)
        return templates[day_index]

    def _sanitize_generated_text(self, text: str) -> str:
        lines = [line.rstrip() for line in text.strip().splitlines()]
        cleaned: list[str] = []
        for line in lines:
            candidate = line.strip().strip('"').strip("'")
            if re.match(r"^\d+\.", candidate):
                candidate = re.sub(r"^\d+\.\s*", "", candidate)
            if candidate:
                cleaned.append(candidate)
        return "\n".join(cleaned).strip()

    def _join_or_none(self, items: list[str]) -> str:
        return "\n\n".join(items) if items else "нет данных"

    def _candidate_summary(self, candidate: SignalCandidate) -> str:
        return (
            f"Монета: {candidate.symbol}\n"
            f"Сторона: {candidate.side}\n"
            f"Таймфрейм: {candidate.timeframe}\n"
            f"Вход: {candidate.entry}\n"
            f"Стоп: {candidate.stop}\n"
            f"Тейк 1: {candidate.take1}\n"
            f"Тейк 2: {candidate.take2}\n"
            f"Причина: {candidate.reason}\n"
            f"Индикаторы: {candidate.indicators}"
        )

    def _fallback_signal(self, candidate: SignalCandidate) -> str:
        coin = candidate.symbol.replace("USDT", "")
        if candidate.side == "long":
            body = f"Смотрю лонг по {coin} от этой зоны. Если дадут реакцию, можно будет забрать движение выше."
        else:
            body = f"Смотрю шорт по {coin} от этой зоны. Если цену сюда еще раз подтянут, можно будет попробовать движение ниже."
        return f"#{coin}\n\n{body}"
