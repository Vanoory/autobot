from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.config import Config


LOGGER = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class Message:
    role: str
    content: str


class MultiLLM:
    def __init__(self, config: Config) -> None:
        self.config = config

    async def complete(self, messages: list[Message], temperature: float = 0.9) -> str:
        errors: list[str] = []

        if self.config.openrouter_api_key:
            try:
                return await self._openai_compatible(
                    base_url="https://openrouter.ai/api/v1/chat/completions",
                    api_key=self.config.openrouter_api_key,
                    model=self.config.openrouter_model,
                    messages=messages,
                    temperature=temperature,
                    extra_headers={
                        "HTTP-Referer": "https://railway.app",
                        "X-Title": "Trading TG Bot",
                    },
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"OpenRouter: {exc}")

        if self.config.groq_api_key:
            try:
                return await self._openai_compatible(
                    base_url="https://api.groq.com/openai/v1/chat/completions",
                    api_key=self.config.groq_api_key,
                    model=self.config.groq_model,
                    messages=messages,
                    temperature=temperature,
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Groq: {exc}")

        if self.config.gemini_api_key:
            try:
                return await self._gemini(messages=messages, temperature=temperature)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"Gemini: {exc}")

        if errors:
            LOGGER.warning("All remote LLM providers failed: %s", " | ".join(errors))
        raise LLMError("No remote LLM provider available")

    async def _openai_compatible(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        messages: list[Message],
        temperature: float,
        extra_headers: dict[str, str] | None = None,
    ) -> str:
        headers = {"Authorization": f"Bearer {api_key}"}
        if extra_headers:
            headers.update(extra_headers)

        payload = {
            "model": model,
            "messages": [{"role": item.role, "content": item.content} for item in messages],
            "temperature": temperature,
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(base_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Invalid provider response: {data}") from exc

    async def _gemini(self, *, messages: list[Message], temperature: float) -> str:
        combined = []
        for message in messages:
            combined.append(f"{message.role.upper()}:\n{message.content}")
        prompt = "\n\n".join(combined)

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.config.gemini_model}:generateContent?key={self.config.gemini_api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature},
        }
        async with httpx.AsyncClient(timeout=45) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            return "".join(part["text"] for part in parts if "text" in part).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Invalid Gemini response: {data}") from exc
