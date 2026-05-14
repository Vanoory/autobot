from __future__ import annotations

import re
from html import escape
from typing import Any


PRICE_RANGE_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?\s*[-–]\s*\d+(?:[.,]\d+)?(?:\$)?)(?!%)")
PERCENT_RE = re.compile(r"(?<!\w)(\d+(?:[.,]\d+)?(?:\s*[-–]\s*\d+(?:[.,]\d+)?)?%)")
KEYWORD_RE = re.compile(
    r"(?i)\b(лонг(?:и|ом|а)?|шорт(?:ы|ом|а)?|б/у|безубыток|тейк(?: ?\d)?|стоп|риск(?:ом|а)?|вне марафона)\b"
)
TAG_RE = re.compile(r"^#\S+")


def format_post_html(kind: str, text: str, payload: dict[str, Any] | None = None) -> str:
    raw = text.strip()
    if not raw:
        return ""

    payload = payload or {}
    if kind == "signal":
        return _format_signal(raw, payload)
    if kind == "take1":
        return _format_take1(raw, payload)
    if kind == "news":
        return _format_lead_and_body(raw, default_title="Новости")
    if kind == "morning":
        return _format_lead_and_body(raw, default_title="Доброе утро")
    return _format_lead_and_body(raw)


def plain_text_for_caption(html_text: str, limit: int = 1024) -> str:
    plain = _strip_tags(html_text)
    if len(plain) <= limit:
        return html_text
    shortened = plain[: max(limit - 1, 0)].rstrip()
    if shortened and len(shortened) < len(plain):
        shortened = f"{shortened}…"
    return escape(shortened, quote=False)


def _format_signal(text: str, payload: dict[str, Any]) -> str:
    title = _signal_title(payload) or _first_nonempty_line(text)
    body_lines, tags = _extract_body_and_tags(text, title)
    body = _render_blockquote(body_lines)
    if not tags:
        tags = ["#сетап"]

    parts = []
    if title:
        parts.append(f"<b>{escape(title, quote=False)}</b>")
    if body:
        parts.append(f"<blockquote>{body}</blockquote>")
    parts.append("\n".join(escape(tag, quote=False) for tag in tags))
    return "\n\n".join(part for part in parts if part)


def _format_take1(text: str, payload: dict[str, Any]) -> str:
    title = _signal_title(payload)
    if title:
        title = f"{title} | апдейт"
    else:
        title = "Апдейт по позиции"
    body_lines, tags = _extract_body_and_tags(text, title)
    body = _render_blockquote(body_lines)
    if not tags:
        tags = ["#апдейт"]

    parts = [f"<b>{escape(title, quote=False)}</b>"]
    if body:
        parts.append(f"<blockquote>{body}</blockquote>")
    parts.append("\n".join(escape(tag, quote=False) for tag in tags))
    return "\n\n".join(part for part in parts if part)


def _format_lead_and_body(text: str, default_title: str | None = None) -> str:
    paragraphs = _paragraphs(text)
    if not paragraphs:
        return ""

    lead = paragraphs[0]
    rest = paragraphs[1:]
    if default_title and len(lead) > 80:
        rest = [lead, *rest]
        lead = default_title

    parts = [f"<b>{_highlight_text(lead)}</b>"]
    if rest:
        parts.append(f"<blockquote>{_render_blockquote(rest)}</blockquote>")
    return "\n\n".join(parts)


def _signal_title(payload: dict[str, Any]) -> str | None:
    symbol = str(payload.get("symbol") or "").upper().strip()
    timeframe = str(payload.get("timeframe") or "").strip()
    if not symbol:
        return None
    if symbol.endswith("USDT"):
        symbol = f"{symbol[:-4]}/USDT"
    return f"{symbol} | {timeframe}" if timeframe else symbol


def _extract_body_and_tags(text: str, title: str | None) -> tuple[list[str], list[str]]:
    lines = [line.strip() for line in text.splitlines()]
    body_lines: list[str] = []
    tags: list[str] = []

    for line in lines:
        if not line:
            body_lines.append("")
            continue
        if TAG_RE.match(line):
            tags.append(line)
            continue
        body_lines.append(line)

    while body_lines and not body_lines[0]:
        body_lines.pop(0)
    while body_lines and not body_lines[-1]:
        body_lines.pop()

    if body_lines and _looks_like_title(body_lines[0], title):
        body_lines = body_lines[1:]
        while body_lines and not body_lines[0]:
            body_lines.pop(0)

    return body_lines, tags


def _looks_like_title(line: str, expected_title: str | None) -> bool:
    normalized = _normalize_token(line)
    if expected_title and normalized == _normalize_token(expected_title):
        return True
    return "|" in line and len(line) <= 32


def _render_blockquote(items: list[str]) -> str:
    paragraphs = _paragraphs("\n".join(items))
    return "\n".join(_highlight_text(paragraph) for paragraph in paragraphs)


def _paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    chunk: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if chunk:
                paragraphs.append(" ".join(chunk))
                chunk = []
            continue
        chunk.append(stripped)
    if chunk:
        paragraphs.append(" ".join(chunk))
    return paragraphs


def _highlight_text(text: str) -> str:
    highlighted = escape(text, quote=False)
    for pattern in (PERCENT_RE, PRICE_RANGE_RE, KEYWORD_RE):
        highlighted = pattern.sub(lambda match: f"<b>{match.group(1)}</b>", highlighted)
    return highlighted


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _normalize_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", value.upper())


def _strip_tags(value: str) -> str:
    return re.sub(r"<[^>]+>", "", value)
