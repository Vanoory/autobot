from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class Draft:
    id: int
    kind: str
    text: str
    status: str
    image_path: str | None
    payload: dict[str, Any]
    created_at: str


@dataclass(slots=True)
class TrackedSignal:
    id: int
    symbol: str
    side: str
    entry: float
    stop: float
    take1: float
    take2: float
    status: str
    target1_announced: bool
    payload: dict[str, Any]


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path

    async def init(self) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS drafts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    image_path TEXT,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS published_posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    text TEXT NOT NULL,
                    image_path TEXT,
                    channel_message_id INTEGER,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    published_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tracked_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry REAL NOT NULL,
                    stop REAL NOT NULL,
                    take1 REAL NOT NULL,
                    take2 REAL NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    target1_announced INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    stored_path TEXT NOT NULL,
                    content_text TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            await db.commit()

    async def create_draft(
        self,
        kind: str,
        text: str,
        image_path: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> Draft:
        now = utc_now()
        payload_json = json.dumps(payload or {}, ensure_ascii=False)
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                """
                INSERT INTO drafts (kind, text, status, image_path, payload_json, created_at, updated_at)
                VALUES (?, ?, 'pending', ?, ?, ?, ?)
                """,
                (kind, text, image_path, payload_json, now, now),
            )
            await db.commit()
            draft_id = cursor.lastrowid
        return Draft(draft_id, kind, text, "pending", image_path, payload or {}, now)

    async def get_draft(self, draft_id: int) -> Draft | None:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM drafts WHERE id = ?", (draft_id,))
            row = await cursor.fetchone()
        if row is None:
            return None
        return Draft(
            id=row["id"],
            kind=row["kind"],
            text=row["text"],
            status=row["status"],
            image_path=row["image_path"],
            payload=json.loads(row["payload_json"]),
            created_at=row["created_at"],
        )

    async def list_pending_drafts(self) -> list[Draft]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT * FROM drafts WHERE status = 'pending' ORDER BY created_at DESC LIMIT 10"
            )
        return [
            Draft(
                id=row["id"],
                kind=row["kind"],
                text=row["text"],
                status=row["status"],
                image_path=row["image_path"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    async def pending_signal_symbols(self) -> set[str]:
        async with aiosqlite.connect(self.path) as db:
            rows = await db.execute_fetchall(
                """
                SELECT payload_json
                FROM drafts
                WHERE status = 'pending' AND kind = 'signal'
                ORDER BY created_at DESC
                """
            )
        symbols: set[str] = set()
        for row in rows:
            try:
                payload = json.loads(row[0] or "{}")
            except json.JSONDecodeError:
                continue
            symbol = str(payload.get("symbol") or "").strip().upper()
            if symbol:
                symbols.add(symbol)
        return symbols

    async def update_draft(
        self,
        draft_id: int,
        *,
        text: str | None = None,
        status: str | None = None,
        image_path: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        assignments: list[str] = ["updated_at = ?"]
        values: list[Any] = [utc_now()]
        if text is not None:
            assignments.append("text = ?")
            values.append(text)
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if image_path is not None:
            assignments.append("image_path = ?")
            values.append(image_path)
        if payload is not None:
            assignments.append("payload_json = ?")
            values.append(json.dumps(payload, ensure_ascii=False))
        values.append(draft_id)
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                f"UPDATE drafts SET {', '.join(assignments)} WHERE id = ?",
                values,
            )
            await db.commit()

    async def add_published_post(
        self,
        kind: str,
        text: str,
        image_path: str | None,
        channel_message_id: int | None,
        payload: dict[str, Any] | None,
    ) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO published_posts (kind, text, image_path, channel_message_id, payload_json, published_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    text,
                    image_path,
                    channel_message_id,
                    json.dumps(payload or {}, ensure_ascii=False),
                    utc_now(),
                ),
            )
            await db.commit()

    async def recent_published_texts(self, limit: int = 5) -> list[str]:
        async with aiosqlite.connect(self.path) as db:
            rows = await db.execute_fetchall(
                "SELECT text FROM published_posts ORDER BY published_at DESC LIMIT ?",
                (limit,),
            )
        return [row[0] for row in rows]

    async def add_knowledge_file(self, filename: str, stored_path: str, content_text: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO knowledge_files (filename, stored_path, content_text, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (filename, stored_path, content_text, utc_now()),
            )
            await db.commit()

    async def knowledge_documents(self) -> list[tuple[str, str]]:
        async with aiosqlite.connect(self.path) as db:
            rows = await db.execute_fetchall(
                "SELECT filename, content_text FROM knowledge_files ORDER BY created_at DESC"
            )
        return [(row[0], row[1]) for row in rows]

    async def create_tracked_signal(self, payload: dict[str, Any]) -> None:
        now = utc_now()
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO tracked_signals (
                    symbol, side, entry, stop, take1, take2, status,
                    target1_announced, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'open', 0, ?, ?, ?)
                """,
                (
                    payload["symbol"],
                    payload["side"],
                    payload["entry"],
                    payload["stop"],
                    payload["take1"],
                    payload["take2"],
                    json.dumps(payload, ensure_ascii=False),
                    now,
                    now,
                ),
            )
            await db.commit()

    async def open_signals(self) -> list[TrackedSignal]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            rows = await db.execute_fetchall(
                "SELECT * FROM tracked_signals WHERE status = 'open' ORDER BY created_at DESC"
            )
        return [
            TrackedSignal(
                id=row["id"],
                symbol=row["symbol"],
                side=row["side"],
                entry=row["entry"],
                stop=row["stop"],
                take1=row["take1"],
                take2=row["take2"],
                status=row["status"],
                target1_announced=bool(row["target1_announced"]),
                payload=json.loads(row["payload_json"]),
            )
            for row in rows
        ]

    async def signal_count(self, status: str = "open") -> int:
        async with aiosqlite.connect(self.path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM tracked_signals WHERE status = ?",
                (status,),
            )
            row = await cursor.fetchone()
        return int(row[0])

    async def signal_symbols(self, status: str = "open") -> set[str]:
        async with aiosqlite.connect(self.path) as db:
            rows = await db.execute_fetchall(
                "SELECT symbol FROM tracked_signals WHERE status = ?",
                (status,),
            )
        return {str(row[0]).strip().upper() for row in rows if row[0]}

    async def mark_signal_target1_announced(self, signal_id: int) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                UPDATE tracked_signals
                SET target1_announced = 1, updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), signal_id),
            )
            await db.commit()

    async def close_signal(self, signal_id: int, status: str) -> None:
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                UPDATE tracked_signals
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, utc_now(), signal_id),
            )
            await db.commit()
