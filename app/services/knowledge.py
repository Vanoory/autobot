from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from app.database import Database


TOKEN_RE = re.compile(r"[0-9a-zA-Zа-яА-ЯёЁ_#@./+-]+", re.UNICODE)


def normalize_tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


class KnowledgeBase:
    def __init__(self, database: Database, storage_dir: Path) -> None:
        self.database = database
        self.storage_dir = storage_dir / "knowledge"

    def extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        if suffix == ".docx":
            doc = Document(str(file_path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        raise ValueError("Unsupported file format. Use txt, md, pdf or docx.")

    async def store_document(self, filename: str, binary: bytes) -> tuple[Path, int]:
        target = self.storage_dir / filename
        stem = target.stem
        counter = 1
        while target.exists():
            target = self.storage_dir / f"{stem}_{counter}{target.suffix}"
            counter += 1
        target.write_bytes(binary)
        text = self.extract_text(target).strip()
        if not text:
            raise ValueError("Не получилось вытащить текст из файла.")
        await self.database.add_knowledge_file(target.name, str(target), text)
        return target, len(text)

    async def retrieve(self, query: str, limit: int = 4) -> list[str]:
        docs = await self.database.knowledge_documents()
        if not docs:
            return []

        query_tokens = Counter(normalize_tokens(query))
        scored: list[tuple[int, str]] = []
        for filename, content in docs:
            chunks = self._split_chunks(content)
            for chunk in chunks:
                score = self._score_chunk(chunk, query_tokens)
                if score > 0:
                    scored.append((score, f"[{filename}]\n{chunk.strip()}"))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in scored[:limit]]

    def _split_chunks(self, text: str, max_len: int = 1400) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n{2,}", text) if part.strip()]
        chunks: list[str] = []
        current = ""
        for part in paragraphs:
            candidate = f"{current}\n\n{part}".strip()
            if len(candidate) <= max_len:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                if len(part) <= max_len:
                    current = part
                else:
                    for index in range(0, len(part), max_len):
                        chunks.append(part[index : index + max_len])
                    current = ""
        if current:
            chunks.append(current)
        return chunks or [text[:max_len]]

    def _score_chunk(self, chunk: str, query_tokens: Counter[str]) -> int:
        chunk_tokens = Counter(normalize_tokens(chunk))
        score = 0
        for token, count in query_tokens.items():
            score += min(count, chunk_tokens[token]) * 3
        score += min(len(chunk) // 300, 3)
        return score
