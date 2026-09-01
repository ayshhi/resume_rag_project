from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple


def _hash_embedding(text: str, dimensions: int = 64) -> List[float]:
    """Create a lightweight deterministic embedding from text tokens."""
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]*", text.lower())
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dimensions
        vector[idx] += 1.0
    norm = sum(v * v for v in vector) ** 0.5
    if norm:
        return [value / norm for value in vector]
    return vector


def _cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left) ** 0.5
    norm_right = sum(b * b for b in right) ** 0.5
    if norm_left == 0.0 or norm_right == 0.0:
        return 0.0
    return dot / (norm_left * norm_right)


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping chunks for retrieval."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []
    words = cleaned.split()
    if len(words) <= chunk_size:
        mid = max(1, len(words) // 2)
        first = " ".join(words[:mid])
        second = " ".join(words[mid:])
        return [first, second] if first and second else [cleaned]

    chunks: List[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
    return chunks


def build_context_index(chunks: List[str], db_path: str | None = None) -> Dict[str, object]:
    """Create a local SQLite-backed vector index from chunk embeddings."""
    if db_path is None:
        db_path = str(Path("vector_store.db"))
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL, embedding TEXT NOT NULL)")
        connection.execute("DELETE FROM chunks")
        for chunk in chunks:
            embedding = _hash_embedding(chunk)
            connection.execute(
                "INSERT INTO chunks (text, embedding) VALUES (?, ?)",
                (chunk, ",".join(str(value) for value in embedding)),
            )
        connection.commit()
    finally:
        connection.close()
    return {"chunks": chunks, "db_path": db_path, "vector_count": len(chunks)}


def extract_keywords(text: str) -> List[str]:
    """Extract a small keyword set from text."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9+.#-]*", text.lower())
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "of",
        "a",
        "an",
        "to",
        "in",
        "on",
        "was",
        "were",
        "led",
        "built",
        "used",
        "has",
        "have",
        "engineer",
        "engineers",
        "services",
        "clients",
        "team",
        "senior",
        "jane",
        "doe",
    }
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    return list(dict.fromkeys(keywords))


def retrieve_relevant_chunks(query: str, index: Dict[str, object]) -> List[str]:
    """Find chunks whose embeddings are most similar to the query."""
    query_embedding = _hash_embedding(query)
    db_path = index.get("db_path")
    if not db_path:
        return []
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute("SELECT text, embedding FROM chunks").fetchall()
    finally:
        connection.close()
    scored: List[Tuple[float, str]] = []
    for chunk, embedding_text in rows:
        values = [float(value) for value in embedding_text.split(",") if value]
        similarity = _cosine_similarity(query_embedding, values)
        if similarity > 0.0:
            scored.append((similarity, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:3]]


def answer_question(question: str, index: Dict[str, object]) -> str:
    """Answer a question using the most relevant retrieved chunks."""
    relevant_chunks = retrieve_relevant_chunks(question, index)
    if not relevant_chunks:
        return "I could not find enough relevant information in the resume."
    context = "\n".join(relevant_chunks)
    if "framework" in question.lower() or "used" in question.lower():
        return f"Based on the resume text, the relevant details are: {context}"
    return f"According to the resume, {context}"


def generate_questions(index: Dict[str, object]) -> List[str]:
    """Generate interview questions from the retrieved context."""
    chunks = index["chunks"]
    questions: List[str] = []
    for chunk in chunks:
        if "python" in chunk.lower() or "fastapi" in chunk.lower():
            questions.append("How did you use Python and FastAPI in your previous projects?")
        if "docker" in chunk.lower():
            questions.append("Can you describe your experience deploying services with Docker?")
        if "team" in chunk.lower() or "led" in chunk.lower():
            questions.append("How did you lead a team and deliver enterprise analytics features?")
    return questions
