from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np


STOPWORDS = {
    "the", "and", "with", "from", "into", "that", "this", "what", "when", "where",
    "which", "their", "there", "about", "window", "note", "active", "thread",
    "answer", "color", "only", "marker", "assigned", "belongs", "current",
}


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


@dataclass(slots=True)
class Window:
    index: int
    text: str
    boundary_text: str


@dataclass(slots=True)
class Checkpoint:
    window: Window
    window_embedding: np.ndarray
    boundary_embedding: np.ndarray
    terms: frozenset[str]

    def score(self, query_embedding: np.ndarray) -> float:
        return max(
            cosine_similarity(query_embedding, self.window_embedding),
            cosine_similarity(query_embedding, self.boundary_embedding),
        )

    def memory_packet(self) -> str:
        return (
            f"[Checkpoint window {self.window.index}]\n"
            f"Boundary:\n{self.window.boundary_text}\n\n"
            f"Window contents:\n{self.window.text}"
        )


def split_windows(text: str, lines_per_window: int) -> list[Window]:
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        return []

    windows: list[Window] = []
    for start in range(0, len(raw_lines), lines_per_window):
        chunk = raw_lines[start : start + lines_per_window]
        windows.append(
            Window(
                index=len(windows),
                text="\n".join(chunk),
                boundary_text=chunk[-1],
            )
        )
    return windows


def extract_terms(text: str) -> frozenset[str]:
    tokens = {
        token
        for token in re.findall(r"[a-zA-Z]{4,}", text.lower())
        if token not in STOPWORDS
    }
    return frozenset(tokens)


def build_checkpoints(
    windows: Iterable[Window],
    embed_text: callable,
) -> list[Checkpoint]:
    checkpoints: list[Checkpoint] = []
    for window in windows:
        checkpoints.append(
            Checkpoint(
                window=window,
                window_embedding=embed_text(window.text),
                boundary_embedding=embed_text(window.boundary_text),
                terms=extract_terms(window.text),
            )
        )
    return checkpoints


def retrieve_checkpoints(
    checkpoints: Iterable[Checkpoint],
    query_embedding: np.ndarray,
    top_k: int,
    exclude_window_ids: set[int] | None = None,
) -> list[Checkpoint]:
    excluded = exclude_window_ids or set()
    ranked = sorted(
        (checkpoint for checkpoint in checkpoints if checkpoint.window.index not in excluded),
        key=lambda checkpoint: checkpoint.score(query_embedding),
        reverse=True,
    )
    return ranked[:top_k]
