from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from residual_stream_lab.trace import NullTraceProvider, TraceCheckpointPayload, TraceProvider


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
class SemanticCheckpointPayload:
    window_embedding: np.ndarray
    boundary_embedding: np.ndarray
    terms: frozenset[str]
    anchors: frozenset[str]


@dataclass(slots=True)
class Checkpoint:
    window: Window
    semantic: SemanticCheckpointPayload
    trace: TraceCheckpointPayload

    def score(self, query_embedding: np.ndarray) -> float:
        return max(
            cosine_similarity(query_embedding, self.semantic.window_embedding),
            cosine_similarity(query_embedding, self.semantic.boundary_embedding),
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


def extract_anchors(text: str) -> frozenset[str]:
    anchors = {
        token.lower()
        for token in re.findall(r"\b[a-zA-Z]+(?:-[a-zA-Z0-9]+)+\b", text)
    }
    anchors.update(
        token.lower()
        for token in re.findall(r"\b[a-zA-Z]{2,}\d{2,}[a-zA-Z0-9-]*\b", text)
    )
    return frozenset(anchors)


def build_checkpoints(
    windows: Iterable[Window],
    embed_text: callable,
    trace_provider: TraceProvider | None = None,
    layer_cutoff_b: int | None = None,
) -> list[Checkpoint]:
    provider = trace_provider or NullTraceProvider()
    checkpoints: list[Checkpoint] = []
    for window in windows:
        checkpoints.append(
            Checkpoint(
                window=window,
                semantic=SemanticCheckpointPayload(
                    window_embedding=embed_text(window.text),
                    boundary_embedding=embed_text(window.boundary_text),
                    terms=extract_terms(window.text),
                    anchors=extract_anchors(window.text),
                ),
                trace=provider.capture_boundary_residual(
                    boundary_token_index=window.index,
                    layer_cutoff_b=layer_cutoff_b,
                    window_text=window.text,
                ),
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
