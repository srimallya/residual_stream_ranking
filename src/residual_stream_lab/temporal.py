from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from residual_stream_lab.checkpointing import Checkpoint, extract_terms


def overlap_weight(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if shared == 0:
        return 0.0
    return shared / len(left | right)


def build_temporal_adjacency(checkpoints: list[Checkpoint]) -> np.ndarray:
    count = len(checkpoints)
    adjacency = np.zeros((count, count), dtype=np.float32)
    for i, left in enumerate(checkpoints):
        for j, right in enumerate(checkpoints):
            if i == j:
                continue
            lexical = overlap_weight(left.terms, right.terms)
            if lexical == 0.0:
                continue
            temporal = 1.0 / (abs(left.window.index - right.window.index) + 1.0)
            adjacency[i, j] = lexical * temporal
    return adjacency


def personalized_pagerank(
    adjacency: np.ndarray,
    seeds: np.ndarray,
    alpha: float = 0.85,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> np.ndarray:
    if adjacency.size == 0:
        return seeds

    transition = adjacency.copy()
    row_sums = transition.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    transition = transition / row_sums

    seed_sum = seeds.sum()
    if seed_sum == 0.0:
        seeds = np.full_like(seeds, 1.0 / len(seeds))
    else:
        seeds = seeds / seed_sum

    scores = seeds.copy()
    for _ in range(max_iter):
        updated = alpha * transition.T.dot(scores) + (1.0 - alpha) * seeds
        if np.linalg.norm(updated - scores, ord=1) < tol:
            scores = updated
            break
        scores = updated
    return scores


def rerank_checkpoints(
    checkpoints: list[Checkpoint],
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int,
    exclude_window_ids: set[int],
    recent_windows: Iterable[Checkpoint],
    pool_factor: int = 4,
) -> list[Checkpoint]:
    eligible = [
        checkpoint for checkpoint in checkpoints if checkpoint.window.index not in exclude_window_ids
    ]
    if not eligible:
        return []

    semantic_scores = np.asarray(
        [checkpoint.score(query_embedding) for checkpoint in eligible],
        dtype=np.float32,
    )
    pool_size = min(len(eligible), max(top_k, top_k * pool_factor))
    candidate_order = np.argsort(semantic_scores)[::-1][:pool_size]
    candidates = [eligible[index] for index in candidate_order]
    candidate_semantic = semantic_scores[candidate_order]

    adjacency = build_temporal_adjacency(candidates)
    query_terms = extract_terms(query_text)
    recent_terms = frozenset().union(*(checkpoint.terms for checkpoint in recent_windows))
    query_seed = np.asarray(
        [overlap_weight(candidate.terms, query_terms) for candidate in candidates],
        dtype=np.float32,
    )
    recent_seed = np.asarray(
        [overlap_weight(candidate.terms, recent_terms) for candidate in candidates],
        dtype=np.float32,
    )

    seeds = 0.35 * candidate_semantic + 0.35 * query_seed + 0.30 * recent_seed
    graph_scores = personalized_pagerank(adjacency, seeds)
    final_scores = (
        0.30 * candidate_semantic
        + 0.25 * graph_scores
        + 0.25 * query_seed
        + 0.20 * recent_seed
    )
    final_order = np.argsort(final_scores)[::-1][:top_k]
    return [candidates[index] for index in final_order]
