from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from residual_stream_lab.checkpointing import Checkpoint, extract_anchors, extract_terms


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
            lexical = overlap_weight(left.semantic.terms, right.semantic.terms)
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


def local_refinement_scores(
    candidates: list[Checkpoint],
    base_scores: np.ndarray,
    query_seed: np.ndarray,
    anchor_seed: np.ndarray,
    *,
    top_k: int,
) -> np.ndarray:
    refined = base_scores.copy()
    if not candidates:
        return refined

    local_count = min(len(candidates), max(top_k + 1, min(6, len(candidates))))
    stage2_order = np.argsort(base_scores)[::-1]
    local_indices = stage2_order[:local_count]
    pivot_indices = stage2_order[: min(2, len(stage2_order))]
    pivot_terms = frozenset().union(*(candidates[index].semantic.terms for index in pivot_indices))
    pivot_anchors = frozenset().union(*(candidates[index].semantic.anchors for index in pivot_indices))

    local_bonus = np.zeros(len(candidates), dtype=np.float32)
    for index in local_indices:
        candidate = candidates[index]
        neighborhood = max(
            1.0 / (abs(candidate.window.index - candidates[pivot].window.index) + 1.0)
            for pivot in pivot_indices
        )
        pivot_term_overlap = overlap_weight(candidate.semantic.terms, pivot_terms)
        pivot_anchor_overlap = overlap_weight(candidate.semantic.anchors, pivot_anchors)
        local_bonus[index] = (
            0.45 * anchor_seed[index]
            + 0.20 * query_seed[index]
            + 0.20 * neighborhood
            + 0.10 * pivot_anchor_overlap
            + 0.05 * pivot_term_overlap
        )

    refined[local_indices] = refined[local_indices] + local_bonus[local_indices]
    return refined


def rerank_checkpoints(
    checkpoints: list[Checkpoint],
    query_embedding: np.ndarray,
    query_text: str,
    top_k: int,
    exclude_window_ids: set[int],
    recent_windows: Iterable[Checkpoint],
    strategy: str = "hybrid",
    pool_factor: int = 4,
) -> list[Checkpoint]:
    eligible = [
        checkpoint for checkpoint in checkpoints if checkpoint.window.index not in exclude_window_ids
    ]
    if not eligible:
        return []
    recent_list = list(recent_windows)

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
    query_anchors = extract_anchors(query_text)
    recent_terms = frozenset().union(*(checkpoint.semantic.terms for checkpoint in recent_windows))
    query_seed = np.asarray(
        [overlap_weight(candidate.semantic.terms, query_terms) for candidate in candidates],
        dtype=np.float32,
    )
    anchor_seed = np.asarray(
        [overlap_weight(candidate.semantic.anchors, query_anchors) for candidate in candidates],
        dtype=np.float32,
    )
    recent_seed = np.asarray(
        [overlap_weight(candidate.semantic.terms, recent_terms) for candidate in candidates],
        dtype=np.float32,
    )
    adjacency_seed = np.asarray(
        [
            max(
                (1.0 / (abs(candidate.window.index - recent.window.index) + 1.0))
                for recent in recent_list
            ) if recent_list else 0.0
            for candidate in candidates
        ],
        dtype=np.float32,
    )

    seeds = 0.55 * candidate_semantic + 0.25 * query_seed + 0.20 * anchor_seed
    graph_scores = personalized_pagerank(adjacency, seeds)
    if strategy == "semantic":
        final_scores = candidate_semantic
    elif strategy == "temporal":
        final_scores = (
            0.65 * candidate_semantic
            + 0.15 * graph_scores
            + 0.10 * recent_seed
            + 0.10 * anchor_seed
        )
    elif strategy == "adjacency":
        final_scores = (
            0.65 * candidate_semantic
            + 0.15 * adjacency_seed
            + 0.10 * query_seed
            + 0.10 * anchor_seed
        )
    elif strategy == "hybrid":
        final_scores = (
            0.55 * candidate_semantic
            + 0.20 * anchor_seed
            + 0.10 * graph_scores
            + 0.10 * adjacency_seed
            + 0.05 * query_seed
        )
    elif strategy == "staged":
        temporal_scores = (
            0.60 * candidate_semantic
            + 0.15 * graph_scores
            + 0.10 * recent_seed
            + 0.10 * anchor_seed
            + 0.05 * query_seed
        )
        final_scores = local_refinement_scores(
            candidates,
            temporal_scores,
            query_seed,
            anchor_seed,
            top_k=top_k,
        )
    else:
        raise ValueError(f"Unsupported rerank strategy: {strategy}")
    final_order = np.argsort(final_scores)[::-1][:top_k]
    return [candidates[index] for index in final_order]
