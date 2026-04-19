from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class TraceMetadata:
    shape: tuple[int, ...]
    dtype: str
    quantization: str = "none"
    backend: str = "unknown"
    provenance: str = "unspecified"


@dataclass(slots=True)
class TraceCheckpointPayload:
    layer_cutoff_b: int | None = None
    boundary_token_index: int | None = None
    boundary_residual: np.ndarray | None = None
    late_layer_deltas: list[np.ndarray] = field(default_factory=list)
    delta_layers: list[int] = field(default_factory=list)
    metadata: TraceMetadata | None = None
    exact_trace: bool = False
    approximate_trace: bool = False

    @property
    def is_exact_reconstructable(self) -> bool:
        if not self.exact_trace:
            return False
        if self.boundary_residual is None or not self.late_layer_deltas:
            return False
        return len(self.late_layer_deltas) == len(self.delta_layers)


@dataclass(slots=True)
class ReconstructionResult:
    reconstructed: np.ndarray
    l2_error: float
    cosine_similarity: float
    exact_trace: bool


@dataclass(slots=True)
class TraceCaptureResult:
    payload: TraceCheckpointPayload
    observed_states: dict[int, np.ndarray]
    token_count: int


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def reconstruct_boundary_state(payload: TraceCheckpointPayload) -> np.ndarray:
    if payload.boundary_residual is None:
        raise ValueError("Trace payload is missing the boundary residual.")
    if len(payload.late_layer_deltas) != len(payload.delta_layers):
        raise ValueError("Trace payload has misaligned delta tensors and layer ids.")

    reconstructed = np.asarray(payload.boundary_residual, dtype=np.float32).copy()
    for delta in payload.late_layer_deltas:
        reconstructed += np.asarray(delta, dtype=np.float32)
    return reconstructed


def verify_reconstruction(
    payload: TraceCheckpointPayload,
    observed_boundary_state: np.ndarray,
) -> ReconstructionResult:
    reconstructed = reconstruct_boundary_state(payload)
    observed = np.asarray(observed_boundary_state, dtype=np.float32)
    delta = reconstructed - observed
    return ReconstructionResult(
        reconstructed=reconstructed,
        l2_error=float(np.linalg.norm(delta)),
        cosine_similarity=cosine_similarity(reconstructed, observed),
        exact_trace=payload.exact_trace,
    )


def build_trace_payload_from_states(
    *,
    layer_states: dict[int, np.ndarray],
    layer_cutoff_b: int,
    token_index: int,
    delta_layers: list[int],
    metadata: TraceMetadata | None = None,
) -> TraceCheckpointPayload:
    if layer_cutoff_b not in layer_states:
        raise ValueError(f"Missing boundary state for layer {layer_cutoff_b}.")

    missing_layers = [layer for layer in delta_layers if layer not in layer_states or (layer - 1) not in layer_states]
    if missing_layers:
        raise ValueError(f"Missing required layer states for deltas: {missing_layers}.")

    boundary_residual = np.asarray(layer_states[layer_cutoff_b], dtype=np.float32).copy()
    late_layer_deltas = [
        np.asarray(layer_states[layer], dtype=np.float32) - np.asarray(layer_states[layer - 1], dtype=np.float32)
        for layer in delta_layers
    ]
    trace_metadata = metadata or TraceMetadata(
        shape=tuple(boundary_residual.shape),
        dtype=str(boundary_residual.dtype),
        provenance="observed-layer-states",
    )
    return TraceCheckpointPayload(
        layer_cutoff_b=layer_cutoff_b,
        boundary_token_index=token_index,
        boundary_residual=boundary_residual,
        late_layer_deltas=late_layer_deltas,
        delta_layers=list(delta_layers),
        metadata=trace_metadata,
        exact_trace=True,
        approximate_trace=False,
    )


def capture_from_layer_states(
    *,
    layer_states: dict[int, np.ndarray],
    layer_cutoff_b: int,
    token_index: int,
    delta_layers: list[int],
    metadata: TraceMetadata | None = None,
) -> TraceCaptureResult:
    payload = build_trace_payload_from_states(
        layer_states=layer_states,
        layer_cutoff_b=layer_cutoff_b,
        token_index=token_index,
        delta_layers=delta_layers,
        metadata=metadata,
    )
    observed = {
        layer: np.asarray(layer_states[layer], dtype=np.float32).copy()
        for layer in sorted({layer_cutoff_b, *delta_layers})
    }
    return TraceCaptureResult(
        payload=payload,
        observed_states=observed,
        token_count=max(token_index + 1, 0),
    )


class TraceProvider(ABC):
    @abstractmethod
    def capture_boundary_residual(
        self,
        *,
        boundary_token_index: int,
        layer_cutoff_b: int | None,
        window_text: str,
    ) -> TraceCheckpointPayload:
        raise NotImplementedError

    @abstractmethod
    def capture_layer_delta(
        self,
        *,
        payload: TraceCheckpointPayload,
        layer_index: int,
        delta: np.ndarray,
    ) -> TraceCheckpointPayload:
        raise NotImplementedError

    @abstractmethod
    def reconstruct_boundary_state(
        self,
        payload: TraceCheckpointPayload,
    ) -> np.ndarray:
        raise NotImplementedError


class NullTraceProvider(TraceProvider):
    def __init__(self, backend: str = "llama-cpp-python") -> None:
        self.backend = backend

    def capture_boundary_residual(
        self,
        *,
        boundary_token_index: int,
        layer_cutoff_b: int | None,
        window_text: str,
    ) -> TraceCheckpointPayload:
        return TraceCheckpointPayload(
            layer_cutoff_b=layer_cutoff_b,
            boundary_token_index=boundary_token_index,
            metadata=TraceMetadata(
                shape=(),
                dtype="unavailable",
                backend=self.backend,
                provenance="null-trace-provider",
            ),
            exact_trace=False,
            approximate_trace=False,
        )

    def capture_layer_delta(
        self,
        *,
        payload: TraceCheckpointPayload,
        layer_index: int,
        delta: np.ndarray,
    ) -> TraceCheckpointPayload:
        return payload

    def reconstruct_boundary_state(
        self,
        payload: TraceCheckpointPayload,
    ) -> np.ndarray:
        raise NotImplementedError(
            "NullTraceProvider cannot reconstruct state because this backend does not expose trace tensors."
        )
