from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from residual_stream_lab.trace import (
    TraceCaptureResult,
    TraceMetadata,
    TraceProvider,
    TraceCheckpointPayload,
    build_trace_payload_from_states,
)


@dataclass(slots=True)
class HFTraceSession:
    input_ids: list[int]
    tokens: list[str]
    layer_states: dict[int, np.ndarray]

    @property
    def token_count(self) -> int:
        return len(self.input_ids)

    @property
    def max_layer(self) -> int:
        return max(self.layer_states)


class HFTraceRunner:
    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        dtype: str = "auto",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(
                "HF trace support requires 'torch' and 'transformers'. "
                "Install them before using the trace backend."
            ) from exc

        self._torch = torch
        self._model_name_or_path = model_name_or_path
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        torch_dtype = self._resolve_torch_dtype(dtype, torch)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
        )
        self.model.to(device)
        self.model.eval()
        self.backend = f"transformers/{type(self.model).__name__}"

    def _resolve_torch_dtype(self, dtype: str, torch_module: object) -> object | None:
        if dtype == "auto":
            return None
        mapping = {
            "float32": torch_module.float32,
            "float16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]

    @property
    def model_name_or_path(self) -> str:
        return self._model_name_or_path

    def trace_text(self, text: str) -> HFTraceSession:
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self._torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True, use_cache=False)

        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        hidden_states = output.hidden_states
        layer_states: dict[int, np.ndarray] = {}
        for index, hidden_state in enumerate(hidden_states):
            logical_layer = index - 1
            values = hidden_state[0].detach().cpu().to(self._torch.float32).numpy()
            layer_states[logical_layer] = values
        return HFTraceSession(
            input_ids=input_ids,
            tokens=tokens,
            layer_states=layer_states,
        )

    def capture_trace(
        self,
        *,
        text: str,
        layer_cutoff_b: int,
        delta_layers: list[int],
        token_index: int = -1,
    ) -> TraceCaptureResult:
        session = self.trace_text(text)
        resolved_token_index = token_index if token_index >= 0 else session.token_count + token_index
        if resolved_token_index < 0 or resolved_token_index >= session.token_count:
            raise ValueError(
                f"Token index {token_index} resolves to {resolved_token_index}, "
                f"but the prompt has {session.token_count} tokens."
            )

        token_layer_states = {
            layer: values[resolved_token_index].copy()
            for layer, values in session.layer_states.items()
        }
        payload = build_trace_payload_from_states(
            layer_states=token_layer_states,
            layer_cutoff_b=layer_cutoff_b,
            token_index=resolved_token_index,
            delta_layers=delta_layers,
            metadata=TraceMetadata(
                shape=tuple(token_layer_states[layer_cutoff_b].shape),
                dtype="float32",
                backend=self.backend,
                provenance=self._model_name_or_path,
            ),
        )
        observed = {
            layer: token_layer_states[layer].copy()
            for layer in sorted({layer_cutoff_b, *delta_layers})
        }
        return TraceCaptureResult(
            payload=payload,
            observed_states=observed,
            token_count=session.token_count,
        )


class HFTraceProvider(TraceProvider):
    def __init__(
        self,
        runner: HFTraceRunner,
        *,
        delta_layers: list[int],
    ) -> None:
        self.runner = runner
        self.delta_layers = list(delta_layers)

    def capture_boundary_residual(
        self,
        *,
        boundary_token_index: int,
        layer_cutoff_b: int | None,
        window_text: str,
    ) -> TraceCheckpointPayload:
        if layer_cutoff_b is None:
            raise ValueError("HFTraceProvider requires a concrete layer cutoff.")
        capture = self.runner.capture_trace(
            text=window_text,
            layer_cutoff_b=layer_cutoff_b,
            delta_layers=self.delta_layers,
            token_index=boundary_token_index,
        )
        return capture.payload

    def capture_layer_delta(
        self,
        *,
        payload: TraceCheckpointPayload,
        layer_index: int,
        delta: np.ndarray,
    ) -> TraceCheckpointPayload:
        payload.late_layer_deltas.append(np.asarray(delta, dtype=np.float32))
        payload.delta_layers.append(layer_index)
        return payload

    def reconstruct_boundary_state(
        self,
        payload: TraceCheckpointPayload,
    ) -> np.ndarray:
        from residual_stream_lab.trace import reconstruct_boundary_state

        return reconstruct_boundary_state(payload)
