from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

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
    attention_mask: list[int]
    layer_states: dict[int, np.ndarray]

    @property
    def token_count(self) -> int:
        return len(self.input_ids)

    @property
    def max_layer(self) -> int:
        return max(self.layer_states)


class HFTraceRunner:
    SNAPSHOT_ALLOW_PATTERNS = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "*.model",
        "*.safetensors",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        dtype: str = "auto",
        model_root: str | None = None,
    ) -> None:
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers.models.gpt2.modeling_gpt2 import create_causal_mask
        except ImportError as exc:  # pragma: no cover - dependency-gated
            raise RuntimeError(
                "HF trace support requires 'torch', 'transformers', and 'huggingface_hub'. "
                "Install them before using the trace backend."
            ) from exc

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self.model_root = Path(model_root or "models").expanduser().resolve()
        self.cache_dir = self.model_root / ".cache" / "huggingface"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_download = snapshot_download
        self._create_causal_mask = create_causal_mask
        resolved_model_path = self._resolve_model_path(model_name_or_path)
        self._model_name_or_path = str(resolved_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)
        torch_dtype = self._resolve_torch_dtype(dtype, torch)
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_name_or_path,
            dtype=torch_dtype,
        )
        self.model.to(device)
        self.model.eval()
        self.backend = f"transformers/{type(self.model).__name__}"
        self.model_dtype = next(self.model.parameters()).dtype

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

    def _local_model_dir(self, repo_id: str) -> Path:
        slug = re.sub(r"[^A-Za-z0-9._-]+", "--", repo_id.strip("/"))
        return self.model_root / slug

    def _is_complete_local_model_dir(self, path: Path) -> bool:
        if not path.exists():
            return False
        has_config = (path / "config.json").exists()
        has_weights = any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
        tokenizer_candidates = [
            path / "tokenizer.json",
            path / "tokenizer_config.json",
            path / "vocab.json",
            path / "merges.txt",
        ]
        has_tokenizer = any(candidate.exists() for candidate in tokenizer_candidates)
        return has_config and has_weights and has_tokenizer

    def _resolve_model_path(self, model_name_or_path: str) -> Path:
        candidate = Path(model_name_or_path).expanduser()
        if candidate.exists():
            return candidate.resolve()

        local_dir = self._local_model_dir(model_name_or_path)
        if not self._is_complete_local_model_dir(local_dir):
            self._snapshot_download(
                repo_id=model_name_or_path,
                local_dir=local_dir,
                cache_dir=self.cache_dir,
                allow_patterns=self.SNAPSHOT_ALLOW_PATTERNS,
            )
        return local_dir.resolve()

    def trace_text(self, text: str) -> HFTraceSession:
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self._torch.no_grad():
            output = self.model(**encoded, output_hidden_states=True, use_cache=False)

        input_ids = encoded["input_ids"][0].detach().cpu().tolist()
        attention_mask = encoded["attention_mask"][0].detach().cpu().tolist()
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
            attention_mask=attention_mask,
            layer_states=layer_states,
        )

    def _require_gpt2_model(self) -> object:
        if not hasattr(self.model, "transformer") or not hasattr(self.model.transformer, "h"):
            raise NotImplementedError(
                "Phase 2A resume is currently implemented only for GPT-2-class decoder models."
            )
        transformer = self.model.transformer
        if not hasattr(transformer, "ln_f") or not hasattr(self.model, "lm_head"):
            raise NotImplementedError(
                "Resume path requires a GPT-2-style transformer stack with final layer norm and lm_head."
            )
        return transformer

    def _build_causal_attention_mask(
        self,
        hidden_tensor: object,
        attention_mask_2d: object,
        position_ids: object,
    ) -> object:
        return self._create_causal_mask(
            config=self.model.config,
            inputs_embeds=hidden_tensor,
            attention_mask=attention_mask_2d,
            past_key_values=None,
            position_ids=position_ids,
        )

    def predict_from_hidden(
        self,
        *,
        hidden_state: np.ndarray,
        start_layer: int,
        attention_mask: list[int],
        target_token_index: int = -1,
    ) -> np.ndarray:
        transformer = self._require_gpt2_model()
        num_layers = len(transformer.h)
        if start_layer < 0 or start_layer > num_layers:
            raise ValueError(f"start_layer must be between 0 and {num_layers}, got {start_layer}.")

        hidden_tensor = self._torch.tensor(hidden_state, dtype=self.model_dtype, device=self.device).unsqueeze(0)
        mask_tensor = self._torch.tensor([attention_mask], device=self.device)
        position_ids = self._torch.arange(hidden_tensor.shape[1], device=self.device).unsqueeze(0)
        causal_attention_mask = self._build_causal_attention_mask(
            hidden_tensor,
            mask_tensor,
            position_ids,
        )

        with self._torch.no_grad():
            for layer_index in range(start_layer, num_layers):
                block = transformer.h[layer_index]
                block_output = block(
                    hidden_tensor,
                    attention_mask=causal_attention_mask,
                    use_cache=False,
                    position_ids=position_ids,
                )
                hidden_tensor = block_output[0] if isinstance(block_output, tuple) else block_output
            hidden_tensor = transformer.ln_f(hidden_tensor)
            logits = self.model.lm_head(hidden_tensor)

        resolved_target = target_token_index if target_token_index >= 0 else logits.shape[1] + target_token_index
        if resolved_target < 0 or resolved_target >= logits.shape[1]:
            raise ValueError(
                f"Target token index {target_token_index} resolves to {resolved_target}, "
                f"but sequence length is {logits.shape[1]}."
            )
        return logits[0, resolved_target].detach().cpu().to(self._torch.float32).numpy()

    def compare_resumed_logits(
        self,
        *,
        text: str,
        boundary_layer: int,
        target_token_index: int = -1,
    ) -> dict[str, float | int]:
        session = self.trace_text(text)
        transformer = self._require_gpt2_model()
        num_layers = len(transformer.h)
        if boundary_layer < -1 or boundary_layer >= num_layers:
            raise ValueError(
                f"boundary_layer must be between -1 and {num_layers - 1}, got {boundary_layer}."
            )

        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self._torch.no_grad():
            output = self.model(**encoded, output_hidden_states=False, use_cache=False)
        direct_logits = output.logits[0].detach().cpu().to(self._torch.float32).numpy()

        resumed_logits = self.predict_from_hidden(
            hidden_state=session.layer_states[boundary_layer],
            start_layer=boundary_layer + 1,
            attention_mask=session.attention_mask,
            target_token_index=target_token_index,
        )

        resolved_target = target_token_index if target_token_index >= 0 else session.token_count + target_token_index
        direct_target = direct_logits[resolved_target]
        delta = resumed_logits - direct_target
        resumed_norm = float(np.linalg.norm(resumed_logits))
        direct_norm = float(np.linalg.norm(direct_target))
        return {
            "boundary_layer": boundary_layer,
            "start_layer": boundary_layer + 1,
            "target_token_index": resolved_target,
            "l2_error": float(np.linalg.norm(delta)),
            "cosine_similarity": float(
                np.dot(resumed_logits, direct_target)
                / (resumed_norm * direct_norm)
            ) if resumed_norm and direct_norm else 0.0,
            "max_abs_diff": float(np.max(np.abs(delta))),
            "vocab_size": int(resumed_logits.shape[0]),
        }

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
