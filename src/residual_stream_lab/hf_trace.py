from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
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
            from transformers.cache_utils import DynamicCache
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
        self._dynamic_cache_class = DynamicCache
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
        encoded = self._encode_text(text)
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

    def _encode_text(self, text: str) -> dict[str, object]:
        encoded = self.tokenizer(text, return_tensors="pt")
        return {name: tensor.to(self.device) for name, tensor in encoded.items()}

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
        past_key_values: object | None = None,
    ) -> object:
        return self._create_causal_mask(
            config=self.model.config,
            inputs_embeds=hidden_tensor,
            attention_mask=attention_mask_2d,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

    def _build_position_ids(self, seq_len: int) -> object:
        return self._torch.arange(seq_len, device=self.device).unsqueeze(0)

    def _prepare_gpt2_hidden_inputs(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        past_key_values: object | None = None,
        position_ids: object | None = None,
    ) -> tuple[object, object, object]:
        transformer = self._require_gpt2_model()
        inputs_embeds = transformer.wte(input_ids)
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = self._torch.arange(inputs_embeds.shape[1], device=self.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        position_embeds = transformer.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        hidden_states = transformer.drop(hidden_states)
        causal_attention_mask = self._build_causal_attention_mask(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_values=past_key_values,
        )
        return hidden_states, causal_attention_mask, position_ids

    def _make_dynamic_cache(self) -> object:
        return self._dynamic_cache_class(config=self.model.config)

    def _run_upper_stack(
        self,
        *,
        hidden_tensor: object,
        start_layer: int,
        attention_mask_tensor: object,
        position_ids: object,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[object, object | None]:
        transformer = self._require_gpt2_model()
        causal_attention_mask = self._build_causal_attention_mask(
            hidden_tensor,
            attention_mask_tensor,
            position_ids,
            past_key_values=past_key_values,
        )
        for layer_index in range(start_layer, len(transformer.h)):
            block = transformer.h[layer_index]
            block_output = block(
                hidden_tensor,
                past_key_values=past_key_values,
                attention_mask=causal_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
            )
            hidden_tensor = block_output[0] if isinstance(block_output, tuple) else block_output
        hidden_tensor = transformer.ln_f(hidden_tensor)
        return hidden_tensor, past_key_values

    def capture_boundary_hidden_from_text(
        self,
        *,
        text: str,
        boundary_layer: int,
    ) -> np.ndarray:
        encoded = self._encode_text(text)
        return self.capture_boundary_hidden_from_inputs(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            boundary_layer=boundary_layer,
        )

    def capture_boundary_hidden_from_inputs(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        boundary_layer: int,
    ) -> np.ndarray:
        transformer = self._require_gpt2_model()
        num_layers = len(transformer.h)
        if boundary_layer < -1 or boundary_layer >= num_layers:
            raise ValueError(
                f"boundary_layer must be between -1 and {num_layers - 1}, got {boundary_layer}."
            )

        with self._torch.no_grad():
            hidden_states, causal_attention_mask, position_ids = self._prepare_gpt2_hidden_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            if boundary_layer == -1:
                return hidden_states[0].detach().cpu().to(self._torch.float32).numpy()

            for layer_index in range(0, boundary_layer + 1):
                block = transformer.h[layer_index]
                block_output = block(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    use_cache=False,
                    position_ids=position_ids,
                )
                hidden_states = block_output[0] if isinstance(block_output, tuple) else block_output
            return hidden_states[0].detach().cpu().to(self._torch.float32).numpy()

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
        with self._torch.no_grad():
            hidden_tensor, _ = self._run_upper_stack(
                hidden_tensor=hidden_tensor,
                start_layer=start_layer,
                attention_mask_tensor=mask_tensor,
                position_ids=position_ids,
                use_cache=False,
            )
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

        encoded = self._encode_text(text)
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

    def _direct_next_token_logits_from_inputs(
        self,
        *,
        input_ids: object,
        attention_mask: object,
    ) -> np.ndarray:
        with self._torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            )
        return output.logits[0, -1].detach().cpu().to(self._torch.float32).numpy()

    def _resumed_next_token_logits_from_inputs(
        self,
        *,
        input_ids: object,
        attention_mask: object,
        boundary_layer: int,
    ) -> np.ndarray:
        boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        )
        attention_mask_list = attention_mask[0].detach().cpu().tolist()
        return self.predict_from_hidden(
            hidden_state=boundary_hidden,
            start_layer=boundary_layer + 1,
            attention_mask=attention_mask_list,
            target_token_index=-1,
        )

    def _top_k_tokens(
        self,
        logits: np.ndarray,
        *,
        top_k: int,
    ) -> list[tuple[int, str, float]]:
        count = max(1, min(top_k, logits.shape[0]))
        top_ids = np.argsort(logits)[-count:][::-1]
        results: list[tuple[int, str, float]] = []
        for token_id in top_ids:
            token_text = self.tokenizer.decode([int(token_id)])
            results.append((int(token_id), token_text, float(logits[token_id])))
        return results

    def compare_next_token(
        self,
        *,
        text: str,
        boundary_layer: int,
        top_k: int = 5,
    ) -> dict[str, object]:
        comparison = self.compare_resumed_logits(
            text=text,
            boundary_layer=boundary_layer,
            target_token_index=-1,
        )
        session = self.trace_text(text)
        encoded = self.tokenizer(text, return_tensors="pt")
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        with self._torch.no_grad():
            output = self.model(**encoded, output_hidden_states=False, use_cache=False)
        direct_logits = output.logits[0, -1].detach().cpu().to(self._torch.float32).numpy()
        resumed_logits = self.predict_from_hidden(
            hidden_state=session.layer_states[boundary_layer],
            start_layer=boundary_layer + 1,
            attention_mask=session.attention_mask,
            target_token_index=-1,
        )

        direct_token_id = int(np.argmax(direct_logits))
        resumed_token_id = int(np.argmax(resumed_logits))
        return {
            **comparison,
            "direct_token_id": direct_token_id,
            "resumed_token_id": resumed_token_id,
            "direct_token_text": self.tokenizer.decode([direct_token_id]),
            "resumed_token_text": self.tokenizer.decode([resumed_token_id]),
            "token_match": direct_token_id == resumed_token_id,
            "direct_top_k": self._top_k_tokens(direct_logits, top_k=top_k),
            "resumed_top_k": self._top_k_tokens(resumed_logits, top_k=top_k),
        }

    def compare_greedy_continuation(
        self,
        *,
        text: str,
        boundary_layer: int,
        steps: int,
        top_k: int = 5,
    ) -> dict[str, object]:
        encoded = self._encode_text(text)
        direct_input_ids = encoded["input_ids"]
        direct_attention_mask = encoded["attention_mask"]
        resumed_input_ids = encoded["input_ids"].clone()
        resumed_attention_mask = encoded["attention_mask"].clone()

        per_step: list[dict[str, object]] = []
        first_divergence_step: int | None = None

        for step in range(steps):
            direct_logits = self._direct_next_token_logits_from_inputs(
                input_ids=direct_input_ids,
                attention_mask=direct_attention_mask,
            )
            resumed_logits = self._resumed_next_token_logits_from_inputs(
                input_ids=resumed_input_ids,
                attention_mask=resumed_attention_mask,
                boundary_layer=boundary_layer,
            )

            delta = resumed_logits - direct_logits
            direct_token_id = int(np.argmax(direct_logits))
            resumed_token_id = int(np.argmax(resumed_logits))
            direct_text = self.tokenizer.decode([direct_token_id])
            resumed_text = self.tokenizer.decode([resumed_token_id])
            resumed_norm = float(np.linalg.norm(resumed_logits))
            direct_norm = float(np.linalg.norm(direct_logits))
            token_match = direct_token_id == resumed_token_id
            exact_logits = bool(np.array_equal(resumed_logits, direct_logits))

            step_result = {
                "step": step + 1,
                "l2_error": float(np.linalg.norm(delta)),
                "cosine_similarity": float(
                    np.dot(resumed_logits, direct_logits) / (resumed_norm * direct_norm)
                ) if resumed_norm and direct_norm else 0.0,
                "max_abs_diff": float(np.max(np.abs(delta))),
                "direct_token_id": direct_token_id,
                "direct_token_text": direct_text,
                "resumed_token_id": resumed_token_id,
                "resumed_token_text": resumed_text,
                "token_match": token_match,
                "exact_logits": exact_logits,
                "direct_top_k": self._top_k_tokens(direct_logits, top_k=top_k),
                "resumed_top_k": self._top_k_tokens(resumed_logits, top_k=top_k),
            }
            per_step.append(step_result)

            if not token_match or not exact_logits:
                first_divergence_step = step + 1
                break

            next_token_tensor = self._torch.tensor([[direct_token_id]], device=self.device, dtype=direct_input_ids.dtype)
            next_mask_tensor = self._torch.ones((1, 1), device=self.device, dtype=direct_attention_mask.dtype)
            direct_input_ids = self._torch.cat([direct_input_ids, next_token_tensor], dim=1)
            direct_attention_mask = self._torch.cat([direct_attention_mask, next_mask_tensor], dim=1)
            resumed_input_ids = self._torch.cat([resumed_input_ids, next_token_tensor], dim=1)
            resumed_attention_mask = self._torch.cat([resumed_attention_mask, next_mask_tensor], dim=1)

        generated_ids = direct_input_ids[0].detach().cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids)
        return {
            "boundary_layer": boundary_layer,
            "steps_requested": steps,
            "steps_completed": len(per_step),
            "exact_match": first_divergence_step is None,
            "first_divergence_step": first_divergence_step,
            "generated_text": generated_text,
            "per_step": per_step,
        }

    def _run_upper_stack_trace(
        self,
        *,
        hidden_tensor: object,
        start_layer: int,
        attention_mask_tensor: object,
        position_ids: object,
        past_key_values: object | None = None,
        use_cache: bool = False,
    ) -> tuple[dict[int, np.ndarray], object | None, np.ndarray]:
        transformer = self._require_gpt2_model()
        causal_attention_mask = self._build_causal_attention_mask(
            hidden_tensor,
            attention_mask_tensor,
            position_ids,
            past_key_values=past_key_values,
        )
        per_layer_last_token: dict[int, np.ndarray] = {}
        for layer_index in range(start_layer, len(transformer.h)):
            block = transformer.h[layer_index]
            block_output = block(
                hidden_tensor,
                past_key_values=past_key_values,
                attention_mask=causal_attention_mask,
                use_cache=use_cache,
                position_ids=position_ids,
            )
            hidden_tensor = block_output[0] if isinstance(block_output, tuple) else block_output
            per_layer_last_token[layer_index] = (
                hidden_tensor[0, -1].detach().cpu().to(self._torch.float32).numpy()
            )
        hidden_tensor = transformer.ln_f(hidden_tensor)
        logits = self.model.lm_head(hidden_tensor)[0, -1].detach().cpu().to(self._torch.float32).numpy()
        return per_layer_last_token, past_key_values, logits

    def _cache_bytes(self, cache: object) -> int:
        total = 0
        for layer in getattr(cache, "layers", []):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None:
                total += int(keys.numel() * keys.element_size())
            if values is not None:
                total += int(values.numel() * values.element_size())
        return total

    def compare_greedy_continuation_kv(
        self,
        *,
        text: str,
        boundary_layer: int,
        steps: int,
        top_k: int = 5,
    ) -> dict[str, object]:
        encoded = self._encode_text(text)
        baseline_started = perf_counter()
        baseline = self.compare_greedy_continuation(
            text=text,
            boundary_layer=boundary_layer,
            steps=steps,
            top_k=top_k,
        )
        baseline_ms = (perf_counter() - baseline_started) * 1000.0

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        )
        boundary_tensor = self._torch.tensor(boundary_hidden, dtype=self.model_dtype, device=self.device).unsqueeze(0)
        cache = self._make_dynamic_cache()
        prefill_started = perf_counter()
        with self._torch.no_grad():
            prefill_position_ids = self._build_position_ids(boundary_tensor.shape[1])
            prefill_hidden, cache = self._run_upper_stack(
                hidden_tensor=boundary_tensor,
                start_layer=boundary_layer + 1,
                attention_mask_tensor=attention_mask,
                position_ids=prefill_position_ids,
                past_key_values=cache,
                use_cache=True,
            )
            prefill_logits = self.model.lm_head(prefill_hidden)[0, -1].detach().cpu().to(self._torch.float32).numpy()
        kv_ms = (perf_counter() - prefill_started) * 1000.0

        kv_input_ids = input_ids.clone()
        kv_attention_mask = attention_mask.clone()
        per_step: list[dict[str, object]] = []
        first_divergence_step: int | None = None

        for step in range(steps):
            baseline_step = baseline["per_step"][step]

            if step == 0:
                logits = prefill_logits
            else:
                next_token_id = baseline["per_step"][step - 1]["direct_token_id"]
                next_token_tensor = self._torch.tensor([[next_token_id]], device=self.device, dtype=kv_input_ids.dtype)
                next_mask_tensor = self._torch.ones((1, 1), device=self.device, dtype=kv_attention_mask.dtype)
                kv_input_ids = self._torch.cat([kv_input_ids, next_token_tensor], dim=1)
                kv_attention_mask = self._torch.cat([kv_attention_mask, next_mask_tensor], dim=1)

                boundary_hidden = self.capture_boundary_hidden_from_inputs(
                    input_ids=kv_input_ids,
                    attention_mask=kv_attention_mask,
                    boundary_layer=boundary_layer,
                )
                last_boundary = boundary_hidden[-1:]
                hidden_tensor = self._torch.tensor(last_boundary, dtype=self.model_dtype, device=self.device).unsqueeze(0)
                position_ids = self._torch.tensor([[kv_input_ids.shape[1] - 1]], device=self.device)

                step_started = perf_counter()
                with self._torch.no_grad():
                    hidden_tensor, cache = self._run_upper_stack(
                        hidden_tensor=hidden_tensor,
                        start_layer=boundary_layer + 1,
                        attention_mask_tensor=kv_attention_mask,
                        position_ids=position_ids,
                        past_key_values=cache,
                        use_cache=True,
                    )
                    logits = self.model.lm_head(hidden_tensor)[0, -1].detach().cpu().to(self._torch.float32).numpy()
                kv_ms += (perf_counter() - step_started) * 1000.0

            direct_logits = self._resumed_next_token_logits_from_inputs(
                input_ids=kv_input_ids,
                attention_mask=kv_attention_mask,
                boundary_layer=boundary_layer,
            )
            delta = logits - direct_logits
            direct_token_id = int(np.argmax(direct_logits))
            kv_token_id = int(np.argmax(logits))
            exact_logits = bool(np.array_equal(logits, direct_logits))
            token_match = direct_token_id == kv_token_id
            step_result = {
                "step": step + 1,
                "l2_error": float(np.linalg.norm(delta)),
                "cosine_similarity": float(
                    np.dot(logits, direct_logits) / (np.linalg.norm(logits) * np.linalg.norm(direct_logits))
                ) if np.linalg.norm(logits) and np.linalg.norm(direct_logits) else 0.0,
                "max_abs_diff": float(np.max(np.abs(delta))),
                "baseline_token_id": direct_token_id,
                "baseline_token_text": self.tokenizer.decode([direct_token_id]),
                "kv_token_id": kv_token_id,
                "kv_token_text": self.tokenizer.decode([kv_token_id]),
                "token_match": token_match,
                "exact_logits": exact_logits,
                "baseline_top_k": self._top_k_tokens(direct_logits, top_k=top_k),
                "kv_top_k": self._top_k_tokens(logits, top_k=top_k),
            }
            per_step.append(step_result)

            if not token_match or not exact_logits:
                first_divergence_step = step + 1
                break

        return {
            "boundary_layer": boundary_layer,
            "steps_requested": steps,
            "steps_completed": len(per_step),
            "exact_match": first_divergence_step is None,
            "first_divergence_step": first_divergence_step,
            "baseline_ms": baseline_ms,
            "kv_ms": kv_ms,
            "cache_bytes": self._cache_bytes(cache),
            "per_step": per_step,
        }

    def diagnose_kv_step_two(
        self,
        *,
        text: str,
        boundary_layer: int,
        top_k: int = 5,
    ) -> dict[str, object]:
        encoded = self._encode_text(text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Step 1 exact next token from the frozen baseline path.
        step1_logits = self._resumed_next_token_logits_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        )
        step1_token_id = int(np.argmax(step1_logits))
        step1_token_text = self.tokenizer.decode([step1_token_id])

        # Exact baseline for step 2: append the step-1 token and recompute.
        step1_token_tensor = self._torch.tensor([[step1_token_id]], device=self.device, dtype=input_ids.dtype)
        step1_mask_tensor = self._torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
        step2_input_ids = self._torch.cat([input_ids, step1_token_tensor], dim=1)
        step2_attention_mask = self._torch.cat([attention_mask, step1_mask_tensor], dim=1)
        exact_boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=step2_input_ids,
            attention_mask=step2_attention_mask,
            boundary_layer=boundary_layer,
        )
        exact_hidden_tensor = self._torch.tensor(
            exact_boundary_hidden,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        exact_position_ids = self._build_position_ids(exact_hidden_tensor.shape[1])
        exact_trace, _, exact_logits = self._run_upper_stack_trace(
            hidden_tensor=exact_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=step2_attention_mask,
            position_ids=exact_position_ids,
            use_cache=False,
        )

        # KV path for step 2: prefill cache on the original prompt, then append only the new token.
        prefill_boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        )
        prefill_hidden_tensor = self._torch.tensor(
            prefill_boundary_hidden,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        cache = self._make_dynamic_cache()
        prefill_position_ids = self._build_position_ids(prefill_hidden_tensor.shape[1])
        _, cache, _ = self._run_upper_stack_trace(
            hidden_tensor=prefill_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=attention_mask,
            position_ids=prefill_position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        step2_boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=step2_input_ids,
            attention_mask=step2_attention_mask,
            boundary_layer=boundary_layer,
        )[-1:]
        kv_hidden_tensor = self._torch.tensor(
            step2_boundary_hidden,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        kv_position_ids = self._torch.tensor([[step2_input_ids.shape[1] - 1]], device=self.device)
        kv_trace, cache, kv_logits = self._run_upper_stack_trace(
            hidden_tensor=kv_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=step2_attention_mask,
            position_ids=kv_position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        layers = sorted(exact_trace)
        per_layer: list[dict[str, object]] = []
        first_divergent_layer: int | None = None
        for layer_index in layers:
            exact_state = exact_trace[layer_index]
            kv_state = kv_trace[layer_index]
            delta = kv_state - exact_state
            l2_error = float(np.linalg.norm(delta))
            max_abs_diff = float(np.max(np.abs(delta)))
            cosine_similarity = float(
                np.dot(kv_state, exact_state) / (np.linalg.norm(kv_state) * np.linalg.norm(exact_state))
            ) if np.linalg.norm(kv_state) and np.linalg.norm(exact_state) else 0.0
            exact_match = bool(np.array_equal(kv_state, exact_state))
            if first_divergent_layer is None and not exact_match:
                first_divergent_layer = layer_index
            per_layer.append(
                {
                    "layer": layer_index,
                    "l2_error": l2_error,
                    "cosine_similarity": cosine_similarity,
                    "max_abs_diff": max_abs_diff,
                    "exact_match": exact_match,
                }
            )

        logit_delta = kv_logits - exact_logits
        return {
            "boundary_layer": boundary_layer,
            "step1_token_id": step1_token_id,
            "step1_token_text": step1_token_text,
            "first_divergent_layer": first_divergent_layer,
            "cache_bytes": self._cache_bytes(cache),
            "logit_l2_error": float(np.linalg.norm(logit_delta)),
            "logit_cosine_similarity": float(
                np.dot(kv_logits, exact_logits) / (np.linalg.norm(kv_logits) * np.linalg.norm(exact_logits))
            ) if np.linalg.norm(kv_logits) and np.linalg.norm(exact_logits) else 0.0,
            "logit_max_abs_diff": float(np.max(np.abs(logit_delta))),
            "exact_top_k": self._top_k_tokens(exact_logits, top_k=top_k),
            "kv_top_k": self._top_k_tokens(kv_logits, top_k=top_k),
            "per_layer": per_layer,
        }

    def diagnose_kv_step_two_three_path(
        self,
        *,
        text: str,
        boundary_layer: int,
        top_k: int = 5,
    ) -> dict[str, object]:
        encoded = self._encode_text(text)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        # Path A: exact resumed baseline. Use the trusted resumed path to get step-1 token,
        # then build step-2 from the captured boundary hidden state of the extended prefix.
        step1_logits_a = self._resumed_next_token_logits_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        )
        step1_token_id = int(np.argmax(step1_logits_a))
        step1_token_text = self.tokenizer.decode([step1_token_id])
        next_token_tensor = self._torch.tensor([[step1_token_id]], device=self.device, dtype=input_ids.dtype)
        next_mask_tensor = self._torch.ones((1, 1), device=self.device, dtype=attention_mask.dtype)
        step2_input_ids = self._torch.cat([input_ids, next_token_tensor], dim=1)
        step2_attention_mask = self._torch.cat([attention_mask, next_mask_tensor], dim=1)

        resumed_boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=step2_input_ids,
            attention_mask=step2_attention_mask,
            boundary_layer=boundary_layer,
        )
        resumed_hidden_tensor = self._torch.tensor(
            resumed_boundary_hidden,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        resumed_position_ids = self._build_position_ids(resumed_hidden_tensor.shape[1])
        resumed_trace, _, resumed_logits = self._run_upper_stack_trace(
            hidden_tensor=resumed_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=step2_attention_mask,
            position_ids=resumed_position_ids,
            use_cache=False,
        )

        # Path B: full-sequence recompute through the whole model, then inspect the same upper stack.
        transformer = self._require_gpt2_model()
        with self._torch.no_grad():
            full_hidden_states, _, _ = self._prepare_gpt2_hidden_inputs(
                input_ids=step2_input_ids,
                attention_mask=step2_attention_mask,
            )
            for layer_index in range(0, boundary_layer + 1):
                block = transformer.h[layer_index]
                full_position_ids = self._build_position_ids(full_hidden_states.shape[1])
                full_mask = self._build_causal_attention_mask(
                    full_hidden_states,
                    step2_attention_mask,
                    full_position_ids,
                )
                block_output = block(
                    full_hidden_states,
                    attention_mask=full_mask,
                    use_cache=False,
                    position_ids=full_position_ids,
                )
                full_hidden_states = block_output[0] if isinstance(block_output, tuple) else block_output
            full_boundary_hidden = full_hidden_states[0].detach().cpu().to(self._torch.float32).numpy()
        full_hidden_tensor = self._torch.tensor(
            full_boundary_hidden,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        full_position_ids = self._build_position_ids(full_hidden_tensor.shape[1])
        full_trace, _, full_logits = self._run_upper_stack_trace(
            hidden_tensor=full_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=step2_attention_mask,
            position_ids=full_position_ids,
            use_cache=False,
        )

        # Path C: KV-aware path. Prefill upper-stack cache on the old prefix, then append only the new token.
        prefill_boundary_hidden = self.capture_boundary_hidden_from_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            boundary_layer=boundary_layer,
        )
        prefill_hidden_tensor = self._torch.tensor(
            prefill_boundary_hidden,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        cache = self._make_dynamic_cache()
        prefill_position_ids = self._build_position_ids(prefill_hidden_tensor.shape[1])
        _, cache, _ = self._run_upper_stack_trace(
            hidden_tensor=prefill_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=attention_mask,
            position_ids=prefill_position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        kv_boundary_last = resumed_boundary_hidden[-1:]
        kv_hidden_tensor = self._torch.tensor(
            kv_boundary_last,
            dtype=self.model_dtype,
            device=self.device,
        ).unsqueeze(0)
        kv_position_ids = self._torch.tensor([[step2_input_ids.shape[1] - 1]], device=self.device)
        kv_trace, cache, kv_logits = self._run_upper_stack_trace(
            hidden_tensor=kv_hidden_tensor,
            start_layer=boundary_layer + 1,
            attention_mask_tensor=step2_attention_mask,
            position_ids=kv_position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        layers = sorted(resumed_trace)
        per_layer: list[dict[str, object]] = []
        first_kv_divergent_layer: int | None = None
        first_recompute_divergent_layer: int | None = None
        for layer_index in layers:
            a = resumed_trace[layer_index]
            b = full_trace[layer_index]
            c = kv_trace[layer_index]

            ab = b - a
            ac = c - a
            bc = c - b

            ab_l2 = float(np.linalg.norm(ab))
            ac_l2 = float(np.linalg.norm(ac))
            bc_l2 = float(np.linalg.norm(bc))
            ab_max = float(np.max(np.abs(ab)))
            ac_max = float(np.max(np.abs(ac)))
            bc_max = float(np.max(np.abs(bc)))
            ab_exact = bool(np.array_equal(a, b))
            ac_exact = bool(np.array_equal(a, c))
            bc_exact = bool(np.array_equal(b, c))

            if first_recompute_divergent_layer is None and not ab_exact:
                first_recompute_divergent_layer = layer_index
            if first_kv_divergent_layer is None and not ac_exact:
                first_kv_divergent_layer = layer_index

            per_layer.append(
                {
                    "layer": layer_index,
                    "ab_l2": ab_l2,
                    "ac_l2": ac_l2,
                    "bc_l2": bc_l2,
                    "ab_max_abs_diff": ab_max,
                    "ac_max_abs_diff": ac_max,
                    "bc_max_abs_diff": bc_max,
                    "ab_exact": ab_exact,
                    "ac_exact": ac_exact,
                    "bc_exact": bc_exact,
                }
            )

        def logit_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
            delta = right - left
            return {
                "l2_error": float(np.linalg.norm(delta)),
                "max_abs_diff": float(np.max(np.abs(delta))),
                "cosine_similarity": float(
                    np.dot(right, left) / (np.linalg.norm(right) * np.linalg.norm(left))
                ) if np.linalg.norm(right) and np.linalg.norm(left) else 0.0,
            }

        return {
            "boundary_layer": boundary_layer,
            "step1_token_id": step1_token_id,
            "step1_token_text": step1_token_text,
            "first_recompute_divergent_layer": first_recompute_divergent_layer,
            "first_kv_divergent_layer": first_kv_divergent_layer,
            "cache_bytes": self._cache_bytes(cache),
            "ab_logits": logit_metrics(resumed_logits, full_logits),
            "ac_logits": logit_metrics(resumed_logits, kv_logits),
            "bc_logits": logit_metrics(full_logits, kv_logits),
            "a_top_k": self._top_k_tokens(resumed_logits, top_k=top_k),
            "b_top_k": self._top_k_tokens(full_logits, top_k=top_k),
            "c_top_k": self._top_k_tokens(kv_logits, top_k=top_k),
            "per_layer": per_layer,
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
