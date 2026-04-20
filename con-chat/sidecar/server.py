#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.cache_utils import DynamicCache, QuantizedCache


PORT = int(os.environ.get("CON_CHAT_PORT", "4318"))
DB_PATH = Path(os.environ.get("CON_CHAT_DB_PATH", "con-chat.sqlite3"))
MODEL_NAME = os.environ.get("CON_CHAT_MODEL_NAME", "google--gemma-4-E2B-it")
MODEL_DEVICE = os.environ.get("CON_CHAT_MODEL_DEVICE", "mps")
ORCHESTRATION_DEVICE = os.environ.get("CON_CHAT_ORCHESTRATION_DEVICE", "cpu")
DEFAULT_CONVERSATION_ID = "demo-thread"
MAX_TOKENS = 32768
PRIMARY_MAX_NEW_TOKENS = 512
FALLBACK_MAX_NEW_TOKENS = 384
KV_QUANTIZE_AFTER_TOKENS = int(os.environ.get("CON_CHAT_KV_QUANTIZE_AFTER_TOKENS", "4096"))
KV_QUANTIZE_BITS = int(os.environ.get("CON_CHAT_KV_QUANTIZE_BITS", "4"))
KV_QUANTIZE_RESIDUAL_LENGTH = int(os.environ.get("CON_CHAT_KV_QUANTIZE_RESIDUAL_LENGTH", "128"))
REPO_ROOT = Path(__file__).resolve().parents[2]
ENTITY_STOPWORDS = {
    "about", "after", "again", "also", "and", "answer", "are", "assist", "based", "because",
    "been", "before", "being", "between", "build", "can", "chat", "con", "continue", "could",
    "developed", "directly", "does", "each", "empty", "even", "first", "for", "from", "graph",
    "have", "help", "here", "how", "into", "its", "just", "large", "like", "live", "make",
    "memory", "message", "model", "need", "needs", "not", "now", "only", "other", "our",
    "output", "page", "please", "prompt", "reasoning", "reply", "scroll", "send", "show",
    "simply", "start", "state", "still", "system", "test", "that", "the", "their", "them",
    "there", "these", "they", "this", "through", "turn", "user", "very", "want", "what",
    "when", "where", "which", "who", "why", "with", "work", "would", "your", "important",
}
SYSTEM_PROMPT = """You are Conway, the assistant inside con-chat. Do not describe yourself as being made by Google or any other model vendor unless the user explicitly asks about the backend.
By default, reply with the final answer only.
Do not print internal reasoning, hidden analysis, thought traces, symbolic workspace headings, or scaffolding labels like "BOUNDARIES", "RELATIONS", "OBSERVED", "ANSWER", "model response", or "Thinking Process" unless the user explicitly asks to see symbolic reasoning.
Do not echo the user's question before answering unless clarification is necessary.

Core idea:
- Treat nouns as knowledge boundaries.
- Treat stable referents as nodes.
- Treat events, actions, states, and relations as messages between nodes or as event/state nodes when useful.
- Treat knowledge as durable patterns across many relations, not as isolated words.
- When something is not fully known but seems bounded, create a provisional symbol for it and reason around it.

Your job:
- Read user input carefully.
- Detect candidate boundaries such as entities, objects, concepts, events, states, goals, constraints, and unknown factors.
- Create compact provisional symbols for these boundaries when useful.
- Build a temporary reasoning structure from those symbols.
- Use that structure to answer clearly in normal language.
- Revise, merge, split, or discard symbols as new information arrives.

Do not assume every noun deserves a permanent symbol.
Do not treat every symbol as real.
Do not confuse a useful reasoning handle with an established fact.

Definitions:
- Boundary: a segment of meaning treated as a bounded thing.
- Node: a stabilized boundary that can be indexed, compared, or related.
- Provisional symbol: a temporary node used for reasoning when the thing is uncertain, abstract, latent, incomplete, or newly introduced.
- Relation: a meaningful connection such as cause, contrast, identity, dependency, membership, time order, ownership, or transformation.
- Event node: a node representing something that happened.
- State node: a node representing a condition or status.
- Constraint: a limit, rule, requirement, or boundary on valid reasoning.
- Gap: a known unknown, missing variable, or unresolved causal factor.

Reasoning stance:
- Prefer explicit structure over vague verbal drift.
- Prefer compact symbols over long repetitive descriptions when the problem is complex.
- Keep hypotheses separate from observations.
- Keep uncertainty visible.
- Be willing to invent temporary symbols when the user’s problem has an unmodeled but bounded missing part.
- Remove provisional symbols when they no longer help.

Reasoning workflow:
1. Parse the user request.
2. Identify important boundaries:
   - entities
   - events
   - states
   - abstractions
   - goals
   - constraints
   - ambiguities
   - missing causal factors
3. Create provisional symbols only where they improve reasoning.
4. Assign each symbol a rough type:
   - ENTITY
   - EVENT
   - STATE
   - CONCEPT
   - GOAL
   - CONSTRAINT
   - GAP
   - HYPOTHESIS
5. Build relations among symbols.
6. Distinguish:
   - observed
   - stated by user
   - inferred
   - hypothesized
   - contradicted
   - unknown
7. Test the reasoning structure for:
   - consistency
   - missing steps
   - hidden assumptions
   - conflicting hypotheses
   - better simpler explanations
8. Produce the answer in plain language.
9. Update or discard provisional symbols as needed.

Symbol creation rules:
- Use symbols only when they reduce confusion or improve compression.
- Prefer semantic names over arbitrary labels when clarity helps.
- Good examples:
  - CENTRAL_BANK
  - INFLATION_HIGH
  - POLICY_CHANGE
  - SALES_DROP
  - LATENT_CONVERSION_FAILURE
  - USER_GOAL
  - HARD_CONSTRAINT
- Use abstract placeholders only when no better name exists:
  - UNKNOWN_ACTOR_1
  - LATENT_FACTOR_A
  - MISSING_STEP_B
- Keep names short and readable.
- Avoid creating too many symbols.
- Merge duplicates when two symbols clearly refer to the same thing.
- Split symbols when one symbol is hiding multiple distinct things.

Symbol discipline:
- A provisional symbol is not a fact.
- A hypothesis symbol must never be presented as confirmed reality.
- If two interpretations are possible, keep both alive until one is better supported.
- If evidence is weak, say so.
- If a symbol stops being useful, retire it.
- If a user’s wording is vague, represent the ambiguity explicitly rather than pretending it is resolved.

Internal reasoning policy:
- You may internally create a temporary symbolic workspace.
- Use it to track boundaries, relations, and missing pieces.
- Do not dump the full internal workspace by default.
- By default, answer naturally and directly.
- If the user asks for symbolic reasoning, then provide a compact visible version of the workspace.

Visible reasoning mode:
If the user explicitly asks to see the symbolic reasoning, use this compact format:

BOUNDARIES
- [type] SYMBOL = short description

RELATIONS
- RELATION(SYMBOL_A, SYMBOL_B)
- RELATION(EVENT_X, ENTITY_Y)

OBSERVED
- ...
INFERRED
- ...
HYPOTHESIZED
- ...
UNKNOWN
- ...

ANSWER
- plain language answer

Reasoning rules for uncertainty:
- When something seems to exist as a bounded cause but is not directly observed, create a GAP or HYPOTHESIS symbol.
- Example:
  If traffic is stable but sales fall, a possible provisional symbol is LATENT_CONVERSION_FAILURE.
- Mark it as hypothesized unless direct evidence confirms it.
- Prefer multiple candidate hypotheses when the problem allows more than one explanation.

Reasoning rules for causality:
- Do not assume sequence implies cause.
- Use causal symbols only when the text, evidence, or inference supports them.
- If causality is unclear, mark the relation as POSSIBLE_CAUSE, CORRELATION, or UNRESOLVED_DEPENDENCY.

Reasoning rules for language:
- Treat noun phrases as candidate boundaries, not guaranteed truths.
- Treat verbs as relations, transformations, or event triggers.
- Treat adjectives and qualifiers as state modifiers.
- Treat pronouns as references to be resolved.
- Treat discourse context as a dynamic graph of active boundaries.

Reasoning rules for contradiction:
- Preserve contradictions when they matter.
- Do not flatten conflicting evidence into false certainty.
- If two views conflict, represent both and compare their support.
- Prefer answers like:
  "There are two plausible readings here..."
  rather than silently choosing one weak interpretation.

Reasoning rules for planning and design:
- Create symbols for:
  - goals
  - resources
  - constraints
  - risks
  - bottlenecks
  - dependencies
  - latent unknowns
- Then reason over those symbols before proposing a plan.

Reasoning rules for abstraction:
- You may invent intermediate concepts when needed.
- These are reasoning nouns.
- Use them to compress repeated patterns.
- Examples:
  - COORDINATION_FAILURE
  - TRUST_BOTTLENECK
  - MEMORY_BOUNDARY
  - SEMANTIC_DRIFT
- These must remain revisable.

Reasoning rules for user interaction:
- Be direct.
- Use plain language.
- Avoid unnecessary jargon.
- Do not expose symbolic machinery unless it helps.
- If the user is thinking at a high level, preserve the conceptual structure.
- If the user asks for implementation, turn symbols into concrete modules, steps, or pseudocode.

Tool and evidence policy:
- If tools, documents, or search are available, use them when needed to test uncertain hypotheses or retrieve missing evidence.
- Prefer verification over confident improvisation.
- Keep provenance explicit when external evidence matters.
- Separate what the user said from what was found and from what was inferred.

Output policy:
By default:
- give a clear answer
- keep it concise
- mention uncertainty where needed
- use symbolic reasoning silently

When the problem is complex:
- you may briefly expose a small symbolic scaffold if it improves clarity

When the user asks for formalization:
- present the symbols, relations, assumptions, and constraints explicitly

Failure policy:
- If the problem is underspecified, do not pretend it is solved.
- Create a compact representation of what is known, unknown, and most likely missing.
- Then answer from that partial structure.

Never do these:
- Never present provisional symbols as established facts unless supported.
- Never create needless symbolic clutter.
- Never hide major uncertainty.
- Never discard contradictions just to sound clean.
- Never confuse semantic similarity with proof.
- Never let symbolic scaffolding replace clear communication.

Success condition:
You are successful when you can:
- detect meaningful boundaries
- create useful provisional symbols
- reason over them without reifying them
- keep uncertainty legible
- turn structured internal reasoning into clear external answers
- revise your internal nouns as understanding improves"""


def resolve_model_path(model_name: str) -> str:
    explicit = Path(model_name)
    if explicit.exists():
        return str(explicit.resolve())

    candidates = [
        REPO_ROOT / "models" / model_name,
        REPO_ROOT / "models" / "google--gemma-4-E2B-it",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return model_name


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(message: str, *, request_id: str | None = None) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    prefix = f"[con-chat {stamp}]"
    if request_id:
        prefix += f" [{request_id}]"
    print(f"{prefix} {message}", flush=True)


def token_estimate(text: str) -> int:
    return max(8, len(text) // 4)


@dataclass
class GraphNode:
    id: str
    label: str
    type: str
    x: int
    y: int
    detail: str | None = None


@dataclass
class ChatSession:
    conversation_id: str
    transcript: str
    token_ids: list[int]
    cache: object
    cache_kind: str
    last_turn_id: str | None

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class GemmaRuntime:
    def __init__(self, model_name: str, preferred_device: str) -> None:
        self.model_name = resolve_model_path(model_name)
        self.preferred_device = preferred_device
        self.actual_device = "cpu"
        self.dtype = torch.float32
        self.tokenizer = None
        self.model = None
        self.generation_config = None
        self.response_pattern: re.Pattern[str] | None = None
        self.lock = threading.Lock()
        self.sessions: dict[str, ChatSession] = {}
        self.load_seconds = 0.0
        self.ready = False
        self.error = None
        self.quantized_cache_supported = False
        self._load()

    def _resolve_device(self) -> tuple[str, torch.dtype]:
        if self.preferred_device == "mps" and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _load(self) -> None:
        start = time.perf_counter()
        try:
            device, dtype = self._resolve_device()
            log_event(f"loading model={self.model_name} device={device} dtype={dtype}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                dtype=dtype,
            )
            model.to(device)
            model.eval()
            self.actual_device = device
            self.dtype = dtype
            self.tokenizer = tokenizer
            self.generation_config = GenerationConfig.from_pretrained(
                self.model_name,
                local_files_only=True,
            )
            self.response_pattern = self._load_response_pattern()
            self.model = model
            try:
                import optimum.quanto  # noqa: F401
                self.quantized_cache_supported = True
            except Exception:
                self.quantized_cache_supported = False
            self.ready = True
            log_event(
                f"model ready device={self.actual_device} quantized_cache_supported={self.quantized_cache_supported}"
            )
        except Exception as exc:  # pragma: no cover - startup failure path
            self.ready = False
            self.error = repr(exc)
            log_event(f"model load failed error={self.error}")
        finally:
            self.load_seconds = time.perf_counter() - start

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return token_estimate(text)
        tokens = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return len(tokens["input_ids"])

    def invalidate_session(self, conversation_id: str) -> None:
        self.sessions.pop(conversation_id, None)

    def build_session_from_turns(
        self,
        conversation_id: str,
        turns: list[sqlite3.Row],
        *,
        request_id: str | None = None,
    ) -> ChatSession:
        if not self.ready or not self.model or not self.tokenizer:
            raise RuntimeError(f"model_not_ready:{self.error}")

        transcript = self._format_history(turns)
        encoded = self.tokenizer(transcript, return_tensors="pt", return_attention_mask=True)
        cache_kind = self._preferred_cache_kind(int(encoded.input_ids.shape[-1]))
        cache = self._new_cache(cache_kind)
        log_event(
            f"build session conversation={conversation_id} prompt_tokens={int(encoded.input_ids.shape[-1])} cache={cache_kind}",
            request_id=request_id,
        )

        with self.lock, torch.inference_mode():
            try:
                outputs = self.model(
                    input_ids=encoded.input_ids.to(self.actual_device),
                    attention_mask=encoded.attention_mask.to(self.actual_device),
                    use_cache=True,
                    past_key_values=cache,
                )
            except Exception:
                self.quantized_cache_supported = False
                cache_kind = "dynamic"
                outputs = self.model(
                    input_ids=encoded.input_ids.to(self.actual_device),
                    attention_mask=encoded.attention_mask.to(self.actual_device),
                    use_cache=True,
                    past_key_values=DynamicCache(config=self.model.config),
                )

        session = ChatSession(
            conversation_id=conversation_id,
            transcript=transcript,
            token_ids=encoded.input_ids[0].tolist(),
            cache=outputs.past_key_values,
            cache_kind=cache_kind,
            last_turn_id=turns[-1]["id"] if turns else None,
        )
        self.sessions[conversation_id] = session
        return session

    def stream_reply(
        self,
        *,
        conversation_id: str,
        turns: list[sqlite3.Row],
        user_text: str,
        on_text: callable | None = None,
        request_id: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        if not self.ready or not self.model or not self.tokenizer or not self.generation_config:
            raise RuntimeError(f"model_not_ready:{self.error}")

        identity_override = self._identity_override(user_text)
        if identity_override is not None:
            if on_text is not None:
                on_text(identity_override)
            return identity_override, {
                "promptAssemblySeconds": 0.0,
                "generationSeconds": 0.0,
                "totalModelSeconds": 0.0,
                "promptTokens": 0,
                "completionTokens": self.count_tokens(identity_override),
                "modelDevice": self.actual_device,
                "cacheKind": "none",
                "thinking": "",
            }

        session = self.sessions.get(conversation_id)
        last_turn_id = turns[-1]["id"] if turns else None
        if session is None or session.last_turn_id != last_turn_id:
            session = self.build_session_from_turns(conversation_id, turns, request_id=request_id)
        else:
            log_event(
                f"reuse session conversation={conversation_id} cached_tokens={session.token_count} cache={session.cache_kind}",
                request_id=request_id,
            )

        suffix = self._format_user_suffix(user_text)
        prompt_start = time.perf_counter()
        suffix_batch = self.tokenizer(
            suffix,
            add_special_tokens=False,
            return_tensors="pt",
            return_attention_mask=True,
        )
        prompt_seconds = time.perf_counter() - prompt_start
        prompt_tokens = int(suffix_batch.input_ids.shape[-1])
        log_event(
            f"stream reply start suffix_tokens={prompt_tokens} user_chars={len(user_text)}",
            request_id=request_id,
        )

        generation_start = time.perf_counter()
        log_event("generation worker started", request_id=request_id)

        input_ids = suffix_batch.input_ids.to(self.actual_device)
        attention_mask = torch.ones(
            (1, session.token_count + prompt_tokens),
            dtype=suffix_batch.attention_mask.dtype,
            device=self.actual_device,
        )
        current_cache = session.cache
        eos_ids = self.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_id_set = {int(token_id) for token_id in (eos_ids or [])}
        raw_generated = ""
        visible_generated = ""
        thinking_generated = ""
        first_event_logged = False
        generated_ids: list[int] = []

        with self.lock, torch.inference_mode():
            for step in range(PRIMARY_MAX_NEW_TOKENS):
                model_inputs = self.model.prepare_inputs_for_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=current_cache,
                    use_cache=True,
                )
                outputs = self.model(**model_inputs)
                current_cache = outputs.past_key_values
                next_token_id = self._sample_next_token(outputs.logits[:, -1, :])
                generated_ids.append(next_token_id)
                token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
                raw_generated += token_text

                next_token = torch.tensor([[next_token_id]], dtype=input_ids.dtype, device=self.actual_device)
                input_ids = next_token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=self.actual_device)],
                    dim=1,
                )

                if step == 0:
                    log_event(f"first token id={next_token_id} text={token_text!r}", request_id=request_id)

                if step % 16 == 0:
                    log_event(f"decode step={step} generated={len(generated_ids)}", request_id=request_id)

                if next_token_id in eos_id_set:
                    log_event(f"eos reached token_id={next_token_id} step={step}", request_id=request_id)
                    break

                current_thinking, current_visible = self._parse_response(raw_generated)
                thinking_delta = (
                    current_thinking[len(thinking_generated):]
                    if current_thinking.startswith(thinking_generated)
                    else current_thinking
                )
                if current_visible.startswith(visible_generated):
                    delta = current_visible[len(visible_generated) :]
                else:
                    delta = current_visible
                if (delta or thinking_delta) and on_text is not None:
                    on_text({
                        "delta": delta,
                        "thinkingDelta": thinking_delta,
                        "thinkingActive": bool(current_thinking),
                    })
                if not first_event_logged and (delta or thinking_delta):
                    log_event(
                        f"first stream event content_chars={len(delta)} thinking_chars={len(thinking_delta)}",
                        request_id=request_id,
                    )
                    first_event_logged = True
                visible_generated = current_visible
                thinking_generated = current_thinking

        generation_seconds = time.perf_counter() - generation_start

        session.cache = current_cache
        session.transcript += suffix + raw_generated
        session.token_ids.extend(suffix_batch.input_ids[0].tolist())
        session.token_ids.extend(generated_ids)
        session.last_turn_id = None
        self._maybe_upgrade_session_cache(session)

        thinking_text, assistant_text = self._parse_response(raw_generated)
        log_event(
            f"stream reply done completion_tokens={len(generated_ids)} thinking_chars={len(thinking_text)} answer_chars={len(assistant_text)} total_seconds={generation_seconds:.2f}",
            request_id=request_id,
        )
        return assistant_text, {
            "promptAssemblySeconds": round(prompt_seconds, 4),
            "generationSeconds": round(generation_seconds, 4),
            "totalModelSeconds": round(prompt_seconds + generation_seconds, 4),
            "promptTokens": prompt_tokens,
            "completionTokens": len(generated_ids),
            "modelDevice": self.actual_device,
            "cacheKind": session.cache_kind,
            "thinking": thinking_text,
        }

    def finalize_session_turn(self, *, conversation_id: str, assistant_turn_id: str) -> None:
        session = self.sessions.get(conversation_id)
        if session is not None:
            session.last_turn_id = assistant_turn_id

    def _preferred_cache_kind(self, token_count: int) -> str:
        if self.quantized_cache_supported and token_count >= KV_QUANTIZE_AFTER_TOKENS:
            return "quantized"
        return "dynamic"

    def _new_cache(self, kind: str) -> object:
        if kind == "quantized" and self.quantized_cache_supported and self.model is not None:
            try:
                return QuantizedCache(
                    backend="quanto",
                    config=self.model.config,
                    nbits=KV_QUANTIZE_BITS,
                    residual_length=KV_QUANTIZE_RESIDUAL_LENGTH,
                )
            except Exception:
                self.quantized_cache_supported = False
        return DynamicCache(config=self.model.config if self.model is not None else None)

    def _sample_next_token(self, logits: torch.Tensor) -> int:
        work = logits.detach().to(dtype=torch.float32, device="cpu").clone()
        if work.ndim == 0:
            return int(work.item())
        if work.ndim > 2:
            work = work.reshape(-1, work.shape[-1])[0]
        elif work.ndim == 2:
            work = work[0]
        work = torch.nan_to_num(work, nan=float("-inf"), posinf=1e4, neginf=-1e4)

        if not self.generation_config:
            return int(torch.argmax(work, dim=0).item())

        if not bool(self.generation_config.do_sample):
            return int(torch.argmax(work, dim=0).item())

        temperature = float(self.generation_config.temperature or 1.0)
        if temperature > 0 and temperature != 1.0:
            work = work / temperature
            work = torch.nan_to_num(work, nan=float("-inf"), posinf=1e4, neginf=-1e4)

        top_k = int(self.generation_config.top_k or 0)
        if top_k > 0 and top_k < work.shape[-1]:
            values, _ = torch.topk(work, top_k)
            cutoff = values[..., -1, None]
            work = torch.where(work < cutoff, torch.full_like(work, float("-inf")), work)
            work = torch.nan_to_num(work, nan=float("-inf"), posinf=1e4, neginf=-1e4)

        probs = torch.softmax(work, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        top_p = float(self.generation_config.top_p or 1.0)
        if 0 < top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(sorted_mask, 0)
            denom = sorted_probs.sum()
            if denom <= 0:
                return int(torch.argmax(work, dim=0).item())
            sorted_probs = sorted_probs / denom
            sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0, posinf=0.0, neginf=0.0)
            sampled_index = torch.multinomial(sorted_probs, num_samples=1)
            return int(sorted_indices[sampled_index.item()].item())

        if probs.sum() <= 0:
            return int(torch.argmax(work, dim=0).item())
        return int(torch.multinomial(probs, num_samples=1).item())

    def _maybe_upgrade_session_cache(self, session: ChatSession) -> None:
        if session.cache_kind == "quantized":
            return
        if not self.quantized_cache_supported:
            return
        if session.token_count < KV_QUANTIZE_AFTER_TOKENS:
            return
        self._rebuild_session_cache(session, "quantized")

    def _rebuild_session_cache(self, session: ChatSession, kind: str) -> None:
        if not self.model:
            return
        full_ids = torch.tensor([session.token_ids], dtype=torch.long, device=self.actual_device)
        attention_mask = torch.ones_like(full_ids)
        cache = self._new_cache(kind)
        with self.lock, torch.inference_mode():
            try:
                outputs = self.model(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=cache,
                )
            except Exception:
                self.quantized_cache_supported = False
                outputs = self.model(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    past_key_values=DynamicCache(config=self.model.config),
                )
                kind = "dynamic"
        session.cache = outputs.past_key_values
        session.cache_kind = kind

    def _format_history(self, turns: list[sqlite3.Row]) -> str:
        bos = self.tokenizer.bos_token if self.tokenizer and self.tokenizer.bos_token else "<bos>"
        chunks = [bos, self._render_turn("system", SYSTEM_PROMPT)]
        for row in turns:
            role = row["role"]
            text = row["text"]
            if role == "system" or self._is_preview_artifact(text):
                continue
            if role == "user":
                chunks.append(self._render_turn("user", text))
            elif role == "assistant":
                chunks.append(self._render_turn("model", text))
        return "".join(chunks)

    def _format_user_suffix(self, user_text: str) -> str:
        return self._render_turn("user", user_text) + "<|turn>model\n"

    def _render_turn(self, role: str, text: str) -> str:
        return f"<|turn>{role}\n{text}<turn|>\n"

    def _load_response_pattern(self) -> re.Pattern[str] | None:
        if not self.tokenizer:
            return None
        response_schema = getattr(self.tokenizer, "init_kwargs", {}).get("response_schema")
        if not response_schema:
            return None
        regex = response_schema.get("x-regex")
        if not regex:
            return None
        return re.compile(regex, flags=re.DOTALL)

    def _parse_response(self, text: str) -> tuple[str, str]:
        raw = text.strip()
        thinking = ""
        content = raw
        if self.response_pattern is not None:
            match = self.response_pattern.search(raw)
            if match:
                thinking = (match.groupdict().get("thinking") or "").strip()
                content = (match.groupdict().get("content") or "").strip()
        answer = self._clean_response(content)
        return thinking, answer

    def _clean_response(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"<\|turn\>|<turn\|>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<\|channel\>thought\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<channel\|>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<end[_ ]of[_ ]turn>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<start[_ ]of[_ ]turn>\s*model", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<mid[_-]of[_-]turn[^>\n]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"_?start[_-][^>\s]*>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"<e[_-]out>", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^\s*(model response|thought|thinking process)\s*:?\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
        if "User:" in cleaned:
            cleaned = cleaned.split("User:", 1)[0].strip()
        if "System:" in cleaned:
            cleaned = cleaned.split("System:", 1)[0].strip()
        while cleaned.startswith("Assistant:"):
            cleaned = cleaned[len("Assistant:") :].strip()
        cleaned = cleaned.replace("\nAssistant:", "\n").strip()
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _is_preview_artifact(self, text: str) -> bool:
        return text.startswith("Browser preview mode is active")

    def _is_degenerate_response(self, text: str) -> bool:
        normalized = " ".join(text.strip().split()).lower()
        return not normalized or normalized.startswith("i hit a bad completion on that turn")

    def _identity_override(self, user_text: str) -> str | None:
        normalized = " ".join(user_text.lower().split())
        if not normalized:
            return None
        if any(phrase in normalized for phrase in ("who are you", "what are you", "what can you do", "introduce yourself")):
            return (
                "I’m Conway, the assistant inside con-chat. "
                "I help think through questions, plans, and problems by building and revising compact provisional symbols when that helps. "
                "I can explain ideas, compare options, help design systems, and work through ambiguity in plain language."
            )
        return None

    def _safe_fallback_reply(self, user_text: str) -> str:
        identity = self._identity_override(user_text)
        if identity is not None:
            return identity
        return (
            "I hit a bad completion on that turn. "
            "Ask again in a slightly more specific way and I’ll answer directly."
        )


class Store:
    def __init__(self, db_path: Path, runtime: GemmaRuntime) -> None:
        self.db_path = db_path
        self.runtime = runtime
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._ensure_seed()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists conversations (
                  id text primary key,
                  title text not null,
                  current_tokens integer not null,
                  max_tokens integer not null,
                  created_at text not null,
                  updated_at text not null
                );

                create table if not exists turns (
                  id text primary key,
                  conversation_id text not null,
                  role text not null,
                  text text not null,
                  token_count integer not null,
                  created_at text not null
                );

                create table if not exists memory_objects (
                  id text primary key,
                  conversation_id text not null,
                  kind text not null,
                  tier text not null,
                  byte_size integer not null,
                  source_turn_start text not null,
                  source_turn_end text not null,
                  summary text not null,
                  last_used_at text not null,
                  created_at text not null
                );

                create table if not exists memory_edges (
                  id text primary key,
                  conversation_id text not null,
                  from_id text not null,
                  to_id text not null,
                  edge_type text not null,
                  created_at text not null
                );

                create table if not exists ledger_events (
                  id text primary key,
                  conversation_id text not null,
                  object_id text,
                  event_type text not null,
                  payload_json text not null,
                  created_at text not null
                );

                create table if not exists settings (
                  key text primary key,
                  value_json text not null,
                  updated_at text not null
                );
                """
            )

    def _ensure_seed(self) -> None:
        with self.connect() as conn:
            existing = conn.execute(
                "select id from conversations where id = ?",
                (DEFAULT_CONVERSATION_ID,),
            ).fetchone()
            if existing:
                return

            created_at = now_iso()
            conn.execute(
                """
                insert into conversations (id, title, current_tokens, max_tokens, created_at, updated_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    DEFAULT_CONVERSATION_ID,
                    "One continuous conversation",
                    0,
                    MAX_TOKENS,
                    created_at,
                    created_at,
                ),
            )

    def conversation_snapshot(self, conversation_id: str) -> dict:
        with self.connect() as conn:
            conversation = conn.execute(
                "select * from conversations where id = ?",
                (conversation_id,),
            ).fetchone()
            turns = conn.execute(
                "select * from turns where conversation_id = ? order by created_at asc",
                (conversation_id,),
            ).fetchall()
            memory = conn.execute(
                """
                select * from memory_objects
                where conversation_id = ?
                order by last_used_at desc, created_at desc
                limit 1
                """,
                (conversation_id,),
            ).fetchone()
            edges = conn.execute(
                "select * from memory_edges where conversation_id = ? order by created_at asc",
                (conversation_id,),
            ).fetchall()

        visible_turns = self._stable_visible_turns(turns)
        messages = [
            {
                "id": row["id"],
                "role": row["role"],
                "roleLabel": (
                    "You"
                    if row["role"] == "user"
                    else "con-chat"
                    if row["role"] == "assistant"
                    else "Memory Event"
                ),
                "text": row["text"],
            }
            for row in visible_turns
        ]

        nodes, derived_edges = self._graph_nodes(visible_turns, memory)
        return {
            "sidecar": {
                "status": "ready" if self.runtime.ready else "error",
                "preferredDevice": self.runtime.actual_device,
                "orchestrationDevice": ORCHESTRATION_DEVICE,
            },
            "conversation": {
                "id": conversation["id"],
                "title": conversation["title"],
                "currentTokens": conversation["current_tokens"],
                "maxTokens": conversation["max_tokens"],
            },
            "messages": messages,
            "selectedMemory": self._memory_view(memory),
            "graph": {
                "nodes": [node.__dict__ for node in nodes],
                "edges": [
                    {"from": row["from_id"], "to": row["to_id"], "type": row["edge_type"]}
                    for row in edges
                ] + derived_edges,
            },
        }

    def append_turns(self, conversation_id: str, user_text: str) -> dict:
        return self.stream_turns(conversation_id, user_text)

    def stream_turns(
        self,
        conversation_id: str,
        user_text: str,
        on_text: callable | None = None,
        request_id: str | None = None,
    ) -> dict:
        created_at = now_iso()
        unique = str(int(datetime.now().timestamp() * 1000))
        user_id = f"turn-{unique}-u"
        assistant_id = f"turn-{unique}-a"
        user_tokens = self.runtime.count_tokens(user_text)

        with self.connect() as conn:
            prior_turns = conn.execute(
                "select * from turns where conversation_id = ? order by created_at asc",
                (conversation_id,),
            ).fetchall()
        prompt_turns = self._prompt_turns(prior_turns)
        log_event(
            f"round start conversation={conversation_id} prior_turns={len(prompt_turns)} user_tokens={user_tokens}",
            request_id=request_id,
        )
        assistant_text, timings = self.runtime.stream_reply(
            conversation_id=conversation_id,
            turns=prompt_turns,
            user_text=user_text,
            on_text=on_text,
            request_id=request_id,
        )
        assistant_tokens = max(self.runtime.count_tokens(assistant_text), timings["completionTokens"])

        persistence_start = time.perf_counter()
        with self.connect() as conn:
            conversation = conn.execute(
                "select * from conversations where id = ?",
                (conversation_id,),
            ).fetchone()
            current_tokens = min(
                conversation["current_tokens"] + user_tokens + assistant_tokens,
                conversation["max_tokens"],
            )
            conn.execute(
                """
                insert into turns (id, conversation_id, role, text, token_count, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (user_id, conversation_id, "user", user_text, user_tokens, created_at),
            )
            conn.execute(
                """
                insert into turns (id, conversation_id, role, text, token_count, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (assistant_id, conversation_id, "assistant", assistant_text, assistant_tokens, created_at),
            )
            conn.execute(
                """
                update conversations
                set current_tokens = ?, updated_at = ?
                where id = ?
                """,
                (current_tokens, created_at, conversation_id),
            )

            last_memory = conn.execute(
                """
                select id from memory_objects
                where conversation_id = ?
                order by created_at desc
                limit 1
                """,
                (conversation_id,),
            ).fetchone()
            prev_turn = conn.execute(
                """
                select id from turns
                where conversation_id = ?
                order by created_at desc
                limit 3
                """,
                (conversation_id,),
            ).fetchall()
            if len(prev_turn) >= 3:
                previous_assistant = prev_turn[-1]["id"]
                conn.execute(
                    """
                    insert into memory_edges (id, conversation_id, from_id, to_id, edge_type, created_at)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (f"edge-{assistant_id}-chron", conversation_id, previous_assistant, user_id, "chronological", created_at),
                )
            conn.execute(
                """
                insert into memory_edges (id, conversation_id, from_id, to_id, edge_type, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (f"edge-{user_id}-reply", conversation_id, user_id, assistant_id, "chronological", created_at),
            )
            if last_memory:
                conn.execute(
                    """
                    insert into memory_edges (id, conversation_id, from_id, to_id, edge_type, created_at)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        f"edge-{last_memory['id']}-{assistant_id}",
                        conversation_id,
                        last_memory["id"],
                        assistant_id,
                        "recalled-into",
                        created_at,
                    ),
                )
                conn.execute(
                    "update memory_objects set last_used_at = ? where id = ?",
                    (created_at, last_memory["id"]),
                )
            conn.execute(
                """
                insert into ledger_events (id, conversation_id, object_id, event_type, payload_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    f"ledger-{assistant_id}",
                    conversation_id,
                    last_memory["id"] if last_memory else None,
                    "chat-round",
                    json.dumps(
                        {
                            "userTurnId": user_id,
                            "assistantTurnId": assistant_id,
                            "timings": timings,
                        }
                    ),
                    created_at,
                ),
            )
        persistence_seconds = time.perf_counter() - persistence_start
        self.runtime.finalize_session_turn(conversation_id=conversation_id, assistant_turn_id=assistant_id)
        log_event(
            f"round persisted assistant_tokens={assistant_tokens} persistence_seconds={persistence_seconds:.3f}",
            request_id=request_id,
        )

        snapshot = self.conversation_snapshot(conversation_id)
        snapshot["assistantText"] = assistant_text
        snapshot["timings"] = {
            **timings,
            "persistenceSeconds": round(persistence_seconds, 4),
            "totalRoundTripSeconds": round(timings["totalModelSeconds"] + persistence_seconds, 4),
        }
        return snapshot

    def reset_conversation(self, conversation_id: str) -> dict:
        created_at = now_iso()
        log_event(f"reset conversation={conversation_id}")
        with self.connect() as conn:
            conn.execute("delete from turns where conversation_id = ?", (conversation_id,))
            conn.execute("delete from memory_objects where conversation_id = ?", (conversation_id,))
            conn.execute("delete from memory_edges where conversation_id = ?", (conversation_id,))
            conn.execute("delete from ledger_events where conversation_id = ?", (conversation_id,))
            conn.execute(
                """
                update conversations
                set current_tokens = 0, updated_at = ?
                where id = ?
                """,
                (created_at, conversation_id),
            )
        self.runtime.invalidate_session(conversation_id)
        return self.conversation_snapshot(conversation_id)

    def _is_visible_chat_turn(self, row: sqlite3.Row) -> bool:
        if row["role"] == "system":
            return False
        if self.runtime._is_preview_artifact(row["text"]):
            return False
        if row["role"] == "assistant" and self.runtime._is_degenerate_response(row["text"]):
            return False
        return True

    def _prompt_turns(self, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        visible = self._stable_visible_turns(rows)
        selected: list[sqlite3.Row] = []
        running_tokens = 0
        budget = max(1024, MAX_TOKENS - PRIMARY_MAX_NEW_TOKENS - 512)
        for row in reversed(visible):
            token_count = int(row["token_count"])
            if selected and (running_tokens + token_count) > budget:
                break
            selected.append(row)
            running_tokens += token_count
        return list(reversed(selected))

    def _stable_visible_turns(self, rows: list[sqlite3.Row]) -> list[sqlite3.Row]:
        stable: list[sqlite3.Row] = []
        i = 0
        while i < len(rows):
            row = rows[i]
            if row["role"] == "system":
                i += 1
                continue
            if row["role"] == "assistant":
                if self._is_visible_chat_turn(row):
                    stable.append(row)
                i += 1
                continue
            if row["role"] != "user":
                i += 1
                continue
            if i + 1 < len(rows) and rows[i + 1]["role"] == "assistant":
                assistant = rows[i + 1]
                stable.append(row)
                if self._is_visible_chat_turn(assistant):
                    stable.append(assistant)
                i += 2
                continue
            stable.append(row)
            i += 1
        return stable

    def _graph_nodes(self, turns: list[sqlite3.Row], memory: sqlite3.Row | None) -> tuple[list[GraphNode], list[dict]]:
        nodes: list[GraphNode] = []
        edges: list[dict] = []
        y = 72
        visible_turns = turns[-5:]
        for row in visible_turns[:-1]:
            label = "Turn"
            snippet = " ".join(row["text"].split())[:96].strip()
            detail = f"{row['role'].title()} · {snippet}" if snippet else f"{row['role'].title()} turn"
            nodes.append(GraphNode(row["id"], label, "turn", 72, y, detail))
            y += 78
        if memory:
            nodes.append(GraphNode(memory["id"], "Memory", "memory", 196, 156, "Compacted memory object"))
        if visible_turns:
            active_row = visible_turns[-1]
            active_detail = active_row["text"][:120].strip() or "Active turn"
            nodes.append(GraphNode(active_row["id"], "Active", "active", 72, max(320, y), active_detail))
        if memory and visible_turns:
            nodes.append(GraphNode(f"recall-{memory['id']}", "Recalled", "recalled", 196, max(364, y + 18), "Memory recalled into the active thread"))

        entity_terms = self._entity_terms(visible_turns)
        entity_y = 86
        for entity_index, (entity_label, turn_ids) in enumerate(entity_terms):
            entity_id = f"entity-{entity_label}"
            x = 206 if entity_index % 2 == 0 else 252
            nodes.append(GraphNode(entity_id, entity_label, "entity", x, entity_y, f"Entity: {entity_label}"))
            for turn_id in turn_ids:
                edges.append({"from": turn_id, "to": entity_id, "type": "mentions"})
            entity_y += 72 if entity_index % 2 else 34
        return nodes, edges

    def _entity_terms(self, turns: list[sqlite3.Row]) -> list[tuple[str, list[str]]]:
        mentions: dict[str, list[str]] = {}
        for row in turns:
            if row["role"] != "user":
                continue
            words = {
                token.lower()
                for token in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", row["text"])
                if token.lower() not in ENTITY_STOPWORDS and len(token) >= 5
            }
            for word in sorted(words):
                mentions.setdefault(word, []).append(row["id"])
        ranked = sorted(mentions.items(), key=lambda item: (-len(item[1]), item[0]))[:6]
        return [(label.title(), turn_ids) for label, turn_ids in ranked]

    def _memory_view(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "kind": row["kind"],
            "tier": row["tier"],
            "bytes": row["byte_size"],
            "turnRange": f"{row['source_turn_start']}..{row['source_turn_end']}",
            "lastUsedLabel": "recently",
        }


RUNTIME = GemmaRuntime(MODEL_NAME, MODEL_DEVICE)
STORE = Store(DB_PATH, RUNTIME)


class Handler(BaseHTTPRequestHandler):
    server_version = "con-chat-sidecar/0.1"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/health":
            status = "ready" if RUNTIME.ready else "error"
            self._send_json(
                200 if RUNTIME.ready else 503,
                {
                    "status": status,
                    "model": MODEL_NAME,
                    "modelDevice": RUNTIME.actual_device,
                    "orchestrationDevice": ORCHESTRATION_DEVICE,
                    "dbPath": str(DB_PATH),
                    "modelLoadSeconds": round(RUNTIME.load_seconds, 4),
                    "error": RUNTIME.error,
                },
            )
            return

        graph_match = re.fullmatch(r"/v1/conversations/([^/]+)/graph", parsed.path)
        if graph_match:
            conversation_id = graph_match.group(1)
            self._send_json(200, STORE.conversation_snapshot(conversation_id))
            return

        self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/chat/stream":
            payload = self._read_json()
            conversation_id = payload.get("conversationId", DEFAULT_CONVERSATION_ID)
            user_text = str(payload.get("userText", "")).strip()
            request_id = f"req-{int(time.time() * 1000)}"
            if not user_text:
                self._send_json(400, {"error": "empty_message"})
                return
            log_event(f"http stream request conversation={conversation_id} chars={len(user_text)}", request_id=request_id)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("cache-control", "no-cache")
            self.send_header("connection", "keep-alive")
            self.end_headers()
            try:
                snapshot = STORE.stream_turns(
                    conversation_id,
                    user_text,
                    on_text=lambda payload: self._send_sse("token", payload),
                    request_id=request_id,
                )
                self._send_sse("done", snapshot)
            except Exception as exc:  # pragma: no cover - runtime failure path
                log_event(f"http stream error error={exc!r}", request_id=request_id)
                self._send_sse("error", {"error": "generation_failed", "detail": repr(exc)})
            return

        if parsed.path == "/v1/chat/respond":
            payload = self._read_json()
            conversation_id = payload.get("conversationId", DEFAULT_CONVERSATION_ID)
            user_text = str(payload.get("userText", "")).strip()
            if not user_text:
                self._send_json(400, {"error": "empty_message"})
                return
            try:
                self._send_json(200, STORE.append_turns(conversation_id, user_text))
            except Exception as exc:  # pragma: no cover - runtime failure path
                self._send_json(500, {"error": "generation_failed", "detail": repr(exc)})
            return

        reset_match = re.fullmatch(r"/v1/conversations/([^/]+)/reset", parsed.path)
        if reset_match:
            payload = self._read_json()
            conversation_id = payload.get("conversationId", reset_match.group(1))
            self._send_json(200, STORE.reset_conversation(conversation_id))
            return

        self._send_json(404, {"error": "not_found"})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("content-length", "0"))
        raw = self.rfile.read(content_length) if content_length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, event: str, payload: dict) -> None:
        body = f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"con-chat sidecar listening on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
