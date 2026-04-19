#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PORT = int(os.environ.get("CON_CHAT_PORT", "4318"))
DB_PATH = Path(os.environ.get("CON_CHAT_DB_PATH", "con-chat.sqlite3"))
MODEL_NAME = os.environ.get("CON_CHAT_MODEL_NAME", "google--gemma-4-E2B-it")
MODEL_DEVICE = os.environ.get("CON_CHAT_MODEL_DEVICE", "mps")
ORCHESTRATION_DEVICE = os.environ.get("CON_CHAT_ORCHESTRATION_DEVICE", "cpu")
DEFAULT_CONVERSATION_ID = "demo-thread"
MAX_TOKENS = 32768
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


class GemmaRuntime:
    def __init__(self, model_name: str, preferred_device: str) -> None:
        self.model_name = resolve_model_path(model_name)
        self.preferred_device = preferred_device
        self.actual_device = "cpu"
        self.dtype = torch.float32
        self.tokenizer = None
        self.model = None
        self.lock = threading.Lock()
        self.load_seconds = 0.0
        self.ready = False
        self.error = None
        self._load()

    def _resolve_device(self) -> tuple[str, torch.dtype]:
        if self.preferred_device == "mps" and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _load(self) -> None:
        start = time.perf_counter()
        try:
            device, dtype = self._resolve_device()
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=True,
                torch_dtype=dtype,
            )
            model.to(device)
            model.eval()
            self.actual_device = device
            self.dtype = dtype
            self.tokenizer = tokenizer
            self.model = model
            self.ready = True
        except Exception as exc:  # pragma: no cover - startup failure path
            self.ready = False
            self.error = repr(exc)
        finally:
            self.load_seconds = time.perf_counter() - start

    def count_tokens(self, text: str) -> int:
        if not self.tokenizer:
            return token_estimate(text)
        tokens = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return len(tokens["input_ids"])

    def generate_reply(self, turns: list[sqlite3.Row]) -> tuple[str, dict]:
        if not self.ready or not self.model or not self.tokenizer:
            raise RuntimeError(f"model_not_ready:{self.error}")

        user_text = turns[-1]["text"] if turns else ""
        identity_override = self._identity_override(user_text)
        if identity_override is not None:
            return identity_override, {
                "promptAssemblySeconds": 0.0,
                "generationSeconds": 0.0,
                "totalModelSeconds": 0.0,
                "promptTokens": 0,
                "completionTokens": self.count_tokens(identity_override),
                "modelDevice": self.actual_device,
            }

        prompt_start = time.perf_counter()
        prompt = self._format_prompt(turns)
        model_inputs = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_seconds = time.perf_counter() - prompt_start

        generation_start = time.perf_counter()
        with self.lock, torch.inference_mode():
            inputs = model_inputs.to(self.actual_device)
            outputs = self.model.generate(
                inputs,
                max_new_tokens=160,
                do_sample=True,
                temperature=0.68,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.12,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generation_seconds = time.perf_counter() - generation_start

        prompt_tokens = int(model_inputs.shape[-1])
        generated_ids = outputs[0, prompt_tokens:]
        assistant_text = self._clean_response(
            self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        )
        completion_tokens = int(generated_ids.shape[-1])

        if self._is_degenerate_response(assistant_text):
            retry_start = time.perf_counter()
            with self.lock, torch.inference_mode():
                retry_outputs = self.model.generate(
                    inputs,
                    max_new_tokens=160,
                    do_sample=True,
                    temperature=0.82,
                    top_p=0.92,
                    top_k=48,
                    repetition_penalty=1.16,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generation_seconds += time.perf_counter() - retry_start
            generated_ids = retry_outputs[0, prompt_tokens:]
            assistant_text = self._clean_response(
                self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            completion_tokens = int(generated_ids.shape[-1])

        if self._is_degenerate_response(assistant_text):
            fallback_prompt = self._format_fallback_prompt(turns[-1]["text"])
            fallback_inputs = self.tokenizer(fallback_prompt, return_tensors="pt").input_ids
            retry_start = time.perf_counter()
            with self.lock, torch.inference_mode():
                fallback_outputs = self.model.generate(
                    fallback_inputs.to(self.actual_device),
                    max_new_tokens=120,
                    do_sample=False,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            generation_seconds += time.perf_counter() - retry_start
            fallback_prompt_tokens = int(fallback_inputs.shape[-1])
            generated_ids = fallback_outputs[0, fallback_prompt_tokens:]
            assistant_text = self._clean_response(
                self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            )
            completion_tokens = int(generated_ids.shape[-1])

        if not assistant_text or self._is_degenerate_response(assistant_text):
            assistant_text = self._safe_fallback_reply(turns[-1]["text"] if turns else "")
            completion_tokens = self.count_tokens(assistant_text)

        return assistant_text, {
            "promptAssemblySeconds": round(prompt_seconds, 4),
            "generationSeconds": round(generation_seconds, 4),
            "totalModelSeconds": round(prompt_seconds + generation_seconds, 4),
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "modelDevice": self.actual_device,
        }

    def _format_prompt(self, turns: list[sqlite3.Row]) -> str:
        chunks: list[str] = [
            "<start_of_turn>user",
            SYSTEM_PROMPT,
            "<end_of_turn>",
        ]
        for row in turns:
            role = row["role"]
            text = row["text"]
            if role == "system":
                continue
            if self._is_preview_artifact(text):
                continue
            if role == "user":
                chunks.extend(["<start_of_turn>user", text, "<end_of_turn>"])
            elif role == "assistant":
                chunks.extend(["<start_of_turn>model", text, "<end_of_turn>"])
        chunks.append("<start_of_turn>model")
        return "\n".join(chunks)

    def _format_fallback_prompt(self, user_text: str) -> str:
        return "\n".join(
            [
                "<start_of_turn>user",
                SYSTEM_PROMPT,
                "Answer the user's question in 2 to 4 direct sentences.",
                "Do not mention Google or a model vendor unless asked directly.",
                "Do not repeat filler. Do not echo the prompt.",
                "Do not show thinking, reasoning steps, headings, bullet plans, or symbolic scaffolding.",
                "Do not number steps. Do not write labels like Analyze, Boundaries, Relations, or Answer.",
                "Start directly with the answer sentence.",
                "Return the final answer only.",
                "<end_of_turn>",
                "<start_of_turn>user",
                user_text,
                "<end_of_turn>",
                "<start_of_turn>model",
            ]
        )

    def _clean_response(self, text: str) -> str:
        cleaned = text.strip()
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
        if "\nthought\n" in cleaned.lower():
            cleaned = cleaned.split("\n", 1)[-1].strip()
        if "Model needs to continue" in cleaned:
            cleaned = cleaned.split("Model needs to continue", 1)[0].strip()
        while cleaned.startswith("Assistant:"):
            cleaned = cleaned[len("Assistant:") :].strip()
        cleaned = cleaned.replace("\nAssistant:", "\n").strip()
        answer_match = re.search(r"\bANSWER\b\s*:?\s*(.+)$", cleaned, flags=re.IGNORECASE | re.DOTALL)
        if answer_match:
            cleaned = answer_match.group(1).strip()
        cleaned = re.sub(r"^\s*why\s+.+?\?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _is_preview_artifact(self, text: str) -> bool:
        return text.startswith("Browser preview mode is active")

    def _is_degenerate_response(self, text: str) -> bool:
        normalized = " ".join(text.strip().split())
        if not normalized:
            return True
        lowered = normalized.lower()
        if "<end" in lowered or "<start" in lowered:
            return True
        if re.search(r"mid[_-]of[_-]turn", lowered):
            return True
        if "model needs to continue" in lowered:
            return True
        if lowered.startswith("thought ") or lowered.startswith("thought\n"):
            return True
        if "thinking process" in lowered or "model response" in lowered:
            return True
        if "boundaries" in lowered and "relations" in lowered:
            return True
        if "analyze the request" in lowered or "self-correction" in lowered:
            return True
        if "identify key boundaries" in lowered or "symbol creation" in lowered:
            return True
        if "possible interpretations" in lowered:
            return True
        if "developed by **google**" in lowered or "developed by google" in lowered:
            return True
        repeated_token = re.fullmatch(r"(\b\w+\b)(?:\s+\1){10,}", normalized, flags=re.IGNORECASE)
        if repeated_token:
            return True
        if re.search(r"\b([A-Za-z])(?:\s+\1){8,}\b", normalized):
            return True
        tokens = re.findall(r"[A-Za-z0-9']+", normalized.lower())
        if len(tokens) < 5:
            return True
        if len(tokens) >= 12:
            counts = Counter(tokens)
            token, frequency = counts.most_common(1)[0]
            if len(token) <= 3 and (frequency / len(tokens)) >= 0.45:
                return True
            if len(set(tokens[-12:])) <= 2:
                return True
            if len(set(tokens)) <= 3 and len(tokens) >= 18:
                return True
        if normalized.endswith("..."):
            return True
        return False

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
        synthetic_user_row = {
            "id": user_id,
            "role": "user",
            "text": user_text,
        }
        assistant_text, timings = self.runtime.generate_reply([*self._prompt_turns(prior_turns), synthetic_user_row])
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
        return visible[-12:]

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
                if self._is_visible_chat_turn(assistant):
                    stable.extend([row, assistant])
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


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"con-chat sidecar listening on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
