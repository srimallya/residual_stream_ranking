#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PORT = int(os.environ.get("CON_CHAT_PORT", "4318"))
DB_PATH = Path(os.environ.get("CON_CHAT_DB_PATH", "con-chat.sqlite3"))
MODEL_NAME = os.environ.get("CON_CHAT_MODEL_NAME", "google--gemma-4-E2B-it")
MODEL_DEVICE = os.environ.get("CON_CHAT_MODEL_DEVICE", "mps")
ORCHESTRATION_DEVICE = os.environ.get("CON_CHAT_ORCHESTRATION_DEVICE", "cpu")
DEFAULT_CONVERSATION_ID = "demo-thread"
MAX_TOKENS = 32768


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


class Store:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
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
                    21184,
                    MAX_TOKENS,
                    created_at,
                    created_at,
                ),
            )

            seed_turns = [
                (
                    "turn-0001",
                    "user",
                    "Keep the conversation continuous, but stop dragging the whole past as live context.",
                ),
                (
                    "turn-0002",
                    "assistant",
                    "I’ll keep the active thread bounded, seal older stable regions into replayable memory objects, and recall them only when they matter.",
                ),
                (
                    "turn-0003",
                    "system",
                    "Turns 18-25 compacted into token@34/fp16 and linked into the graph.",
                ),
            ]
            for turn_id, role, text in seed_turns:
                conn.execute(
                    """
                    insert into turns (id, conversation_id, role, text, token_count, created_at)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (turn_id, DEFAULT_CONVERSATION_ID, role, text, token_estimate(text), created_at),
                )

            conn.execute(
                """
                insert into memory_objects (
                  id, conversation_id, kind, tier, byte_size, source_turn_start, source_turn_end, summary, last_used_at, created_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "memory-0001",
                    DEFAULT_CONVERSATION_ID,
                    "token@34/fp16",
                    "warm",
                    1536,
                    "turn-0018",
                    "turn-0025",
                    "Compacted replay-token memory for a stabilized prior exchange.",
                    created_at,
                    created_at,
                ),
            )

            seed_edges = [
                ("edge-0001", "turn-0001", "turn-0002", "chronological"),
                ("edge-0002", "turn-0002", "turn-0003", "chronological"),
                ("edge-0003", "turn-0003", "memory-0001", "compressed-into"),
                ("edge-0004", "memory-0001", "turn-0003", "recalled-into"),
            ]
            for edge_id, from_id, to_id, edge_type in seed_edges:
                conn.execute(
                    """
                    insert into memory_edges (id, conversation_id, from_id, to_id, edge_type, created_at)
                    values (?, ?, ?, ?, ?, ?)
                    """,
                    (edge_id, DEFAULT_CONVERSATION_ID, from_id, to_id, edge_type, created_at),
                )

            conn.execute(
                """
                insert into ledger_events (id, conversation_id, object_id, event_type, payload_json, created_at)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    "ledger-0001",
                    DEFAULT_CONVERSATION_ID,
                    "memory-0001",
                    "memory-created",
                    json.dumps({"kind": "token@34/fp16", "tier": "warm"}),
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
            for row in turns
        ]

        nodes = self._graph_nodes(turns, memory)
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
                ],
            },
        }

    def append_turns(self, conversation_id: str, user_text: str) -> dict:
        created_at = now_iso()
        user_id = f"turn-{int(datetime.now().timestamp() * 1000)}-u"
        assistant_id = f"turn-{int(datetime.now().timestamp() * 1000)}-a"
        user_tokens = token_estimate(user_text)
        assistant_text = self._assistant_reply(user_text)
        assistant_tokens = token_estimate(assistant_text)

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
                    json.dumps({"userTurnId": user_id, "assistantTurnId": assistant_id}),
                    created_at,
                ),
            )

        snapshot = self.conversation_snapshot(conversation_id)
        snapshot["assistantText"] = assistant_text
        return snapshot

    def _graph_nodes(self, turns: list[sqlite3.Row], memory: sqlite3.Row | None) -> list[GraphNode]:
        nodes: list[GraphNode] = []
        y = 72
        visible_turns = turns[-5:]
        for row in visible_turns[:-1]:
            nodes.append(GraphNode(row["id"], row["id"].replace("turn-", "Turn "), "turn", 72, y))
            y += 78
        if memory:
            nodes.append(GraphNode(memory["id"], "Memory", "memory", 196, 156))
        if visible_turns:
            nodes.append(GraphNode(visible_turns[-1]["id"], "Active", "active", 72, max(320, y)))
        if memory and visible_turns:
            nodes.append(GraphNode(f"recall-{memory['id']}", "Recalled", "recalled", 196, max(364, y + 18)))
        return nodes

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

    def _assistant_reply(self, user_text: str) -> str:
        return (
            "The local sidecar received your turn, stored it in SQLite, and kept the bounded-thread model intact. "
            "The next slice will replace this deterministic reply with real Gemma generation and replay-object rollout."
        )


STORE = Store(DB_PATH)


class Handler(BaseHTTPRequestHandler):
    server_version = "con-chat-sidecar/0.1"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/v1/health":
            self._send_json(
                200,
                {
                    "status": "ready",
                    "model": MODEL_NAME,
                    "modelDevice": MODEL_DEVICE,
                    "orchestrationDevice": ORCHESTRATION_DEVICE,
                    "dbPath": str(DB_PATH),
                },
            )
            return

        if parsed.path == f"/v1/conversations/{DEFAULT_CONVERSATION_ID}/graph":
            self._send_json(200, STORE.conversation_snapshot(DEFAULT_CONVERSATION_ID))
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
            self._send_json(200, STORE.append_turns(conversation_id, user_text))
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
