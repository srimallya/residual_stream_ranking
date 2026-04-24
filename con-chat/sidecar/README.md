# Python Sidecar Contract

The sidecar is the local inference service that powers `con-chat`.

It should stay Python-native, keep Gemma resident, and leave Node/Electron focused on product shell concerns.

## Responsibilities

- load repo-local Gemma
- keep model resident when possible
- execute long forward / generation on `mps` when beneficial
- keep orchestration and bookkeeping cheap and explicit
- expose chat and post-rollover memory operations over a local IPC boundary

## Suggested Runtime Split

- CPU:
  - routing
  - case setup
  - bookkeeping
  - ledger updates
  - report generation
- MPS:
  - Gemma long prefill
  - Gemma decode
  - replay/generation math when the workload is large enough to amortize accelerator overhead

## Required Capabilities

- health probe
- chat response generation
- active-window token accounting
- memory-object compaction
- future memory-object retrieval
- memory ledger updates

See `openapi.yaml` for the first API contract.

## Current State

The first live product slice is now in place:

- Electron main boots this sidecar locally
- the sidecar persists SQLite state
- `/v1/health` is used for readiness
- `/v1/chat/respond` handles a real persisted chat round-trip
- `/v1/chat/stream` streams visible tokens over Server-Sent Events
- `/v1/conversations/{conversationId}/graph` hydrates the left-rail graph and visible thread

The response path is now Gemma-backed:

- the sidecar loads Gemma once at startup
- model residency and load timing are surfaced through `/v1/health`
- `/v1/chat/respond` now:
  - assembles the current thread with `tokenizer.apply_chat_template`
  - passes the special prompt as a real system message
  - reuses KV cache for normal active-window session continuity
  - runs local Gemma generation
  - persists the resulting assistant turn
  - returns timing breakdowns for prompt assembly, generation, persistence, and total round-trip

Memory objects are created only after the active window crosses the 32k boundary. They are reconstructive v1 objects with summary text and optional grouped-int4 KV compression; exact replay validation is not part of the normal chat path.
