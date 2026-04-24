# con-chat

`con-chat` is a local macOS chat app where the user sees one lifetime conversation. Inside the active context window it behaves like normal Gemma chat: the sidecar passes a real system message plus normal user/assistant turns through the tokenizer's official chat template.

## Product Shape

- macOS desktop app built with Electron
- local Python inference sidecar for Gemma + replay logic
- one continuous conversation from the user's point of view
- bounded active context window that rolls older stabilized turns into reconstructive memory objects only after the window boundary is reached
- collapsible left rail that visualizes the conversation and memory graph as it grows

## Runtime Split

The app should not run the model stack inside Node.

- Electron main/renderer:
  - UI
  - persistence
  - installer / OS integration
  - lifecycle and reporting
- Python sidecar:
  - Gemma loading
  - MPS execution
  - tokenization
- post-boundary memory-object creation
- future retrieval / routing
  - ledger updates

That split keeps the product shell stable while reusing the existing replay work instead of rewriting it into JavaScript.

## Context Protocol

For the v1 app:

- the user stays in one visible thread
- the active thread grows turn-by-turn until it exceeds the `32k` active window
- below that boundary, no residual memory is injected and no memory objects are created
- older stable regions are compacted after rollover into reconstructive memory objects
- v1 memory stores summary text, anchors, token accounting, and an optional TurboQuant-style compressed KV payload
- residual/replay memory is a future advanced memory-object type, not required for normal chat
- there is no retry-based exact-answer validation

## Packaging

The intended shipping target is a signed macOS `.dmg` that bundles:

- Electron app shell
- local sidecar code
- Gemma model snapshot
- replay/runtime assets

The current `electron-builder.yml` is a starting point, not the final notarized pipeline.

## Current Scaffold

This directory currently contains:

- `app/`: Electron shell and renderer UI
- `sidecar/`: Python sidecar contract and packaging notes
- `docs/`: product architecture notes for this app

The scaffold is intentionally thin but already executable as a product boundary:

- Electron now boots a real local Python sidecar
- the sidecar persists turns, memory objects, graph edges, ledger events, and settings in SQLite
- the renderer bootstraps from that persisted state instead of hardcoded demo messages
- one real end-to-end chat turn now flows:
  - renderer -> Electron main -> Python sidecar -> SQLite -> Electron main -> renderer
- the sidecar loads Gemma once, keeps it resident, assembles prompts with `tokenizer.apply_chat_template`, uses KV cache for active-session continuity, and streams real tokens over SSE

The next slice is improving post-rollover memory retrieval. Normal chat should remain a plain Gemma chat-template path inside the active window.
