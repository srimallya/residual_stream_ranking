# con-chat

`con-chat` is the product shell for the replay protocol in this repository: a local macOS chat app where the conversation feels continuous, while the live model state stays bounded.

## Product Shape

- macOS desktop app built with Electron
- local Python inference sidecar for Gemma + replay logic
- one continuous conversation from the user's point of view
- bounded active context window that rolls older stabilized turns into memory objects
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
  - replay-object creation
  - retrieval / routing
  - ledger updates

That split keeps the product shell stable while reusing the existing replay work instead of rewriting it into JavaScript.

## Context Protocol

The target protocol is:

`live_state + recalled_state + current_input -> next_state`

For the v1 app:

- the user stays in one visible thread
- the active thread grows turn-by-turn until it approaches `32k` tokens
- older stabilized regions are compacted into memory objects
- the default compact object is replay-token `fp16`
- richer replay or text fallback is used only when necessary

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
- the sidecar now loads Gemma once, keeps it resident, and serves real local generation instead of a deterministic stub

The next slice is bounded active-window assembly: keep Gemma generation on the current thread, then start sealing older stable ranges into replay objects at the `32k` boundary.
