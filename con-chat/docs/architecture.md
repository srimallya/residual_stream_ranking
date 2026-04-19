# con-chat Architecture

## V1 Goals

- Ship a local macOS chat app with a bundled Gemma model.
- Keep the user's conversation visually continuous.
- Prevent unbounded KV-cache growth by sealing older context into replayable memory objects.
- Make memory visible through a graph in the collapsible left rail.

## Core Components

### Electron Main Process

Owns:

- app lifecycle
- BrowserWindow creation
- sidecar launch and health checks
- file-system persistence
- IPC between renderer and sidecar
- packaging integration

### Renderer

Owns:

- chat transcript
- composer
- graph rail
- memory inspector
- runtime status and token budget

### Python Sidecar

Owns:

- Gemma model residency
- MPS model execution
- active-window assembly
- replay object creation
- retrieval/routing
- memory ledger updates

Current implemented slice:

- local HTTP sidecar process
- SQLite persistence for:
  - conversations
  - turns
  - memory objects
  - graph edges
  - ledger events
  - settings
- real end-to-end chat round-trip through Electron -> sidecar -> SQLite -> renderer
- resident Gemma generation inside the sidecar, with health and timing surfaced separately from orchestration

## Active Conversation Policy

- Active window target: `32k` tokens
- The user experiences one visible thread
- Older stable ranges are sealed into memory objects
- Default memory object: replay-token `fp16`
- Escalation path:
  - exact replay token
  - richer local replay band
  - text fallback

## Memory Graph

Node types:

- turn node
- active segment node
- memory object node

Edge types:

- chronological
- compressed-into
- recalled-into
- topical / project grouping

The default graph view should stay legible:

- active thread centered
- memory objects grouped in the rail
- recalled edges highlighted only when used

## Packaging Direction

The intended app experience is "install and start chatting", which implies a heavy but simple `.dmg`.

Bundle targets:

- Electron app
- sidecar code
- Gemma model
- local runtime artifacts

Operational note:

- keep CPU orchestration and UI logic in Electron / sidecar control flow
- let MPS do the dense Gemma model pass
- do not assume moving the whole harness to `mps` helps
