# Residual Stream Ranking

Residual Stream Ranking is a small Python research harness for testing long-context retrieval strategies against a local GGUF model.

The current system uses a practical sidecar-memory design:

- keep a short exact recent context window
- compress older windows into checkpoint objects
- retrieve a semantic candidate pool
- rerank that pool with a staged retrieval stack
- replay the selected checkpoints back to the model as memory packets
- optionally expand the selected checkpoint with a tiny local neighborhood

This is intentionally a prototype for ranking and replay behavior. It does not claim exact late-layer residual reconstruction because the current `llama-cpp-python` backend does not expose the per-layer residual deltas needed for that experiment.

## Current State

The project now has a valid corpus-backed result on the Apollo benchmark with `Qwen3.5-2B-Q8_0.gguf`.

On a valid 12-case Apollo run:

- `recent`: `0.00` accuracy
- `retrieval`: `0.17` accuracy
- `temporal` with staged reranking: `0.42` accuracy
- `temporal_expanded` with one neighbor window: `0.58` accuracy
- `full`: `0.75` accuracy
- `oracle`: `1.00` accuracy

The current interpretation is:

- staged reranking largely fixes the ranking problem
- local expansion improves downstream memory sufficiency
- the remaining gap to `full` and `oracle` is now a real replay-sufficiency / model-use gap
- parse/answer-channel instability is no longer the dominant source of error on the main Apollo path

## What It Does

The CLI exposes three useful evaluation paths:

- `residual-lab benchmark`: synthetic long-context recall benchmark
- `residual-lab benchmark-apollo`: corpus-backed Apollo benchmark with parse-valid full/oracle gating
- `residual-lab route-apollo`: routing-only Apollo comparison for reranker ablations

The benchmark modes are:

- `full`: pass the full document to the model
- `recent`: pass only the most recent windows
- `retrieval`: combine recent windows with top-k retrieved historical checkpoints
- `temporal`: semantic pool selection plus staged reranking
- `temporal_expanded`: staged reranking plus a tiny local neighborhood around the selected memory

Each checkpoint stores:

- window text and boundary text
- a semantic payload used today for retrieval
  - window embedding
  - boundary embedding
  - extracted terms
  - extracted anchors
- a trace payload reserved for future exact-reconstruction experiments
  - boundary residual at a layer cutoff
  - selected late-layer deltas

Today only the semantic payload is populated. The trace payload exists to make the separation explicit:

- semantic payload answers: "which past region should I retrieve?"
- trace payload answers: "if the backend exposes the right internals, what exact state pieces could I reconstruct?"

The stronger retrieval path is now a staged stack:

1. semantic pool selection
2. temporal / graph reranking inside that pool
3. local anchor-aware refinement

An optional expansion step can then add one local neighbor window around the selected checkpoint to test replay sufficiency.

The Apollo evaluator reports routing and use diagnostics:

- target hit rate
- oracle-correct memory accuracy
- top-1 / top-2 / top-4 recall
- mean reciprocal rank (MRR)
- parse given hit
- correct given hit and parse
- wrong window rate
- staged-vs-oracle gap rows for failed cases

## Project Layout

```text
src/residual_stream_lab/
  checkpointing.py   checkpoint building and retrieval
  trace.py           trace payload interface and reconstruction helpers
  temporal.py        temporal reranking logic
  apollo.py          corpus-backed Apollo benchmark case generation
  synthetic.py       synthetic benchmark generation
  llm.py             local GGUF model runner
  cli.py             Typer CLI entrypoint
tests/
  test_trace.py      trace reconstruction contract tests
models/
  Qwen3.5-2B-GGUF/
  Qwen3.5-4B-GGUF/
data/
  apollo11_clean.txt
```

## Requirements

- Python 3.10 or newer
- a local GGUF model file
- a working C/C++ toolchain if `llama-cpp-python` needs to build locally

## Installation

```bash
uv venv .venv
uv pip install --python .venv/bin/python -e .
```

If you are not using `uv`, you can install the package with standard `pip` inside any Python 3.10+ virtual environment.

## Usage

Run the synthetic benchmark:

```bash
.venv/bin/residual-lab benchmark \
  --model-path models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --windows 8 \
  --window-lines 6 \
  --recent-windows 2 \
  --top-k 2 \
  --queries 8
```

Run the Apollo corpus benchmark:

```bash
.venv/bin/residual-lab benchmark-apollo \
  --model-path models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --corpus-path data/apollo11_clean.txt \
  --case-count 12 \
  --windows 12 \
  --window-lines 24 \
  --recent-windows 2 \
  --top-k 2 \
  --n-ctx 4096
```

Run the stronger staged Apollo path:

```bash
.venv/bin/residual-lab benchmark-apollo \
  --model-path models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --corpus-path data/apollo11_clean.txt \
  --case-count 12 \
  --windows 12 \
  --window-lines 24 \
  --recent-windows 2 \
  --top-k 2 \
  --n-ctx 4096 \
  --rerank-strategy staged
```

Run the staged local-expansion ablation:

```bash
.venv/bin/residual-lab benchmark-apollo \
  --model-path models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --corpus-path data/apollo11_clean.txt \
  --case-count 12 \
  --windows 12 \
  --window-lines 24 \
  --recent-windows 2 \
  --top-k 2 \
  --n-ctx 4096 \
  --rerank-strategy staged \
  --local-expansion-neighbors 1
```

Run the Apollo routing-only ablation:

```bash
.venv/bin/residual-lab route-apollo \
  --model-path models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --corpus-path data/apollo11_clean.txt \
  --case-count 6 \
  --windows 12 \
  --window-lines 24 \
  --recent-windows 2 \
  --n-ctx 4096
```

Useful flags:

- `--model-path`: path to the GGUF model
- `--windows`: total windows in the generated document or corpus slice
- `--window-lines`: lines per window
- `--recent-windows`: exact local context horizon
- `--top-k`: number of retrieved checkpoints
- `--queries`: number of synthetic benchmark queries
- `--seed`: deterministic benchmark seed
- `--n-ctx`: inference context size
- `--rerank-strategy`: retrieval strategy for temporal mode, including `staged`
- `--local-expansion-neighbors`: neighbor radius for the `temporal_expanded` ablation

Verify offline hidden-state reconstruction with a Hugging Face backend:

```bash
.venv/bin/residual-lab trace-verify \
  --model-name-or-path gpt2 \
  --prompt "The capital of France is" \
  --layer-cutoff-b 6 \
  --delta-layers 7,8,9,10,11
```

Verify phase 2A resumed execution for GPT-2-class decoder models:

```bash
.venv/bin/residual-lab trace-resume-verify \
  --model-name-or-path gpt2 \
  --prompt "The capital of France is" \
  --boundary-layer 6
```

Probe compact replay objects against the exact replay baseline:

```bash
.venv/bin/residual-lab trace-compact-sweep \
  --model-name-or-path models/gpt2 \
  --prompt "The capital of France is" \
  --boundary-layer 6 \
  --replay-layer 10 \
  --delta-depths 0,1,2,4
```

`trace-verify` captures observed layer outputs for one token, stores the boundary state plus incremental deltas, and checks offline reconstruction. `trace-resume-verify` takes the next narrow step: inject a captured full-sequence boundary hidden state at one layer, run the remaining layers, and compare resumed logits against the direct pass for one target token.
`trace-compact-sweep` asks a different question: if exact prefix states are kept, how many late-layer deltas for the target token must remain in the replay object before next-token behavior lines up with the exact replay baseline again?

The compact replay branch now has a dedicated frontier table in [docs/compact_frontier.md](docs/compact_frontier.md), which separates one-step quality from continuation stability.

Bridge routed Apollo selection into tracked replay evaluation:

```bash
.venv/bin/python -m residual_stream_lab.cli bridge-apollo-replay \
  --model-path models/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf \
  --hf-model-name-or-path models/gpt2 \
  --case-count 6 \
  --top-k 4 \
  --replay-boundary-layer 6 \
  --replay-layer 10 \
  --replay-steps 10
```

`bridge-apollo-replay` keeps the staged Apollo router (`semantic pool -> temporal/PageRank rerank -> graph-local refinement`) and, on routed top-1 hits, scores tracked replay objects alongside a plain `text@window` control on the selected region. The command now also emits a read-only memory ledger report with tier counts, the low-utility tail, and reporting-only archive/prune suggestions for the replay objects it observed.

## Interpreting Results

Expected behavior for the current prototype:

- `recent` should lose far-past information as the context horizon shrinks
- `retrieval` should improve over `recent`, but may still fail from poor ranking
- `temporal` should improve materially over plain retrieval when staged reranking is enabled
- `temporal_expanded` may improve over `temporal` when the remaining gap is local memory sufficiency
- `full` remains the main control path
- `oracle` remains the replay ceiling

The current benchmark evidence supports this decomposition:

- routing used to be the dominant bottleneck
- staged reranking fixes a large part of that routing loss
- local expansion helps on harder cases, especially the medium-distance bucket
- the remaining gap is now mostly replay sufficiency / model use after correct retrieval

## Current Limitations

- checkpoint replay is a text-memory proxy, not exact residual-state replay
- the Apollo benchmark is still a constructed harness, not a production task suite
- quality depends heavily on the selected model and embedding behavior
- true hidden-state injection is currently implemented only for a narrow GPT-2-class phase 2A verification path
- compact replay is currently a narrow target-token experiment, not yet a general continuation path
- the memory ledger is currently observability-only; it reports tier suggestions but does not demote, archive, or prune anything yet
- the bundled GGUF models are local dependencies and are not intended to be committed to git

## Roadmap

Likely next steps from the current state:

1. Turn the new memory ledger into an actual lifecycle layer with conservative warm/cold/archive transitions.
2. Stress `token@10/fp16` on harder routed slices before trying smarter lossy replay-token codecs.
3. Run context-budget ablations such as `4096` vs `8192`.
4. Broaden the HF trace backend beyond the current verification path and harden resumed-forward agreement checks.
5. Add next-token continuation after resumed execution.
6. Compare semantic replay, offline trace reconstruction, and resumed execution on the same cases.

## License

This repository is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
