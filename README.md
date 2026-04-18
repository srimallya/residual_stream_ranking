# Residual Stream Ranking

Residual Stream Ranking is a small Python research harness for testing long-context retrieval strategies against a local GGUF model.

The current implementation focuses on a practical sidecar-memory approach:

- keep a short exact recent context window
- compress older windows into checkpoint objects
- retrieve the most relevant checkpoints at query time
- replay those checkpoints back to the model as memory packets
- compare semantic retrieval against a temporal reranking pass

This is intentionally a prototype for ranking and replay behavior. It does not claim exact late-layer residual reconstruction because the current `llama-cpp-python` backend does not expose the per-layer residual deltas needed for that experiment.

## What It Does

The CLI command `residual-lab benchmark` runs a synthetic long-context recall benchmark with four modes:

- `full`: pass the full document to the model
- `recent`: pass only the most recent windows
- `retrieval`: combine recent windows with top-k retrieved historical checkpoints
- `temporal`: apply semantic retrieval plus temporal graph reranking using recent context as the active thread

Each checkpoint stores:

- window text and boundary text
- a semantic payload used today for retrieval
  - window embedding
  - boundary embedding
  - extracted terms
- a trace payload reserved for future exact-reconstruction experiments
  - boundary residual at a layer cutoff
  - selected late-layer deltas

Retrieval uses embedding similarity between the question and each checkpoint. Temporal mode adds a lightweight reranker over checkpoint links derived from shared terms and temporal proximity.

Today only the semantic payload is populated. The trace payload exists to make the separation explicit:

- semantic payload answers: "which past region should I retrieve?"
- trace payload answers: "if the backend exposes the right internals, what exact state pieces could I reconstruct?"

The benchmark also reports two diagnostic metrics:

- target hit rate: whether the target historical window was selected at all
- oracle-correct memory accuracy: answer quality when the correct old checkpoint is injected directly

## Project Layout

```text
src/residual_stream_lab/
  checkpointing.py   checkpoint building and retrieval
  trace.py           trace payload interface and reconstruction helpers
  temporal.py        temporal reranking logic
  synthetic.py       synthetic benchmark generation
  llm.py             local GGUF model runner
  cli.py             Typer CLI entrypoint
tests/
  test_trace.py      trace reconstruction contract tests
models/
  Qwen3.5-2B-Q8_0.gguf
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

Run the benchmark against the bundled model:

```bash
.venv/bin/residual-lab \
  --model-path models/Qwen3.5-2B-Q8_0.gguf \
  --windows 8 \
  --window-lines 6 \
  --recent-windows 2 \
  --top-k 2 \
  --queries 8
```

Useful flags:

- `--model-path`: path to the GGUF model
- `--windows`: total synthetic windows in the generated document
- `--window-lines`: lines per synthetic window
- `--recent-windows`: exact local context horizon
- `--top-k`: number of retrieved checkpoints
- `--queries`: number of benchmark queries
- `--seed`: deterministic benchmark seed
- `--n_ctx`: inference context size

## Interpreting Results

Expected behavior for this prototype:

- `recent` should lose far-past information as the context horizon shrinks
- `retrieval` should improve recall over `recent`
- `temporal` may improve over pure retrieval when the recent thread helps rank older checkpoints
- `full` remains the ceiling for this setup

## Current Limitations

- the benchmark is synthetic, not task-grounded on production traces
- checkpoint replay is a text-memory proxy, not exact residual-state replay
- quality depends heavily on the selected model and embedding behavior
- the bundled GGUF model is a local dependency and is not intended to be committed to git

## Roadmap

Likely next steps if you want to validate the stronger residual-stream idea:

1. Swap to a backend that exposes layerwise residual or hidden-state traces.
2. Populate the trace payload with exact residual-state checkpoint objects.
3. Add evaluation for reconstruction fidelity alongside answer quality.
4. Benchmark on real long-context tasks, not only synthetic recall prompts.

## License

This repository is licensed under the GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
