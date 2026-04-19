# Findings Log

## 2026-04-19

### Residual Replay Milestones

- Phase 1 is now real on the Hugging Face path: the repo captures true layerwise hidden states and verifies offline reconstruction from a boundary state plus stored deltas.
- Phase 2A is validated for the GPT-2-class Hugging Face backend: resumed forward from a captured boundary hidden state reproduces direct logits exactly when the resumed path uses the same causal-mask and position-id contract as the direct model forward.
- Phase 2B is validated for the same path: resumed next-token logits, greedy next token, and top-k next-token ranking match the direct path exactly.
- Phase 2C is validated for the same path: short-horizon greedy continuation remains exact across the tested sweep.

### Important Failure and Fix

- The first phase 2A implementation was wrong for early and mid boundaries even though cosine stayed high.
- Observed failure pattern before the fix:
  - boundary layer 0: large L2 and max-abs logit error
  - boundary layer 6: still wrong
  - boundary layer 10: exact
- Root cause: the resumed path was not using GPT-2's actual causal-mask contract. It passed a padding-style mask instead of the direct forward path's `create_causal_mask(...)` behavior and explicit `position_ids`.
- After switching the resumed path to the same causal-mask and position-id semantics as direct forward, exact agreement held across tested boundaries.

### Verified Runtime Results

Prompt:

```text
The capital of France is
```

Validated exact agreement for boundary layers:

- `0`
- `6`
- `10`

Validated exact continuation horizons:

- `5`
- `10`
- `20`
- `50`

Observed result across the tested grid:

- resumed forward: exact
- next-token prediction: exact
- greedy continuation: exact
- first divergence: none observed

Representative exact metrics:

- `L2 error = 0.0`
- `cosine similarity = 1.0`
- `max_abs_diff = 0.0`

### Current Scope

What is validated:

- exact trace capture
- exact resumed forward agreement
- exact next-token agreement
- exact greedy continuation for the current GPT-2-class Hugging Face path

What is not validated yet:

- KV-aware continuation
- broader model-family support
- compact replay objects beyond full boundary hidden state
- long-horizon continuation beyond the current tested grid

### Current Baseline

The repo now has a trustworthy correctness baseline for the narrow GPT-2-class replay path:

- lower stack is recomputed up to the chosen boundary
- upper stack is resumed from the captured boundary hidden state
- direct and resumed execution agree exactly on the tested grid

This baseline should be preserved when adding KV-aware continuation or broader replay experiments.

### KV-Aware Continuation: First Pass

- A narrow KV-aware upper-stack continuation path was added for the GPT-2-class Hugging Face backend.
- Comparison target: the frozen exact replay baseline (`recompute lower stack` + `resume upper stack`).
- Measured axes from the start:
  - correctness
  - first divergence step
  - wall-clock time
  - cache footprint

Observed behavior on prompt:

```text
The capital of France is
```

Tested boundary layers:

- `0`
- `6`
- `10`

Tested horizon:

- `5`

Observed result across the tested KV grid:

- step 1: exact
- step 2: first numerical divergence
- greedy token still matched at step 2
- exact logit equality did not hold past step 1

Representative divergence at step 2:

- `L2 ~= 0.01`
- `max_abs_diff ~= 2.2e-4`
- cosine remained effectively `1.0`
- greedy token still matched

Representative timing and cache behavior:

- boundary `0`: baseline `~291.5 ms`, KV path `~30.5 ms`, cache `405,504` bytes
- boundary `6`: baseline `~217.7 ms`, KV path `~16.9 ms`, cache `184,320` bytes
- boundary `10`: baseline `~292.7 ms`, KV path `~10.8 ms`, cache `36,864` bytes

Current interpretation:

- the KV-aware path is already much faster than the frozen exact baseline
- it is not yet exact enough to replace that baseline
- the first divergence appearing consistently at step 2 suggests a stable cache-contract or numerical-path mismatch rather than random drift

Current status:

- frozen exact baseline: trusted
- KV-aware branch: promising, faster, but not yet execution-faithful enough

### KV Diagnosis: First Divergent Layer

Step-2 diagnosis on prompt:

```text
The capital of France is
```

Boundary tested:

- `6`

Observed result:

- first divergent layer: `7`
- this is the first upper-stack layer after the replay boundary
- divergence is already present at that first cached upper block and then compounds across later upper layers

Representative per-layer step-2 state diff:

- layer `7`: `L2 ~= 2.766e-05`, `max_abs_diff ~= 1.144e-05`
- layer `8`: `L2 ~= 6.038e-05`
- layer `9`: `L2 ~= 8.671e-05`
- layer `10`: `L2 ~= 1.054e-04`
- layer `11`: `L2 ~= 1.594e-04`

Interpretation:

- the first cache reuse already disagrees slightly with the frozen exact baseline
- the bug is therefore not "later drift only"
- the most likely remaining issue is a cache read/write contract mismatch at the first reused upper block, rather than a problem in the downstream LM head

### KV Diagnosis: Layer-7 Substage Cut

For step 2 at boundary `6`, the first reused upper block (`layer 7`) was cut into substages.

Observed exact-vs-KV comparisons at `layer 7`:

- input hidden state: exact
- `ln_1` output: exact
- query projection (`Q`): exact
- new key projection (`K_new`): exact
- new value projection (`V_new`): first nonzero difference
- attention output: nonzero difference inherited from the earlier `V` mismatch
- post-attention state: nonzero difference
- MLP output: larger nonzero difference
- post-block hidden state: larger nonzero difference

Representative numbers:

- `V_new` diff: `L2 ~= 5.52e-06`, `max_abs_diff ~= 1.19e-06`
- attention output diff: `L2 ~= 1.23e-05`
- post-block hidden diff: `L2 ~= 2.77e-05`, `max_abs_diff ~= 1.14e-05`

Additional cache comparison:

- prefill cache at layer 7 matches exact prefill `K/V` exactly
- after step-2 append, combined cached `K/V` no longer match the exact full-sequence `K/V`

Interpretation:

- the first visible bug is not in the boundary hidden state, layernorm, or query projection
- the first nonzero difference appears in the step-2 value path at the first reused upper block
- the cache update/read contract for the reused `V` path is now the highest-value place to inspect next

### KV Diagnosis: Reference Path Is Not Prefix-Bit-Stable

Further diagnosis showed that the frozen exact reference path is itself not bit-identical on the old prefix when the next token is appended and the whole sequence is recomputed.

Observed on boundary layer `6` for prompt:

```text
The capital of France is
```

Comparing:

- original prompt prefix
- step-2 full-sequence recompute, restricted back to the old prefix positions

Observed differences:

- boundary layer 6 prefix hidden state:
  - `L2 ~= 6.70e-05`
  - `max_abs_diff ~= 3.05e-05`
- layer-7 `ln_1` prefix:
  - `L2 ~= 6.54e-06`
- layer-7 fused projection prefix:
  - `Q`: `L2 ~= 3.23e-05`
  - `K`: `L2 ~= 3.34e-05`
  - `V`: `L2 ~= 2.53e-05`

Interpretation:

- the "exact" full-sequence recompute path is sequence-length-dependent at the bit level, even on old prefix positions
- therefore, not all step-2 KV mismatch should be interpreted as a pure cache bug
- at least part of the observed KV drift is inherited from the reference path itself when comparing:
  - cached old prefix states from step 1
  - recomputed old prefix states from the longer step-2 sequence

Updated read:

- there is still a meaningful mismatch at the first reused upper block
- but the comparison target is no longer "prefix positions are bit-stable under full recompute"
- the next diagnostic task is to separate:
  - unavoidable reference-path recompute drift
  - from true cache-contract drift introduced by KV reuse

### KV Diagnosis: Three-Path Separation

Three-path step-2 comparison for boundary `6`:

- Path A: frozen exact resumed baseline
- Path B: full-sequence recompute comparator
- Path C: KV-aware path

Observed result on the tested cut:

- Path A and Path B match exactly at the logit level
- Path A and Path B also match exactly across all tested upper layers (`7` to `11`)
- all observed step-2 drift sits in Path C relative to both A and B

Representative numbers:

- `A vs B` logits:
  - `L2 = 0.0`
  - `max_abs_diff = 0.0`
- `A vs C` logits:
  - `L2 ~= 0.010388`
  - `max_abs_diff ~= 2.2125e-04`
- `B vs C` logits:
  - same as `A vs C`

Per-layer result:

- first `A vs C` divergent layer: `7`
- first `A vs B` divergent layer: none

Interpretation:

- for the tested step-2 cut, the full-sequence recompute path is not the source of the observed upper-stack drift
- the current measurable error term is KV-specific on this cut
- the highest-value remaining bug boundary is therefore still the first reused upper block, with the value path as the first observed nonzero mismatch
