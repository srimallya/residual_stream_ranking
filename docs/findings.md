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

### KV Diagnosis: The First KV-Specific Mismatch Is Before Cache Append

At step 2, layer `7`, the same last-token `ln_1` output was fed through `c_attn` in two ways:

- as part of the full resumed sequence
- as a single-token resumed input for the KV-aware step

Observed result:

- `ln_1` last-token input: exact
- fused `c_attn` output: nonzero difference
- `Q` slice: exact
- `K` slice: exact
- `V` slice: first nonzero difference
- formatted `V`: same nonzero difference as the raw `V` slice

Representative numbers:

- fused `c_attn` last-token diff:
  - `L2 ~= 5.52e-06`
  - `max_abs_diff ~= 1.19e-06`
- `V` slice diff:
  - same as fused output diff
- `Q` and `K` slice diffs:
  - `0.0`

Interpretation:

- the first measurable KV-specific mismatch is not in cache append/readback
- it appears earlier, at the fused attention projection when the same token is evaluated in:
  - full-sequence mode
  - versus single-token resumed mode
- the mismatch is highly specific to the value path; query and key remain exact

Updated bug boundary:

- same hidden state
- same layernorm output
- different fused `c_attn` value segment depending on execution mode

This makes the next question more precise:

- why does the GPT-2 `c_attn` path produce a slightly different `V` slice for the same token when evaluated as part of a full sequence versus as a single-token resumed input?

### KV Diagnosis: Projection Artifact Survives Contiguity and Manual Checks

Additional micro-tests on step 2 / layer `7`:

- compared `c_attn(full_sequence)[:, -1, :]` against `c_attn(last_token_only)`
- forced the last-token tensor through:
  - direct view
  - `.contiguous()`
  - `.clone()`
  - standalone one-token tensor
- checked dtype, device, contiguity, and strides
- compared `attn.c_attn(x)` against an explicit manual `torch.addmm(bias, x, weight)` on the one-token input

Observed result:

- the same tiny V-only mismatch survives all last-token tensor variants
- `Q` remains exact
- `K` remains exact
- `V` remains the only differing slice
- manual `addmm` matches the module output exactly for the one-token input

Representative numbers:

- fused `c_attn` last-token diff:
  - `L2 ~= 5.52e-06`
  - `max_abs_diff ~= 1.19e-06`
- this same diff appears for:
  - view
  - contiguous copy
  - clone
  - standalone one-token tensor
- module vs manual `addmm` on the one-token input:
  - `L2 = 0.0`
  - `max_abs_diff = 0.0`

Interpretation:

- the current mismatch is not explained by:
  - non-contiguous input
  - dtype mismatch
  - device mismatch
  - module-vs-manual linear implementation on the one-token path
- the remaining effect is a shape-dependent projection artifact:
  - the same token produces a slightly different fused value segment when projected as part of a longer sequence than when projected alone

Practical read:

- the KV path may already be as semantically correct as this projection behavior allows on the current backend
- exact bit-level equality across full-sequence and one-token projection modes may not be attainable without forcing the same projection shape/path on both sides

### KV Operational Comparison

Operational comparison on prompt:

```text
The capital of France is
```

Boundary tested:

- `6`

Horizon tested:

- `20`

Comparison mode:

- exact baseline follows its own greedy tokens
- `kv_fast` follows its own greedy tokens
- compare token agreement, top-k overlap, latency, and cache size

Observed result:

- token agreement: `1.00`
- first divergence step: none observed
- top-5 overlap: `5/5` at every tested step

Representative performance numbers:

- exact latency / step: `~20.73 ms`
- `kv_fast` latency / step: `~6.77 ms`
- final cache size: `768,000` bytes

Interpretation:

- despite the tiny shape-dependent fused projection drift, the current `kv_fast` path is behaviorally stable on the tested 20-step horizon
- for this cut, the drift behaves like a numerical floor rather than an operational failure

Current operational read:

- exact baseline remains the gold correctness reference
- `kv_fast` is now a plausible execution-faithful fast path with a known numerical caveat

### KV Operational Envelope: Widened Sweep

Widened operational sweeps were then run against the frozen exact baseline using:

- boundaries: `0`, `6`, `10`
- horizons: `20`, `50`
- top-k: `5`

Prompt:

```text
The capital of France is
```

Observed result across the full tested grid:

- token agreement: `1.00` for every run
- first divergence step: none observed
- top-5 overlap: `5/5` at every tested step for every run

Representative timing and memory:

- boundary `0`, horizon `20`:
  - exact latency / step: `~20.71 ms`
  - `kv_fast` latency / step: `~11.35 ms`
  - final cache size: `1,548,288` bytes
- boundary `0`, horizon `50`:
  - exact latency / step: `~29.16 ms`
  - `kv_fast` latency / step: `~10.09 ms`
  - final cache size: `3,717,120` bytes
- boundary `6`, horizon `20`:
  - exact latency / step: `~20.93 ms`
  - `kv_fast` latency / step: `~5.98 ms`
  - final cache size: `702,720` bytes
- boundary `6`, horizon `50`:
  - exact latency / step: `~26.81 ms`
  - `kv_fast` latency / step: `~5.67 ms`
  - final cache size: `1,689,600` bytes
- boundary `10`, horizon `20`:
  - exact latency / step: `~21.11 ms`
  - `kv_fast` latency / step: `~3.34 ms`
  - final cache size: `140,544` bytes
- boundary `10`, horizon `50`:
  - exact latency / step: `~28.67 ms`
  - `kv_fast` latency / step: `~3.26 ms`
  - final cache size: `337,920` bytes

Second prompt:

```text
The capital of France is Paris and the capital of Japan is
```

Observed result across the same grid:

- token agreement: `1.00` for every run
- first divergence step: none observed
- top-5 overlap: `5/5` at every tested step for every run

Representative timing and memory:

- boundary `0`, horizon `20`:
  - exact latency / step: `~25.51 ms`
  - `kv_fast` latency / step: `~12.16 ms`
  - final cache size: `1,745,280` bytes
- boundary `0`, horizon `50`:
  - exact latency / step: `~31.62 ms`
  - `kv_fast` latency / step: `~10.41 ms`
  - final cache size: `4,190,208` bytes
- boundary `6`, horizon `20`:
  - exact latency / step: `~26.18 ms`
  - `kv_fast` latency / step: `~5.97 ms`
  - final cache size: `792,192` bytes
- boundary `6`, horizon `50`:
  - exact latency / step: `~30.17 ms`
  - `kv_fast` latency / step: `~5.87 ms`
  - final cache size: `1,904,640` bytes
- boundary `10`, horizon `20`:
  - exact latency / step: `~24.80 ms`
  - `kv_fast` latency / step: `~3.46 ms`
  - final cache size: `158,208` bytes
- boundary `10`, horizon `50`:
  - exact latency / step: `~31.79 ms`
  - `kv_fast` latency / step: `~3.31 ms`
  - final cache size: `380,160` bytes

Interpretation:

- the known `shape-dependent fused projection drift` remains numerically detectable in the microscope path
- but across the widened operational envelope it still does not produce behavioral divergence
- for the tested GPT-2-class Hugging Face path, `kv_fast` is now behaviorally validated across:
  - boundaries `0`, `6`, `10`
  - horizons `20`, `50`
  - two prompts of different prefix lengths

Updated operational read:

- exact replay path remains the correctness baseline
- `kv_fast` is substantially faster
- `kv_fast` is behaviorally indistinguishable from the exact baseline on the current tested envelope
- the remaining projection drift is best treated as a backend numerical floor unless a wider sweep proves otherwise

### Compact Replay: First Narrow Sweep

A first compact-replay experiment was added against the frozen exact baseline.

Scope of the experiment:

- model: repo-local `gpt2`
- boundary layer: `6`
- replay layer: `10`
- token under replay: last prompt token
- exact prefix states at replay layer are kept
- only the target token replay object is compacted
- compact variants keep the boundary token state plus the last `N` deltas between layers `7` and `10`

This is intentionally narrow. The question is:

- how much target-token replay state can be dropped before next-token behavior diverges from the exact replay baseline?

Prompt:

```text
The capital of France is
```

Object sizes:

- exact replay token at layer `10`: `3,072` bytes
- full boundary-plus-delta trace from layers `6 -> 10`: `15,360` bytes

Observed compact variants:

- depth `0` (boundary only):
  - object size: `3,072` bytes
  - greedy token match: yes
  - top-5 overlap: `3/5`
  - very large logit drift
- depth `1` (keep layer `10` delta only):
  - object size: `6,144` bytes
  - greedy token match: yes
  - top-5 overlap: `3/5`
- depth `2` (keep layers `9,10`):
  - object size: `9,216` bytes
  - greedy token match: yes
  - top-5 overlap: `5/5`
- depth `4` (keep layers `7,8,9,10`):
  - object size: `15,360` bytes
  - greedy token match: yes
  - top-5 overlap: `5/5`
  - residual logit drift remains tiny but nonzero

Second prompt:

```text
The capital of France is Paris and the capital of Japan is
```

Observed compact variants:

- depth `0` (boundary only):
  - greedy token match: no
  - top-5 overlap: `1/5`
- depth `1` (keep layer `10` delta only):
  - greedy token match: yes
  - top-5 overlap: `2/5`
- depth `2` (keep layers `9,10`):
  - greedy token match: yes
  - top-5 overlap: `4/5`
- depth `4` (keep layers `7,8,9,10`):
  - greedy token match: yes
  - top-5 overlap: `5/5`
  - residual logit drift remains tiny but nonzero

Interpretation:

- compact replay is now a real experimental branch rather than a roadmap bullet
- boundary-only replay can preserve the greedy token on very easy prompts, but it is not fidelity-stable
- keeping only the final late-layer delta is still too weak for top-k fidelity
- keeping the last two deltas is enough to recover full top-5 overlap on the shorter prompt, but not yet on the longer one
- the full local boundary-plus-delta trace is behaviorally aligned on both tested prompts, though still not bit-exact

Current compact-replay read:

- the minimal faithful replay object is smaller than "store everything blindly" but larger than boundary-only
- fidelity appears to improve monotonically as more late-layer deltas are restored
- the next useful move is to widen this compact sweep across:
  - more prompts
  - more boundary / replay cuts
  - short continuation horizons rather than next-token only

### Compact Replay: Replay-Layer Sweep

The compact-replay sweep was then widened across replay layers `8`, `9`, `10`, and `11` while keeping:

- boundary layer: `6`
- token under replay: last prompt token
- top-k width: `5`

Prompts tested:

```text
The capital of France is
The capital of France is Paris and the capital of Japan is
Alice opened the red door and found a small brass key that
```

Observed pattern:

- for replay layer `8`, keeping the full local delta band (`7,8`) was enough to recover full top-5 overlap on all three prompts
- for replay layer `9`, keeping the full local delta band (`7,8,9`) was enough to recover full top-5 overlap on all three prompts
- for replay layer `10`, keeping the full local delta band (`7,8,9,10`) was enough to recover full top-5 overlap on all three prompts
- for replay layer `11`, keeping the full local delta band (`7,8,9,10,11`) was required to recover full top-5 overlap on all three prompts

Representative sufficiency curve:

- replay layer `8`:
  - depth `0` or `1`: often preserves the greedy token, but top-5 overlap remains partial
  - depth `2` (full band): reaches `5/5` top-5 overlap with tiny residual drift
- replay layer `9`:
  - intermediate depths are stronger than boundary-only but still lossy
  - depth `3` (full band): reaches `5/5` top-5 overlap with tiny residual drift
- replay layer `10`:
  - depth `2` or `3` can already recover the greedy token and sometimes full top-5 overlap on easier prompts
  - the narrative prompt remained unstable at intermediate depths even with `5/5` top-k overlap in one case
  - depth `4` (full band): reaches `5/5` top-5 overlap with tiny residual drift
- replay layer `11`:
  - boundary-only and shallow delta bands often fail badly
  - some prompts require the full five-delta local band before the greedy token recovers
  - depth `5` (full band): reaches `5/5` top-5 overlap with tiny residual drift

Important nuance:

- the broad trend is monotone improvement as more late-layer deltas are restored
- but greedy-token behavior at intermediate depths is not perfectly monotone
- one clear example:
  - narrative prompt at replay layer `10`
  - depth `2` matched the greedy token
  - depth `3` had `5/5` top-k overlap but the greedy token changed
  - depth `4` restored both greedy-token match and full top-k overlap

Interpretation:

- the compact replay object is not just "boundary state plus a little bit more" in a smooth scalar sense
- intermediate delta bands can preserve the candidate set while still perturbing the local ranking inside that set
- deeper replay layers demand a correspondingly deeper late-delta band
- for the tested prompts and cuts, the full local boundary-plus-delta band from boundary `6` up to replay layer `r` is behaviorally aligned at replay layers `8` through `11`

Updated compact-replay read:

- a local late-band trace is now a credible compact replay design
- boundary-only replay is clearly insufficient once the task or replay layer gets harder
- greedy-token recovery and top-k recovery are related but distinct compact-fidelity metrics
- the next useful compact branch is horizon testing:
  - take the strongest reduced objects
  - run short continuation horizons against the exact baseline
  - measure where the compact object stops being behaviorally faithful

### Compact Replay: Short-Horizon Continuation

The compact branch was then extended from one-step prediction to short-horizon continuation against the exact replay baseline.

Tested configuration:

- boundary layer: `6`
- replay layer: `10`
- horizon: `10` greedy steps
- compact candidates:
  - depth `0`: boundary only
  - depth `2`: keep `9,10`
  - depth `3`: keep `8,9,10`
  - depth `4`: keep `7,8,9,10` (full local band)

Prompt:

```text
The capital of France is
```

Observed result:

- depth `0`:
  - token agreement: `0.10`
  - top-5 full-overlap steps: `0/10`
  - first divergence step: `2`
- depth `2`:
  - token agreement: `0.60`
  - top-5 full-overlap steps: `3/10`
  - first divergence step: `7`
- depth `3`:
  - token agreement: `1.00`
  - top-5 full-overlap steps: `5/10`
  - no greedy-token divergence in the tested horizon
- depth `4`:
  - token agreement: `1.00`
  - top-5 full-overlap steps: `10/10`
  - no divergence in the tested horizon

Second prompt:

```text
The capital of France is Paris and the capital of Japan is
```

Observed result:

- depth `0`:
  - token agreement: `0.50`
  - top-5 full-overlap steps: `0/10`
  - first divergence step: `1`
- depth `2`:
  - token agreement: `0.80`
  - top-5 full-overlap steps: `3/10`
  - first divergence step: `9`
- depth `3`:
  - token agreement: `1.00`
  - top-5 full-overlap steps: `5/10`
  - no greedy-token divergence in the tested horizon
- depth `4`:
  - token agreement: `1.00`
  - top-5 full-overlap steps: `10/10`
  - no divergence in the tested horizon

Third prompt:

```text
Alice opened the red door and found a small brass key that
```

Observed result:

- depth `0`:
  - token agreement: `0.00`
  - top-5 full-overlap steps: `0/10`
  - first divergence step: `1`
- depth `2`:
  - token agreement: `0.70`
  - top-5 full-overlap steps: `4/10`
  - first divergence step: `6`
- depth `3`:
  - token agreement: `0.10`
  - top-5 full-overlap steps: `1/10`
  - first divergence step: `1`
- depth `4`:
  - token agreement: `1.00`
  - top-5 full-overlap steps: `10/10`
  - no divergence in the tested horizon

Interpretation:

- the full local band (`7,8,9,10`) is not just one-step aligned; it remains continuation-faithful across the tested 10-step horizon on all three prompts
- reduced objects can remain locally plausible while failing over continuation
- depth `3` is especially instructive:
  - it preserved greedy-token agreement on both geography prompts
  - but only `5/10` steps retained full top-5 overlap
  - and it failed badly on the narrative prompt
- therefore, compact one-step quality is not enough to certify continuation stability

Updated compact-continuation read:

- the compact replay frontier is real and non-smooth
- intermediate reduced objects can look strong on easy prompts and still collapse on harder continuation
- the current strongest compact candidate is:
  - boundary layer `6`
  - replay layer `10`
  - full local late band `7,8,9,10`
- that candidate is behaviorally aligned over the tested 10-step horizon, while thinner objects trade away stability for size

Canonical artifact:

- the compact branch now has a dedicated frontier table in `docs/compact_frontier.md`
- that table keeps one-step status separate from continuation stability
- it should be treated as the main evidence surface for compact replay, rather than scattered narrative summaries

Prompt-family widening:

- the compact frontier has now been widened beyond factual and narrative prompts to include:
  - procedural / instruction-like continuation
  - code-like completion
- the main conclusion still holds:
  - the full late-band object remains the only continuation-safe compact replay object on the tested set
- the new rows sharpen the failure pattern:
  - procedural prompts show that a one-step-perfect reduced object can still collapse early in continuation
  - code-like prompts show that thinner objects can preserve token agreement while still losing ranking stability

Horizon extension:

- the fixed prompt-family panel has now also been extended from `10` to `20` greedy steps at the same compact cut:
  - boundary layer `6`
  - replay layer `10`
  - compact objects `0 / 2 / 3 / 4`
- the central conclusion survives intact:
  - the full late-band object `7,8,9,10` remains the only continuation-safe compact replay object on the tested panel
- the 20-step extension makes the prompt-family dependence clearer:
  - `depth 3` is no longer plausibly safe in general
  - `depth 2` remains continuation-unsafe everywhere
  - code-like prompts still allow token agreement to survive longer than ranking stability

Replay-layer probe:

- the next mechanism probe held the same fixed prompt-family panel and `20`-step horizon while shifting replay layer:
  - replay layer `9`
  - replay layer `11`
- initial replay-layer-`11` failure turned out to be contaminated by a GPT-2 trace-contract bug:
  - `trace_text()` had been reading the model's final hidden-state slot rather than the pre-`ln_f` block-`11` output
  - this made replay layer `11` mean different tensors in different code paths
- after switching GPT-2 tracing to explicit manual block stepping, replay layer `11` was re-run
- corrected result:
  - replay layer `9`: the full local band `7,8,9` is continuation-safe across the tested panel
  - replay layer `10`: the full local band `7,8,9,10` remains continuation-safe across the tested panel
  - replay layer `11`: the full local band `7,8,9,10,11` is also continuation-safe across the tested panel
- corrected mechanism claim:
  - the safe-object story is again "full local band at the tested replay cut is sufficient"
  - replay-layer location still matters for thinner objects, but layer `11` is no longer an exception

Reduced-object map:

- with the full local band restored as the safe reference object, the current compact question is now squarely about reduced-object failure modes
- a reduced-object matrix was run across replay layers `9/10/11` on the fixed prompt-family panel
- current read:
  - every reduced object is continuation-unsafe somewhere on the panel
  - code-like prompts remain the most forgiving at the token level
  - narrative and procedural prompts remain the hardest stress tests
  - deeper reduced objects help, but not in a smooth or universally reliable way

Alternative object class:

- a qualitatively different compact object class was added to the comparison surface:
  - direct replay token at the replay layer
- this object is continuation-safe by construction in the current setup and much smaller than the full local late band:
  - `3,072` bytes versus `12,288` / `15,360` / `18,432` bytes at replay layers `9` / `10` / `11`
- this changes the next design question:
  - the problem is no longer "is anything smaller than the full band possible?"
  - it is now "can a compressed object class approach direct replay-token fidelity without storing the exact replay-layer token?"

Replay-token surrogates:

- the next surrogate family compressed the replay token itself rather than the late-band trace
- tested at replay layer `10` on the fixed prompt-family panel:
  - `token@10/fp16`: `1,536` bytes
  - `token@10/int8`: `772` bytes
- result:
  - `fp16` replay-token storage preserved continuation behavior across the tested panel
  - `int8` replay-token storage preserved token agreement across the tested panel, with only modest top-k degradation
- updated design axis:
  - the most promising compact branch is now replay-token compression, not further late-band thinning

Cross-layer surrogate widening:

- the replay-token surrogate family was then widened across replay layers `9`, `10`, and `11`
- corrected read:
  - `fp16` replay-token storage is continuation-safe across the tested panel at all three replay layers
  - `int8` replay-token storage remains strong, but now shows real cut sensitivity:
    - replay layers `9` and `10`: token agreement stayed perfect on the tested panel
    - replay layer `11`: factual-simple dropped to token agreement `0.60` with divergence at step `7`
- updated frontier:
  - `fp16` replay-token compression is now the strongest safe compact surrogate family on the tested panel
  - `int8` replay-token compression is the first genuinely interesting lossy boundary

Routing -> replay bridge:

- a bridge command now connects the staged Apollo router to the tracked replay path:
  - semantic pool selection
  - temporal/PageRank rerank inside the pool
  - graph-local refinement
  - then hit-conditioned replay-object evaluation on the routed top-1 region
- first live validation on a small Apollo slice:
  - router: local Qwen3.5 2B GGUF
  - replay backend: repo-local `gpt2`
  - cases: `3`
  - staged routing top-1 hit rate: `0.67`
  - staged routing top-k hit rate: `1.00`
- on the two routed hits in that slice, the tracked replay objects all remained clean over the tested `10`-step horizon:
  - `text@window` control
  - `token@10`
  - `token@10/fp16`
  - `token@10/int8`
  - full late band `delta_depth=4`

Current bridge read:

- the project now has an actual routing -> replay architecture path, not just two disconnected sub-results
- the first bridge validation is small, but it shows the intended stack is executable end to end
- the bridge surface now includes a plain text control, so tracked replay objects can be compared against ordinary routed text on the same hit slice
- the next bridge question is no longer "can we connect them?" but "which replay object wins once the routed hit slice gets larger and harder?"

Widened bridge slice:

- the staged router was then widened to a `9`-case Apollo slice with the same replay cut:
  - boundary `6`
  - replay layer `10`
  - replay horizon `10`
- staged routing stayed strong enough to make the bridge meaningful:
  - top-1 hit rate: `0.78`
  - top-k hit rate: `1.00`

This was the first bridge run that actually separated replay objects:

- `text@window`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
- `token@10`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
- `token@10/fp16`:
  - token agreement: `1.00`
  - top-5 full rate: `0.99`
- `token@10/int8`:
  - token agreement: `0.86`
  - top-5 full rate: `0.79`
  - divergence rate: `0.14`
- full late band `delta_depth=4`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`

Bucketed view:

- near hits:
  - `int8` replay token stays reasonably strong
  - token agreement: `1.00`
  - top-5 full rate: `0.87`
- medium hits:
  - still easy on this slice
  - `int8` replay token token agreement: `1.00`
  - top-5 full rate: `0.90`
- far hits:
  - first clearly discriminative bucket
  - `text@window`, exact replay token, `fp16`, and full late band all stayed clean
  - `int8` replay token dropped to:
    - token agreement: `0.67`
    - top-5 full rate: `0.67`
    - divergence rate: `0.33`

Updated bridge read:

- the routed bridge is now discriminative on a harder slice
- `int8` replay-token compression is the first object to degrade materially under routed Apollo pressure
- `fp16` replay-token storage remains extremely strong and is now close to the exact/text/full-band group on the tested bridge slice

Wider discriminative bridge slice:

- the bridge was widened again to a `12`-case Apollo slice with the same fixed router and object panel
- staged routing stayed strong:
  - top-1 hit rate: `0.83`
  - top-k hit rate: `1.00`

The separation persisted and sharpened:

- `text@window`: token agreement `1.00`, top-5 full rate `1.00`
- `token@10`: token agreement `1.00`, top-5 full rate `1.00`
- `token@10/fp16`: token agreement `1.00`, top-5 full rate `0.99`
- `token@10/int8`:
  - token agreement: `0.81`
  - top-5 full rate: `0.73`
  - divergence rate: `0.20`
- full late band `delta_depth=4`: token agreement `1.00`, top-5 full rate `1.00`

Bucketed view on the widened slice:

- near hits:
  - `int8` still degrades, but more mildly
  - token agreement: `1.00`
  - top-5 full rate: `0.90`
- medium hits:
  - `int8` degrades sharply
  - token agreement: `0.55`
  - top-5 full rate: `0.55`
  - divergence rate: `0.50`
- far hits:
  - `int8` remains clearly degraded
  - token agreement: `0.75`
  - top-5 full rate: `0.65`
  - divergence rate: `0.25`

Current bridge conclusion:

- the routed bridge is now robustly discriminative on widened Apollo slices
- `text@window`, exact replay token, `fp16`, and the full late band remain in the stable group on the tested bridge slices
- `int8` replay-token compression is now a demonstrated loser under routed pressure, not just a suggestive boundary

Larger bucket-balanced bridge slice:

- the bridge was widened again to an `18`-case Apollo slice, keeping the same fixed router, replay cut, and object panel
- staged routing remained stable:
  - top-1 hit rate: `0.78`
  - top-k hit rate: `1.00`
- routed top-1 hits covered `14` cases, which gives a more balanced read across the `near` / `medium` / `far` buckets

Overall hit-conditioned replay summary:

- `text@window`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
- `token@10`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
- `token@10/fp16`:
  - token agreement: `1.00`
  - top-5 full rate: `0.99`
- `token@10/int8`:
  - token agreement: `0.86`
  - top-5 full rate: `0.79`
  - divergence rate: `0.14`
- full late band `delta_depth=4`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`

Bucketed view on the larger slice:

- near hits:
  - `int8` stays usable at the token level but still sheds ranking stability
  - token agreement: `1.00`
  - top-5 full rate: `0.90`
- medium hits:
  - `int8` remains a real failure mode rather than a one-off anomaly
  - token agreement: `0.78`
  - top-5 full rate: `0.75`
  - divergence rate: `0.25`
- far hits:
  - `int8` also remains clearly degraded
  - token agreement: `0.80`
  - top-5 full rate: `0.70`
  - divergence rate: `0.20`
- `text@window`, exact replay token, `fp16`, and full late band all stayed behaviorally clean in every bucket on this slice

Updated bridge read:

- the bridge hierarchy survives the larger, more balanced Apollo slice
- `token@10/fp16` remains effectively in the safe group, with only slight ranking softness
- `token@10/int8` continues to be the first replay object that fails under routed pressure, and it now does so in both the medium and far buckets rather than only in a single stressed corner

Harder fp16 stress cut:

- the bridge was then stressed by widening the Apollo slice again and doubling the replay horizon to `20` steps:
  - cases: `24`
  - routed top-1 hits: `18`
  - top-1 hit rate: `0.75`
  - top-k hit rate: `1.00`
- this was the right pressure test for the practical compact candidate because it gave iterative drift more room to show up without changing the router or the replay object panel

Hit-conditioned replay summary on the harder cut:

- `text@window`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
- `token@10`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
- `token@10/fp16`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`
  - divergence rate: `0.00`
- `token@10/int8`:
  - token agreement: `0.88`
  - top-5 full rate: `0.79`
  - divergence rate: `0.17`
- full late band `delta_depth=4`:
  - token agreement: `1.00`
  - top-5 full rate: `1.00`

Bucketed view on the harder cut:

- near hits:
  - `fp16` remained clean except for negligible ranking softness
  - `int8` degraded to token agreement `0.93`, top-5 full rate `0.82`, divergence rate `0.17`
- medium hits:
  - `fp16` stayed clean
  - `int8` degraded to token agreement `0.83`, top-5 full rate `0.78`, divergence rate `0.20`
- far hits:
  - `fp16` stayed clean
  - `int8` degraded to token agreement `0.86`, top-5 full rate `0.78`, divergence rate `0.14`

Updated bridge conclusion:

- `token@10/fp16` has now survived the routed bridge under both wider slices and a doubled continuation horizon, and remains behaviorally welded to the exact/text/full-band group on the tested Apollo envelope
- `token@10/int8` remains the first repeatedly demonstrated lossy failure boundary and continues to degrade across all routed difficulty buckets when the bridge is stressed harder

Memory ledger skeleton:

- the bridge now includes an observability-only memory ledger layer instead of treating replay objects as disposable benchmark rows
- a new `memory_ledger.py` module defines a stable `MemoryObject` schema with:
  - `object_id`
  - `kind`
  - `bytes`
  - `rank_history`
  - `topk_frequency`
  - `downstream_utility`
  - `replay_usage_count`
  - `created_at`
  - `last_retrieved_at`
  - `last_reinjected_at`
  - `last_useful_at`
  - `jump_score`
  - `tier`
  - `pinned`
  - `source_case_id`
  - `source_region_id`
- `bridge-apollo-replay` now updates that ledger passively on every routed top-1 replay evaluation without changing replay behavior
- the bridge output now includes a read-only tier report:
  - object counts by tier
  - recent promotions / demotions
  - low-utility tail
  - reporting-only archive/prune candidates
  - pinned objects

Current lifecycle policy:

- default everything to `warm`
- allow explicit `pinned`
- report `cold` / `archived` suggestions only
- do not auto-demote, archive, or prune anything yet

Current value of the ledger:

- the repo now has the protocol substrate for bounded live cognition plus routed autobiographical recall
- replay selection and replay quality can be observed as memory-object behavior, not just benchmark rows
- policy can now be added later without first rebuilding the data model or the bridge instrumentation

Harder recommendation-layer cut:

- the reporting-only recommendation layer was then run on the harder discriminative bridge cut:
  - cases: `24`
  - routed top-1 hits: `18`
  - replay horizon: `20`
- this was the right policy pressure test because the bridge already has a known loser on this slice:
  - `text@window`, exact replay token, full late band, and `fp16` remain behaviorally clean
  - `int8` remains the first repeated loser across the medium and far buckets

Observed ledger behavior on the harder cut:

- the low-utility tail populated with failing `token@10/int8` objects exactly as expected
- the suggestion table stayed empty
- no safe-group objects were falsely suggested for demotion or archive

What this means:

- the ledger substrate is working: it can already distinguish the weak replay-object tail from the safe group
- the current recommendation policy is too conservative for the present bridge instrumentation
- specifically, every replay object is currently recorded as entering the routed top-k by construction on top-1 hit cases, so `topk_frequency` does not yet help separate weak replay objects from strong ones

Updated policy read:

- reporting-only recommendations are behaving safely, but not yet informatively enough to drive tier suggestions on the harder bridge slice
- the next calibration step is to refine the recommendation signal so it keys more directly off replay-object behavior quality, not just routed-hit participation

Replay-quality-calibrated recommendation layer:

- the recommendation layer was then recalibrated to score replay objects directly on continuation behavior instead of leaning on routed-hit participation
- the key added signals were:
  - token agreement over horizon
  - top-k full-overlap rate
  - divergence presence and step
  - bucket-aware penalties for medium / far failures
- this kept routed selection as the gate and replay quality as the value signal

Harder policy rerun after recalibration:

- reran the same harder bridge cut:
  - cases: `24`
  - routed top-1 hits: `18`
  - replay horizon: `20`
- replay behavior itself stayed unchanged:
  - `text@window`, exact replay token, full late band, and `fp16` remained in the safe group
  - `int8` remained the repeated loser

What changed in the ledger:

- the `Tier Transition Suggestions` table now populated
- all populated suggestions were `token@10/int8` objects
- no safe-group objects were falsely suggested
- the weaker `int8` cases now split into:
  - `archived` suggestions for the worst medium / far failures
  - `cold` suggestions for softer ranking-only failures

Interpretation:

- the recommendation layer is now finally aligned with the bridge’s known weak tail
- the policy remains reporting-only, but it has crossed from "safe but silent" to "informative and plausibly calibrated"
- the next lifecycle step can now be conservative tier transitions, because the scoring is no longer blind to replay-object quality

First reversible lifecycle action:

- enabled `warm -> cold` transitions only
- `archived` and `pruned` remained reporting-only
- the action stayed conservative:
  - only `cold` suggestions were eligible
  - pinned objects remained ineligible
  - every applied transition was logged with confidence, metrics, bucket, reason, and timestamp

Hard lifecycle run:

- reran the same harder bridge cut:
  - cases: `24`
  - routed top-1 hits: `18`
  - replay horizon: `20`
- replay behavior itself remained unchanged:
  - `text@window`, exact replay token, full late band, and `fp16` remained clean
  - `int8` remained the repeated loser

Tier effects:

- before:
  - `warm = 90`
  - `cold = 0`
- after:
  - `warm = 83`
  - `cold = 7`
- `archived` remained `0` because archive transitions were still reporting-only

What actually moved:

- all applied transitions were `token@10/int8` objects
- no safe-group objects were moved
- the applied cold set was dominated by ranking-softness cases, including:
  - near-bucket cases with token agreement still at `1.00` but degraded top-k stability
  - medium/far cases with stronger ranking loss

What this means:

- the first reversible lifecycle action is behaving safely at the object-family level: only the known loser family moved
- but it is already a little aggressive within that family, because it cools some `int8` cases that still preserve token agreement and only lose ranking stability
- that is acceptable for a reversible `cold` tier, but it is a real calibration signal before any stronger lifecycle action is enabled

Refined cold calibration:

- the `cold` rule was then narrowed so ranking-only softness is not enough by itself in easy conditions
- the refined policy now treats `cold` as:
  - real continuation weakness (`token drift` or `divergence`)
  - or ranking weakness in harder buckets (`medium` / `far`)
- near-bucket ranking softness alone no longer triggers `cold`

Hard lifecycle rerun after calibration:

- reran the same harder bridge cut again:
  - cases: `24`
  - routed top-1 hits: `18`
  - replay horizon: `20`
- replay behavior remained unchanged:
  - `text@window`, exact replay token, full late band, and `fp16` remained clean
  - `int8` remained the repeated loser

Tier effects after refinement:

- before:
  - `warm = 90`
  - `cold = 0`
- after:
  - `warm = 89`
  - `cold = 1`

Interpretation:

- the first mutable tier is now more conservative in the right way
- only one `int8` object, a far-bucket ranking-weak case, was actually cooled
- near-only ranking softness no longer gets auto-cooled
- archive suggestions still capture the more severe `int8` failures, but remain non-mutating

Persistence across runs:

- the bridge now persists its ledger state in a repo-local JSON file:
  - `artifacts/memory_ledger/bridge_apollo_replay.json`
- that makes weak-evidence counts, consecutive weak runs, and resurgence tracking meaningful across repeated bridge executions instead of resetting every time

Second persisted bridge run:

- reran the same harder bridge cut against the persisted ledger
- replay metrics stayed unchanged again:
  - `text@window`, exact replay token, full late band, and `fp16` remained clean
  - `int8` remained the repeated loser

What persistence changed:

- no new `warm -> cold` transitions were applied on the second persisted run
- the existing `cold` set remained stable:
  - before: `warm = 85`, `cold = 5`
  - after: `warm = 85`, `cold = 5`
- several of those existing cold `int8` objects now became archive-eligible in reporting:
  - repeated weak evidence count reached `2`
  - consecutive weak runs reached `2`
  - no resurgence was observed

Interpretation:

- the lifecycle policy now has real inertia across runs instead of twitching on a single weak outing
- `cold` remains reversible and conservative
- `archived` is now backed by repeated persisted weakness rather than one-off suggestion noise, while still remaining non-mutating

Recovery suggestion layer:

- added the reverse lifecycle path in reporting only:
  - strong resurgence is now tracked via:
    - `strong_recovery_count`
    - `consecutive_strong_runs`
    - `last_strong_recovery_at`
  - `cold -> warm` suggestions require:
    - strong token agreement
    - strong top-k stability
    - no divergence
    - recovery under `medium` / `far` pressure rather than only easy near hits
- bridge output now has a dedicated recovery-suggestions table, separate from archive-style transition suggestions

First live recovery check:

- reran a small bridge slice against the persisted ledger
- no recovery suggestions appeared, which is the correct conservative outcome for the current data:
  - the existing cold set has not yet shown strong resurgence under meaningful pressure
  - no false warm-ups were proposed

Current read:

- the protocol now has a symmetric reporting surface:
  - weak objects can cool
  - repeated weakness can become archive-eligible in reporting
  - strong resurgence can become warm-eligible in reporting
- but only the reversible `warm -> cold` step is still allowed to mutate state
