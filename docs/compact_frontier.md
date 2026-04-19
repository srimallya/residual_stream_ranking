# Compact Replay Frontier

This document is the canonical compact-replay artifact for the current GPT-2-class Hugging Face path.

It tracks the compact objects that have actually been tested against the exact replay baseline, with one-step status kept separate from continuation stability.

## Current Setup

- model: repo-local `gpt2`
- boundary layer: `6`
- replay layer: `10`
- horizon for continuation checks: `10` greedy steps
- exact reference: replay from the exact hidden state at replay layer `10`
- compact object: exact target-token boundary state at layer `6` plus a trailing late-delta band

Object definitions:

- `depth 0`: boundary only
- `depth 2`: keep late deltas `9,10`
- `depth 3`: keep late deltas `8,9,10`
- `depth 4`: keep late deltas `7,8,9,10` (full local band)

Prompt families used so far:

- factual-simple: `"The capital of France is"`
- factual-compositional: `"The capital of France is Paris and the capital of Japan is"`
- narrative: `"Alice opened the red door and found a small brass key that"`
- procedural / instruction-like: `"To boil an egg, first fill a pot with water and"`
- code-like completion: `"def add_numbers(a, b):\n    return"`

## Frontier Table

| Object Kept | Prompt Family | One-Step Status | Token Agreement / 10 | First Divergence Step | Top-5 Full-Overlap Steps | Current Read |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `depth 0` | factual-simple | greedy match, `3/5` top-k | `0.10` | `2` | `0/10` | Too thin |
| `depth 2` (`9,10`) | factual-simple | greedy match, `5/5` top-k | `0.60` | `7` | `3/10` | Locally plausible, not continuation-safe |
| `depth 3` (`8,9,10`) | factual-simple | greedy match, `5/5` top-k | `1.00` | none | `5/10` | Tokens stable, ranking not stable |
| `depth 4` (`7,8,9,10`) | factual-simple | greedy match, `5/5` top-k | `1.00` | none | `10/10` | Safe on tested horizon |
| `depth 0` | factual-compositional | greedy mismatch, `1/5` top-k | `0.50` | `1` | `0/10` | Too thin |
| `depth 2` (`9,10`) | factual-compositional | greedy match, `4/5` top-k | `0.80` | `9` | `3/10` | Better, still unstable |
| `depth 3` (`8,9,10`) | factual-compositional | greedy match, `5/5` top-k | `1.00` | none | `5/10` | Tokens stable, ranking not stable |
| `depth 4` (`7,8,9,10`) | factual-compositional | greedy match, `5/5` top-k | `1.00` | none | `10/10` | Safe on tested horizon |
| `depth 0` | narrative | greedy mismatch, `3/5` top-k | `0.00` | `1` | `0/10` | Too thin |
| `depth 2` (`9,10`) | narrative | greedy match, `5/5` top-k | `0.70` | `6` | `4/10` | One-step flattering, continuation-unsafe |
| `depth 3` (`8,9,10`) | narrative | greedy mismatch, `5/5` top-k | `0.10` | `1` | `1/10` | Warning case |
| `depth 4` (`7,8,9,10`) | narrative | greedy match, `5/5` top-k | `1.00` | none | `10/10` | Safe on tested horizon |
| `depth 0` | procedural / instruction-like | greedy mismatch, `1/5` top-k | `0.20` | `1` | `0/10` | Too thin |
| `depth 2` (`9,10`) | procedural / instruction-like | greedy match, `5/5` top-k | `0.20` | `3` | `2/10` | One-step flattering, continuation-unsafe |
| `depth 3` (`8,9,10`) | procedural / instruction-like | greedy match, `4/5` top-k | `0.90` | `10` | `6/10` | Strong but still not safe |
| `depth 4` (`7,8,9,10`) | procedural / instruction-like | greedy match, `5/5` top-k | `1.00` | none | `10/10` | Safe on tested horizon |
| `depth 0` | code-like completion | greedy mismatch, `2/5` top-k | `0.00` | `1` | `0/10` | Too thin |
| `depth 2` (`9,10`) | code-like completion | greedy match, `4/5` top-k | `1.00` | none | `3/10` | Tokens stable, ranking not stable |
| `depth 3` (`8,9,10`) | code-like completion | greedy match, `5/5` top-k | `1.00` | none | `4/10` | Tokens stable, ranking not stable |
| `depth 4` (`7,8,9,10`) | code-like completion | greedy match, `5/5` top-k | `1.00` | none | `10/10` | Safe on tested horizon |

## What The Table Means

- One-step quality is not a sufficient certificate of continuation stability.
- Full top-k overlap is stronger than greedy-token match, but it still does not guarantee continued alignment for thinner objects.
- The frontier is non-smooth:
  - `depth 3` looks strong on the two factual prompts
  - the same object fails badly on the narrative prompt
- Prompt family matters:
  - procedural continuation punishes `depth 2` much harder than its one-step result suggests
  - code-like continuation allows thinner objects to preserve tokens while still losing ranking stability
- The current safe compact object is the full local late band:
  - boundary layer `6`
  - replay layer `10`
  - deltas `7,8,9,10`

## Current Claim

Reduced replay objects can preserve one-step behavior without preserving iterative continuation. On the tested set, the full local late-band object is the only continuation-faithful compact replay object so far.

## Horizon Extension: 20-Step Check

The same fixed prompt-family panel and the same compact objects were then extended from `10` to `20` greedy steps.

This does not change the frontier qualitatively. It sharpens it.

| Object Kept | Prompt Family | Token Agreement / 20 | First Divergence Step | Top-5 Full-Overlap Steps | Current Read |
| --- | --- | ---: | ---: | ---: | --- |
| `depth 0` | factual-simple | `0.05` | `2` | `0/20` | Too thin |
| `depth 2` (`9,10`) | factual-simple | `0.35` | `7` | `4/20` | Degrades early |
| `depth 3` (`8,9,10`) | factual-simple | `1.00` | none | `11/20` | Tokens stable, ranking decays |
| `depth 4` (`7,8,9,10`) | factual-simple | `1.00` | none | `20/20` | Safe on tested horizon |
| `depth 0` | factual-compositional | `0.30` | `1` | `0/20` | Too thin |
| `depth 2` (`9,10`) | factual-compositional | `0.45` | `9` | `3/20` | Degrades early |
| `depth 3` (`8,9,10`) | factual-compositional | `0.60` | `13` | `8/20` | Not continuation-safe |
| `depth 4` (`7,8,9,10`) | factual-compositional | `1.00` | none | `20/20` | Safe on tested horizon |
| `depth 0` | narrative | `0.00` | `1` | `0/20` | Too thin |
| `depth 2` (`9,10`) | narrative | `0.35` | `6` | `5/20` | One-step flattering, continuation-unsafe |
| `depth 3` (`8,9,10`) | narrative | `0.05` | `1` | `1/20` | Warning case |
| `depth 4` (`7,8,9,10`) | narrative | `1.00` | none | `20/20` | Safe on tested horizon |
| `depth 0` | procedural / instruction-like | `0.15` | `1` | `0/20` | Too thin |
| `depth 2` (`9,10`) | procedural / instruction-like | `0.10` | `3` | `2/20` | One-step flattering, continuation-unsafe |
| `depth 3` (`8,9,10`) | procedural / instruction-like | `0.50` | `10` | `6/20` | Stronger, still unstable |
| `depth 4` (`7,8,9,10`) | procedural / instruction-like | `1.00` | none | `20/20` | Safe on tested horizon |
| `depth 0` | code-like completion | `0.05` | `1` | `0/20` | Too thin |
| `depth 2` (`9,10`) | code-like completion | `0.85` | `18` | `6/20` | Tokens persist, ranking weak |
| `depth 3` (`8,9,10`) | code-like completion | `1.00` | none | `11/20` | Tokens stable, ranking decays |
| `depth 4` (`7,8,9,10`) | code-like completion | `1.00` | none | `20/20` | Safe on tested horizon |

### What The 20-Step Extension Adds

- The full local band still holds across the entire fixed prompt-family panel.
- Thinner objects keep decaying with prompt-family-specific shapes rather than converging to one simple failure mode.
- `depth 3` is now clearly not safe in general:
  - it survives the simple factual and code-like prompts at the token level
  - it degrades on factual-compositional prompts
  - it fails badly on the narrative prompt
  - it is only moderately stable on the procedural prompt
- `depth 2` remains continuation-unsafe everywhere, even when it looks deceptively good on one-step checks or on code-like token agreement.

### Updated Safe Object

On the current GPT-2 compact frontier, the only continuation-safe reduced replay object on the tested panel through `20` steps is still:

- boundary layer `6`
- replay layer `10`
- full local late band `7,8,9,10`

## Nearby Replay-Layer Probe

The next mechanism question was whether the same fixed prompt-family panel and the same `20`-step horizon would preserve the "full local band is safe" story when the replay layer shifts slightly.

Tested replay layers:

- `9`
- `11`

Boundary layer remained fixed at `6`.

### Replay Layer 9

Depths tested:

- `0` through `3`
- full local band at this cut is `depth 3` = deltas `7,8,9`

Observed result:

- the full local band (`7,8,9`) is continuation-safe across the entire fixed prompt-family panel:
  - token agreement: `1.00`
  - top-5 full-overlap steps: `20/20`
  - first divergence: none
- thinner objects remain unsafe and prompt-family-dependent

Representative table:

| Object Kept | Prompt Family | Token Agreement / 20 | First Divergence Step | Top-5 Full-Overlap Steps | Current Read |
| --- | --- | ---: | ---: | ---: | --- |
| `depth 0` | factual-simple | `0.20` | `4` | `0/20` | Too thin |
| `depth 2` (`8,9`) | factual-simple | `0.20` | `4` | `3/20` | Unsafe |
| `depth 3` (`7,8,9`) | factual-simple | `1.00` | none | `20/20` | Safe |
| `depth 0` | factual-compositional | `0.25` | `6` | `1/20` | Too thin |
| `depth 2` (`8,9`) | factual-compositional | `0.35` | `6` | `1/20` | Unsafe |
| `depth 3` (`7,8,9`) | factual-compositional | `1.00` | none | `20/20` | Safe |
| `depth 0` | narrative | `0.00` | `1` | `0/20` | Too thin |
| `depth 2` (`8,9`) | narrative | `0.35` | `6` | `5/20` | Unsafe |
| `depth 3` (`7,8,9`) | narrative | `1.00` | none | `20/20` | Safe |
| `depth 0` | procedural / instruction-like | `0.00` | `1` | `0/20` | Too thin |
| `depth 2` (`8,9`) | procedural / instruction-like | `0.40` | `9` | `4/20` | Unsafe |
| `depth 3` (`7,8,9`) | procedural / instruction-like | `1.00` | none | `20/20` | Safe |
| `depth 0` | code-like completion | `0.25` | `6` | `0/20` | Too thin |
| `depth 2` (`8,9`) | code-like completion | `1.00` | none | `11/20` | Tokens stable, ranking not stable |
| `depth 3` (`7,8,9`) | code-like completion | `1.00` | none | `20/20` | Safe |

### Replay Layer 11

Depths tested:

- `0` through `5`
- full local band at this cut is `depth 5` = deltas `7,8,9,10,11`

Observed result after fixing the GPT-2 trace contract:

- the full local band (`7,8,9,10,11`) is continuation-safe across the fixed prompt-family panel
- thinner objects remain unsafe and prompt-family-dependent

Representative result:

| Object Kept | Prompt Family | Token Agreement / 20 | First Divergence Step | Top-5 Full-Overlap Steps | Current Read |
| --- | --- | ---: | ---: | ---: | --- |
| `depth 0` | factual-simple | `0.00` | `1` | `0/20` | Too thin |
| `depth 4` (`8,9,10,11`) | factual-simple | `0.40` | `7` | `5/20` | Unsafe |
| `depth 5` (`7,8,9,10,11`) | factual-simple | `1.00` | none | `20/20` | Safe |
| `depth 0` | factual-compositional | `0.00` | `1` | `0/20` | Too thin |
| `depth 4` (`8,9,10,11`) | factual-compositional | `0.60` | `13` | `9/20` | Unsafe |
| `depth 5` (`7,8,9,10,11`) | factual-compositional | `1.00` | none | `20/20` | Safe |
| `depth 0` | narrative | `0.00` | `1` | `0/20` | Too thin |
| `depth 4` (`8,9,10,11`) | narrative | `0.35` | `6` | `5/20` | Unsafe |
| `depth 5` (`7,8,9,10,11`) | narrative | `1.00` | none | `20/20` | Safe |
| `depth 0` | procedural / instruction-like | `0.00` | `1` | `0/20` | Too thin |
| `depth 4` (`8,9,10,11`) | procedural / instruction-like | `0.40` | `9` | `4/20` | Unsafe |
| `depth 5` (`7,8,9,10,11`) | procedural / instruction-like | `1.00` | none | `20/20` | Safe |
| `depth 0` | code-like completion | `0.00` | `1` | `0/20` | Too thin |
| `depth 4` (`8,9,10,11`) | code-like completion | `1.00` | none | `11/20` | Tokens stable, ranking not stable |
| `depth 5` (`7,8,9,10,11`) | code-like completion | `1.00` | none | `20/20` | Safe |

### What The Replay-Layer Probe Means

- replay-layer locality still matters, but the earlier "layer 11 is impossible" result was a trace-contract bug, not a real compact frontier fact
- after fixing GPT-2 tracing to record pre-`ln_f` block outputs consistently, the current "safe full local band" story holds at replay layers `9`, `10`, and `11`
- thinner objects remain replay-layer- and prompt-family-sensitive

## Current Mechanism Read

On the tested GPT-2 compact frontier:

- replay layer `9`:
  - full local band `7,8,9` is continuation-safe
- replay layer `10`:
  - full local band `7,8,9,10` is continuation-safe
- replay layer `11`:
  - full local band `7,8,9,10,11` is continuation-safe

So the current safe-object story is:

- a full local late band is sufficient at the tested replay layers `9`, `10`, and `11`
- thinner objects remain unsafe in replay-layer- and prompt-family-specific ways

## Reduced-Object Failure Map

With the full local band restored as the safe reference object, the next useful question is how thinner objects fail across replay layers and prompt families.

This section summarizes the reduced objects only:

- replay layer `9`: depths `0`, `1`, `2`
- replay layer `10`: depths `0`, `1`, `2`, `3`
- replay layer `11`: depths `0`, `1`, `2`, `3`, `4`

The full-band object at each replay layer is excluded here because it is the safe reference.

### Headline Pattern

- every reduced object is continuation-unsafe somewhere on the fixed prompt-family panel
- deeper reduced objects are often better, but not smoothly or universally better
- code-like prompts are the most forgiving at the token level
- narrative and procedural prompts expose instability early

### Best Reduced Object By Replay Layer

| Replay Layer | Best Reduced Object | What It Preserves | Why It Is Still Unsafe |
| --- | --- | --- | --- |
| `9` | `depth 2` (`8,9`) | code-like token agreement `1.00`, `11/20` top-5 full-overlap steps | factual, narrative, and procedural families still diverge early |
| `10` | `depth 3` (`8,9,10`) | factual-simple and code-like token agreement `1.00`, `11/20` top-5 full-overlap steps | factual-compositional degrades, narrative collapses, procedural only partly survives |
| `11` | `depth 4` (`8,9,10,11`) | code-like token agreement `1.00`, `11/20` top-5 full-overlap steps | factual, narrative, and procedural families still diverge materially |

### Failure Characteristics By Prompt Family

| Prompt Family | Typical Reduced-Object Failure Shape |
| --- | --- |
| factual-simple | reduced objects can preserve tokens for a while, but ranking degrades early unless the full band is present |
| factual-compositional | reduced objects degrade faster than simple factual prompts; partial bands are not reliably safe |
| narrative | harsh stress test; seemingly strong reduced objects can collapse almost immediately |
| procedural / instruction-like | early collapse even after good one-step behavior; reduced objects are not trustworthy here |
| code-like completion | most forgiving for token agreement; still exposes ranking instability clearly |

### Current Practical Read

- if the objective is continuation safety, use the full local band at the tested replay cut
- if the objective is exploratory compression only, code-like and simple factual prompts can flatter reduced objects
- but reduced-object success on those easier families does not generalize across the fixed panel

## Alternative Object Class: Direct Replay Token

The late-band family is not the only compact object class.

A qualitatively different object is:

- the exact target-token state at the replay layer itself

In the current compact setup, prefix states at the replay layer are already kept exact. That means the direct replay-token object can be evaluated cleanly against the same continuation baseline.

### Why It Matters

This object class is much smaller than the full local late band:

| Replay Layer | Direct Replay Token | Full Local Late Band |
| --- | ---: | ---: |
| `9` | `3,072` bytes | `12,288` bytes |
| `10` | `3,072` bytes | `15,360` bytes |
| `11` | `3,072` bytes | `18,432` bytes |

And it is continuation-safe by construction on the tested panel, because it restores the exact token state at the replay cut.

### Current Read

- the plain "thinner late band" family is not generally safe
- but a different compact object class does exist:
  - direct replay token at the replay layer
- so the next object-design question is no longer:
  - "can anything smaller than the full band work?"
- it is now:
  - "can we find a compact object class that approaches direct replay-token fidelity without requiring the exact replay-layer token itself?"

## Replay-Token Surrogates

The next surrogate family tested was a direct compression of the replay token itself.

Tested at replay layer `10` on the fixed prompt-family panel:

- `token@10`: exact replay token (`3,072` bytes)
- `token@10/fp16`: replay token stored in `float16` (`1,536` bytes)
- `token@10/int8`: replay token stored as symmetric int8 plus one float32 scale (`772` bytes)

### 10-Step Result

| Object | Bytes | Prompt Family | Token Agreement / 10 | Top-5 Full-Overlap Steps | Current Read |
| --- | ---: | --- | ---: | ---: | --- |
| `token@10` | `3,072` | all tested families | `1.00` | `10/10` | Exact reference |
| `token@10/fp16` | `1,536` | factual-simple | `1.00` | `10/10` | Safe |
| `token@10/fp16` | `1,536` | factual-compositional | `1.00` | `10/10` | Safe |
| `token@10/fp16` | `1,536` | narrative | `1.00` | `10/10` | Safe |
| `token@10/fp16` | `1,536` | procedural / instruction-like | `1.00` | `10/10` | Safe |
| `token@10/fp16` | `1,536` | code-like completion | `1.00` | `10/10` | Safe |
| `token@10/int8` | `772` | factual-simple | `1.00` | `8/10` | Tokens stable, small ranking loss |
| `token@10/int8` | `772` | factual-compositional | `1.00` | `10/10` | Safe on tested horizon |
| `token@10/int8` | `772` | narrative | `1.00` | `8/10` | Tokens stable, ranking loss |
| `token@10/int8` | `772` | procedural / instruction-like | `1.00` | `9/10` | Tokens stable, minor ranking loss |
| `token@10/int8` | `772` | code-like completion | `1.00` | `9/10` | Tokens stable, minor ranking loss |

### What This Changes

- a viable compressed replay-token family already exists
- `fp16` replay-token storage cuts the exact replay-token object in half while preserving continuation behavior on the tested panel
- even simple `int8` replay-token storage is much stronger than thinner late bands:
  - token agreement remains perfect on the tested panel
  - ranking fidelity degrades slightly, but far less catastrophically than the late-band reductions

### Current Compression Read

- if exact replay-token storage is allowed, `fp16` is already a strong practical surrogate
- if more aggressive compression is needed, `int8` replay-token storage is the first credible lossy candidate
- this means the next compression problem is now more specific:
  - not "compress late bands more"
  - but "compress the replay token better"

## Replay-Token Surrogates Across Replay Layers

The replay-token surrogate family was then widened across replay layers `9`, `10`, and `11` on the same fixed prompt-family panel.

### `fp16` Replay Token

Result:

- `fp16` replay-token storage is continuation-safe across the tested panel at replay layers `9`, `10`, and `11`
- token agreement stayed `1.00`
- top-5 full-overlap steps stayed `10/10`

This makes `fp16` the first compressed replay-token surrogate that is safe across the tested replay cuts.

### `int8` Replay Token

Result:

- `int8` replay-token storage remains strong, but it is no longer uniformly safe across replay layers
- replay layers `9` and `10`:
  - token agreement stayed `1.00` across the fixed panel
  - top-5 full-overlap typically fell to `8/10`, `9/10`, or `10/10`
- replay layer `11`:
  - still strong on most prompt families
  - but factual-simple dropped to token agreement `0.60` with first divergence at step `7`

Representative summary:

| Replay Layer | Object | Prompt Family | Token Agreement / 10 | Top-5 Full-Overlap Steps | Current Read |
| --- | --- | --- | ---: | ---: | --- |
| `9` | `token@9/fp16` | all tested families | `1.00` | `10/10` | Safe |
| `9` | `token@9/int8` | factual-simple | `1.00` | `9/10` | Strong but lossy |
| `9` | `token@9/int8` | narrative | `1.00` | `10/10` | Safe on tested horizon |
| `10` | `token@10/fp16` | all tested families | `1.00` | `10/10` | Safe |
| `10` | `token@10/int8` | procedural / instruction-like | `1.00` | `9/10` | Strong but lossy |
| `11` | `token@11/fp16` | all tested families | `1.00` | `10/10` | Safe |
| `11` | `token@11/int8` | factual-simple | `0.60` | `6/10` | Unsafe at this cut |
| `11` | `token@11/int8` | narrative | `1.00` | `8/10` | Strong but lossy |

### Updated Compression Frontier

- `fp16` replay-token compression dominates every previously tested late-band reduction
- `int8` replay-token compression is the first genuinely interesting lossy frontier:
  - much smaller than the exact replay token
  - often preserves perfect token agreement
  - but now shows real cut sensitivity at replay layer `11`

So the next compression question is sharper again:

- how far can replay-token compression go before cut-sensitive failure begins?

## Next Widening

- keep this table as the main artifact for the compact branch
- add more prompt families before adding many more object variants
- widen replay-layer coverage only after the prompt-family picture is clearer
- if a reduced object looks promising, require a continuation check before treating it as viable
