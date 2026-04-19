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

Observed result:

- no tested compact object at replay layer `11` was continuation-safe on the fixed prompt-family panel
- even the full local band (`7,8,9,10,11`) failed early across every prompt family
- top-5 full-overlap steps stayed at `0/20` for the full local band on the tested panel

Representative result for the full local band:

| Prompt Family | Token Agreement / 20 | First Divergence Step | Top-5 Full-Overlap Steps | Current Read |
| --- | ---: | ---: | ---: | --- |
| factual-simple | `0.15` | `2` | `0/20` | Not safe |
| factual-compositional | `0.05` | `1` | `0/20` | Not safe |
| narrative | `0.10` | `1` | `0/20` | Not safe |
| procedural / instruction-like | `0.00` | `1` | `0/20` | Not safe |
| code-like completion | `0.05` | `2` | `0/20` | Not safe |

### What The Replay-Layer Probe Means

- replay-layer locality matters, not just delta depth
- the current "safe full local band" story holds at replay layers `9` and `10`
- that story does **not** hold at replay layer `11`
- therefore, the compact replay frontier depends on where the replay object re-enters the stack, not just how much late-band information it keeps

## Current Mechanism Read

On the tested GPT-2 compact frontier:

- replay layer `9`:
  - full local band `7,8,9` is continuation-safe
- replay layer `10`:
  - full local band `7,8,9,10` is continuation-safe
- replay layer `11`:
  - even the full local band `7,8,9,10,11` is not continuation-safe

So the current safe-object story is more specific than "keep the full late band":

- a full local late band is sufficient at some replay layers
- but not at all replay layers

## Next Widening

- keep this table as the main artifact for the compact branch
- add more prompt families before adding many more object variants
- widen replay-layer coverage only after the prompt-family picture is clearer
- if a reduced object looks promising, require a continuation check before treating it as viable
