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

## What The Table Means

- One-step quality is not a sufficient certificate of continuation stability.
- Full top-k overlap is stronger than greedy-token match, but it still does not guarantee continued alignment for thinner objects.
- The frontier is non-smooth:
  - `depth 3` looks strong on the two factual prompts
  - the same object fails badly on the narrative prompt
- The current safe compact object is the full local late band:
  - boundary layer `6`
  - replay layer `10`
  - deltas `7,8,9,10`

## Current Claim

Reduced replay objects can preserve one-step behavior without preserving iterative continuation. On the tested set, the full local late-band object is the only continuation-faithful compact replay object so far.

## Next Widening

- keep this table as the main artifact for the compact branch
- add more prompt families before adding many more object variants
- widen replay-layer coverage only after the prompt-family picture is clearer
- if a reduced object looks promising, require a continuation check before treating it as viable
