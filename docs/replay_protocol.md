# Replay Protocol

This document describes the protocol behind the repo's replay system so another developer can port it to a different model family, runtime, or preferred architecture.

The goal is not "huge live context." The goal is:

- bounded active inference state
- selective routed recall
- replay objects that reintroduce prior state without unbounded KV growth

The product target is:

`live_state + recalled_state + current_input -> next_state`

Where:

- `live_state` is the fresh working context
- `recalled_state` is memory selected from prior interaction
- `current_input` is the new user turn or task input

## Protocol Layers

The protocol in this repo has five layers.

1. Retrieval
- build a semantic pool from prior regions
- rerank that pool with temporal / graph signals
- refine locally around the best candidate

Current implementation:
- semantic pool
- temporal / PageRank-style rerank
- graph-local refinement

2. Replay Object
- choose a memory object for the selected region
- replay the smallest object that preserves behavior for the task

Current object families:
- exact replay token
- `fp16` replay token
- `int8` replay token
- late-band delta object
- plain text fallback

3. Resume Contract
- define the tensor boundary precisely
- resume from that boundary using the model's native forward semantics

This is architecture-specific. It is the most important invariant in the whole system.

4. Continuation Check
- compare resumed behavior against direct behavior
- only treat an object as valid if the resumed path preserves the target behavior

5. Memory Lifecycle
- log every memory object's retrieval and replay behavior
- keep memory tiers explicit
- allow reversible tier changes before destructive ones

## Required Invariants

Any backend port should preserve these invariants.

1. Boundary semantics must be explicit
- define whether a stored state is:
  - embedding output
  - post-layer output
  - pre-final-norm output
  - post-final-norm output
- never mix these in capture vs replay

2. Resume path must mirror the native model forward
- mask contract
- position handling
- norm placement
- lm head application
- cache / shared-state behavior

3. Replay validation must be staged
- phase 1: offline reconstruction
- phase 2A: resumed forward logit agreement
- phase 2B: next-token agreement
- phase 2C: short-horizon continuation

4. Exactness and behavioral agreement are different
- exact replay is stronger than behavioral replay
- compact replay should be reported honestly as:
  - exact
  - behaviorally aligned
  - lossy

## Porting Checklist

Use this when adding another model family.

1. Phase 1: trace only
- load the model locally from `models/`
- capture hidden states at declared layer boundaries
- verify offline reconstruction from:
  - boundary state
  - incremental deltas

2. Identify the native text stack
- find the text-only submodel actually used for generation
- identify:
  - embedding path
  - layer list
  - final norm
  - output head
  - any per-layer side inputs

3. Identify mask and position contract
- causal mask function
- sliding / local attention if present
- RoPE or learned position embeddings
- cache position or position ids

4. Identify extra carried state
- some models need more than the hidden state to resume correctly
- examples:
  - shared KV state
  - per-layer inputs
  - architecture-specific cached projections

5. Build a native `predict_from_hidden(...)`
- input:
  - hidden state at replay boundary
  - start layer
  - original attention mask
  - original input ids if needed
  - carried auxiliary state if needed
- output:
  - logits for the requested token

6. Validate 2A
- one prompt
- one boundary layer
- one token position
- compare direct logits vs resumed logits

7. Validate 2B
- compare direct next-token logits vs resumed next-token logits
- compare greedy next token and top-k set

8. Validate 2C
- short greedy continuation
- track first divergence step

Do not skip the ladder. If 2A fails, 2B and 2C are not real.

## Architecture Notes

### GPT-2-class

The GPT-2 path in this repo works because the replay code mirrors:

- `create_causal_mask(...)`
- explicit `position_ids`
- transformer block loop
- final `ln_f`
- `lm_head`

The original 2A failure was a mask-contract bug. High cosine was misleading. Exactness only arrived after the resumed path used the same causal-mask contract as the direct path.

### Gemma 4

Gemma required a native path. It is not a GPT-2 variant.

Important Gemma-specific pieces:

- text replay runs through the language model stack, not a GPT-2 `transformer.h` loop
- mask construction uses architecture-native mask mapping
- mixed `full_attention` / `sliding_attention` layers matter
- rotary position embeddings matter
- `per_layer_inputs` matter
- late cuts may require carried shared-KV side state from earlier producer layers

This is the main portability lesson:

Do not port by deleting guards. Port by rebuilding the model's actual resume contract.

Operational note:

- do not assume `mps` or another accelerator will help the whole harness uniformly
- for Gemma on this machine, the useful split is currently:
  - CPU for orchestration and control-heavy evaluation
  - `mps` for dense long-context model passes
- measure long prefill separately from harness overhead

## Compact Replay Strategy

The repo now has evidence for several object families.

Current practical ranking:

1. exact replay token
- safe by construction

2. `fp16` replay token
- strongest practical compact default so far

3. `int8` replay token
- first meaningful lossy boundary

4. thinner late-band delta objects
- informative, but not generally safe

Porting advice:

- start compact work from replay-token surrogates
- do not start by thinning delta bands
- compare compact objects against the exact replay baseline

Useful metrics:

- token agreement
- top-k overlap
- first divergence step
- bytes
- latency

## Routed Memory Protocol

The replay stack is only useful if routing finds the right prior region.

Current routed protocol:

1. semantic candidate pool
2. temporal / graph rerank inside the pool
3. local refinement around the best candidate
4. replay the selected region with:
  - exact token
  - `fp16`
  - `int8`
  - full local band
  - text fallback

The bridge question is:

On the same routed evidence, which replay object preserves behavior best, and does tracked replay beat plain text replay?

When porting to another model:

- keep the router fixed at first
- swap only the replay backend
- compare the same object panel before changing routing

## Memory Lifecycle Protocol

The long-term protocol is not just replay. It is replay plus memory hygiene.

Each memory object should track:

- object id
- kind
- bytes
- source region
- rank history
- top-k frequency
- downstream utility
- replay usage count
- jump score
- last useful timestamp
- tier
- pinned flag

Suggested tiers:

- `pinned`
- `warm`
- `cold`
- `archived`
- `pruned`

Policy order:

1. observe
2. score
3. suggest
4. allow reversible mutation
5. only later allow destructive mutation

Recommended asymmetry:

- cooling can be relatively quick because it is reversible
- warming should require stronger repeated evidence
- archiving should require repeated weakness with no resurgence

## What to Measure

When validating another backend or architecture, report both correctness and operational cost.

Correctness:

- offline reconstruction error
- resumed-forward logit agreement
- next-token agreement
- greedy continuation agreement
- top-k overlap
- first divergence step

Operational:

- wall-clock time
- model load time
- replay step latency
- memory object bytes
- whether GPU / accelerator backends help or hurt in practice

## Recommended Build Order

If you are building this on another model family:

1. phase 1 trace capture
2. phase 2A native resumed-forward
3. phase 2B next-token agreement
4. phase 2C short continuation
5. replay-token compact surrogates
6. routed bridge
7. memory ledger and tier lifecycle

This order matters. It keeps the system falsifiable instead of decorative.

## Practical Defaults

If you want a conservative default protocol today:

- retrieval: staged router
- safe compact object: replay token `fp16`
- stronger fallback: exact replay token or richer local band
- text fallback: use when replay object is insufficient
- active memory policy:
  - keep active state bounded
  - route recall explicitly
  - log memory utility continuously
  - cool weak objects conservatively
  - delay destructive forgetting

## Current Repo Evidence

As of the current repo state:

- GPT-2-class HF replay is exact through phase 2C
- Gemma 4 2B HF replay is exact through phase 2C
- routed replay with GPT-2 is discriminative enough to separate strong compact objects from lossy ones
- routed Gemma replay reaches the bridge correctly, but the current generic CPU harness is expensive enough that broader Gemma bridge sweeps need a lighter evaluation path
- Gemma on `mps` is viable for `8k`-token needle-style long-context prefill, while the surrounding bridge/orchestration work still fits CPU better on this machine

See also:

- [README.md](../README.md)
- [findings.md](./findings.md)
- [compact_frontier.md](./compact_frontier.md)
