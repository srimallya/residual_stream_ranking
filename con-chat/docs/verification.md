# con-chat Verification

Use Python 3.10+ for sidecar checks. On this machine, `python` is Python 2.7, so use `python3` or `.venv/bin/python`.

## Syntax Checks

```bash
python3 -m py_compile con-chat/sidecar/server.py con-chat/sidecar/kv_codec.py con-chat/sidecar/smoke_kv_codec.py
cd con-chat && npm run lint:syntax
```

## KV Codec Smoke Test

```bash
python3 con-chat/sidecar/smoke_kv_codec.py
```

The output reports original bytes, compressed bytes, compression ratio, and reconstruction relative error for fake KV tensors.

## Chat Template Path

Start the sidecar with a temporary DB:

```bash
CON_CHAT_PORT=4319 \
CON_CHAT_DB_PATH=/tmp/con-chat-verify.sqlite3 \
python3 con-chat/sidecar/server.py
```

Send a streamed request:

```bash
curl -N http://127.0.0.1:4319/v1/chat/stream \
  -H 'content-type: application/json' \
  -d '{"conversationId":"verify-thread","userText":"Say hello in one short sentence."}'
```

In sidecar logs, check for `official chat template prompt preview=...`. The preview should show the tokenizer-rendered Gemma format. If the local tokenizer snapshot lacks `chat_template`, the sidecar loads `sidecar/chat_template_gemma4.jinja`, copied from the official `google/gemma-4-E2B-it` Hugging Face repository.

## Streaming

The `curl -N` command should print multiple `event: token` Server-Sent Events before the final `event: done`. The renderer should append `delta` text into the pending assistant message. It should show thinking only when the sidecar sends a natural `thinkingDelta`; it should not create a permanent fake thinking block.

After completion, fetch the graph:

```bash
curl http://127.0.0.1:4319/v1/conversations/verify-thread/graph
```

The final assistant message in `messages` should match the visible streamed answer.

## KV Cache Reuse

Send two streamed turns to the same `conversationId`. The first turn should log `build session ... cache=...`; the second should log `reuse session ... cached_tokens=...`. A rebuild should only occur when persisted turn state no longer matches the in-memory session state.

## Rollover

Below `32768` active tokens:

- no memory object is created;
- no memory is injected into the prompt;
- chat behaves like normal Gemma using the official chat template.

Above `32768` active tokens:

- the oldest stable turn range is sealed;
- active tokens trim toward `CON_CHAT_ROLLOVER_TARGET_TOKENS`, default `30000`;
- the memory object stores summary text, anchors, token count, and KV compression accounting when enabled;
- the graph shows memory only after rollover.

To disable the KV codec while keeping normal chat intact:

```bash
CON_CHAT_KV_CODEC_ENABLED=0 python3 con-chat/sidecar/server.py
```

## Hygiene

Before committing, run:

```bash
git status --short --ignored
```

Source/doc changes are expected. Runtime DBs, `.playwright-cli/`, generated logs, and `con-chat/state/` should be ignored or absent.
