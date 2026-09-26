# llama.cpp patch: activation capture and steering

`llama-cpp-capture-activations.patch` adds per-request activation capture and additive steering to
`llama-server`. The toolkit needs it: every story comes back from `/completion` together with the
mean residual-stream activation at every layer over the tokens the model generated.

## Base and build

The patch is a plain `git diff` against upstream ggml-org/llama.cpp commit
`9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e` ("docs: improve Windows build instructions (#27381)",
2026-08-21). It applies cleanly there and builds `llama-server` (checked with a CPU build).
Later upstream revisions may need small manual fixes, because the server files change often.

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git checkout 9a286ac98d2cab74231bd3f1fc3f2b8bdf05422e
git apply /path/to/persona-emotion-vectors/patches/llama-cpp-capture-activations.patch
cmake -B build -DGGML_CUDA=ON          # or your backend
cmake --build build --target llama-server -j
```

Run with one slot and bind to localhost (the endpoints below are unauthenticated and some read or
write server-side file paths):

```bash
./build/bin/llama-server -m model.gguf --host 127.0.0.1 --port 8080 -np 1 --jinja
```

Speculative decoding must be off for capture: draft-verification batches mix accepted and
rejected tokens, so the server refuses `capture_activations` when a draft model is active.

## What it adds

All of it is off by default: with capture disabled the eval callback declines every tensor and no
response carries extra fields. Registering the callback does cost speed, though (see the note at
the end); `LLAMA_NO_ACTIVATION_CAPTURE=1` skips registering it for capture-free deployments.

**Per-request accumulation** (what this toolkit uses). `/completion` accepts

```json
{"capture_activations": true, "capture_skip": 24, "capture_layers": [10, 20]}
```

and the final result carries

```json
"activation_means": {"n_layers": 30, "n_embd": 2816, "skip": 24, "n_tokens_seen": 212,
                     "n_tokens_used": 187, "layers": {"0": [...], "1": [...]}}
```

the mean of each layer's residual output (`l_out`) over the decoded tokens after the first
`capture_skip`. Accumulation starts when the slot begins generating and ends when the final
response is built, so it rides back in the same request. The accumulator is global, so the server
must run with `-np 1` (otherwise the field holds an `error`). `capture_layers` limits the returned
layers (default: all).

**Additive steering.** `POST /activations/steer`

```json
{"add": [{"layer": 12, "vector": [...], "scale": 1.0}], "add_only_decode": true}
```

adds `scale * vector` to the residual at those layers on every decode step (a `"path"` to a raw
float32 file of `n_embd` values may replace `"vector"`). `{"clear_add": true}` removes it;
`{"clear": true}` clears all steering.

**Projection steering** (same endpoint, `"layers"` form): attenuates each token's component along
K unit directions per layer, `h' = h - alpha * sum_k (d_k . h) d_k`, with directions read from
`"dirs_path"` files; optionally adds back a fraction along M other directions (`"world_path"`,
`"beta"`). Not used by this toolkit.

**Live means and collection.** `GET /activations` returns the per-layer means of the last decoded
token batch (full vectors, or with `top_k` the largest dimensions by magnitude);
`POST /activations {"enabled": true}` toggles live capture;
`POST /activations/collect` streams per-token activations of one layer (or an FFN intermediate
tensor via `"tensor"`) to a binary file for SAE-style training. Not used by this toolkit.

**Eval callback on graph reuse.** A one-line change in `src/llama-context.cpp` sets the eval
callback on every ubatch, including when a compute graph is reused, so capture sees every step.

Note: a registered eval callback forces node-by-node graph evaluation (no CUDA graphs or backend
fusion), which is slower than an unpatched server even while capture is off. Run the patched
binary for the study and your usual binary otherwise, or set `LLAMA_NO_ACTIVATION_CAPTURE=1`.
Outside collection mode, prefill batches are declined by the callback.
