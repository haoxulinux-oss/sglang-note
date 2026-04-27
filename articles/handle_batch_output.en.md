# Analysis of `_handle_batch_output()`

**Location:** `python/sglang/srt/managers/tokenizer_manager.py:1628-1863`

This is the core dispatcher that `handle_loop` calls when a "batch output" message arrives. It is responsible for:
- **Splitting** a single batch from scheduler/detokenizer into per-`rid` results;
- Constructing an `out_dict` for each `rid` (later yielded to the HTTP response);
- Waking up the corresponding request coroutines;
- For `finished` requests: timing, metrics, cleaning up `rid_to_state`, releasing LoRA.

---

## 1. The three batch-output structures

All three are dataclasses defined in `python/sglang/srt/managers/io_struct.py`. They are the standard cross-process payloads from **scheduler/detokenizer → tokenizer_manager**. They share one shape: every field is a list aligned by index with `rids`.

### 1.1 `BatchStrOutput` — text generation result (most common)

Used for `/generate`, `/v1/chat/completions`, etc. that need string output. Sent **after the DetokenizerManager has converted token ids to text**.

Main fields:

| Field | Meaning |
|---|---|
| `rids: List[str]` | Request id for each element in the batch |
| `output_strs: List[str]` | Newly generated text delta for this step |
| `output_ids: List[List[int]]` | Newly generated token ids for this step |
| `finished_reasons: List[Optional[dict]]` | None = still running; non-None = finished (type=stop/length/abort, with status_code/message) |
| `prompt_tokens / completion_tokens / cached_tokens / reasoning_tokens` | Billing / metrics |
| `cached_tokens_details` | RadixCache / HiCache hit breakdown by tier |
| `retraction_counts` | Number of preempt-and-retract events (KV evictions) |
| `output_hidden_states` | Hidden states, if requested |
| `routed_experts` | Per-token MoE expert routing info |
| `time_stats` | Scheduler-side time samples (queue / prefill / decode / send) |
| `load` | Current load snapshot for this rank in DP mode |
| `dp_ranks` | Which DP rank served the request |
| `customized_info` | Custom info from models / plugins |

Semantics: **the union of "all requests that produced new tokens this batch step"**, with text already produced by the detokenizer.

### 1.2 `BatchTokenIDOutput` — raw token ids (skip detokenize)

When the server starts with `--skip-tokenizer-init`, the TokenizerManager owns no tokenizer and the detokenizer does not convert ids to text. The scheduler then sends token ids directly.

Compared to `BatchStrOutput` it **lacks `output_strs`**; otherwise nearly identical. Clients must detokenize themselves (suitable for token-in / token-out downstream services, benchmarks, speculative decoding orchestrators).

### 1.3 `BatchEmbeddingOutput` — embedding inference result

Used by `/encode`, `/v1/embeddings`, cross-encoders — tasks that compute a pooled vector and don't generate. Each request produces a **single one-shot result**.

Main fields:

| Field | Meaning |
|---|---|
| `rids: List[str]` | Request ids |
| `embeddings: List[List[float]]` | Pooled vectors (or `[CLS]` vectors etc.) |
| `pooled_hidden_states` | Optional: pre-pool hidden states |
| `prompt_tokens / finished_reasons / retraction_counts / time_stats` | Billing / state fields shared with generation |

It does not carry generation-side fields like `output_ids / completion_tokens / cached_tokens`. That is exactly what the function's `if not isinstance(recv_obj, BatchEmbeddingOutput): meta_info.update({...completion_tokens...})` guard checks.

---

## 2. Step-by-step walkthrough

### 2.1 Header: batch-notify buffer

```python
pending_notify: dict[str, ReqState] = {}
batch_notify_size = self.server_args.batch_notify_size
```

Instead of calling `state.event.set()` immediately for every rid, we accumulate a small batch and set them together — this reduces asyncio wake-up jitter. `batch_notify_size` is tunable.

### 2.2 Main loop: split by rid

```python
for i, rid in enumerate(recv_obj.rids):
    state = self.rid_to_state.get(rid, None)
    if state is None:
        logger.error(...)
        continue
```

For each batch entry, look up the corresponding `ReqState` by `rid`. Missing usually means the request was already aborted and removed from `rid_to_state` — these "orphan" results are dropped with an error log.

### 2.3 Build the common `meta_info`

```python
meta_info = {
    "id": rid,
    "finish_reason": recv_obj.finished_reasons[i],
    "prompt_tokens": recv_obj.prompt_tokens[i],
    "weight_version": self.server_args.weight_version,
    "total_retractions": recv_obj.retraction_counts[i],
}
```

`weight_version` lets clients identify which weight version produced this result after a hot weight update. `total_retractions` is critical for stability monitoring — frequent retractions usually mean the KV pool is overloaded.

### 2.4 Optional extension fields

- `enable_metrics + time_stats`: stamp scheduler-stage time samples (recv → queue → prefill → decode → send) into `meta_info`, so the client can see end-to-end + per-stage latency.
- `return_logprob`: call `convert_logprob_style` to add per-token logprobs and top-k logprobs to `meta_info`. With `return_text_in_logprobs=True` and tokenizer enabled, token ids are also detokenized into text snippets.
- Non-embedding: add `reasoning_tokens / completion_tokens / cached_tokens(_details)`. `cached_tokens_details` reflects hits in RadixCache, HiCache L1/L2, disk, etc.
- `output_hidden_states`, `routed_experts`, `customized_info`, `dp_ranks` are appended on demand.

### 2.5 Mark finished

```python
state.finished = recv_obj.finished_reasons[i] is not None
```

**This is the single write site for "is this request done."** `_wait_one_response` reads this flag.

### 2.6 Three branches building `out_dict`

#### 2.6.1 `BatchStrOutput` branch

```python
delta_text = recv_obj.output_strs[i]
delta_output_ids = recv_obj.output_ids[i]
state.append_text(delta_text)
state.output_ids.extend(delta_output_ids)
```

Accumulate this step's delta into `state`. Then output in one of three modes:

- **incremental streaming** (`incremental_streaming_output=True` and `stream=True`): emit the delta text/ids directly, slice logprobs with `_slice_streaming_output_meta_info`, and update `last_output_offset`. The client sees true incremental frames.
- **non-incremental streaming** (default streaming mode):
  - Intermediate frames carry `text=None, output_ids=state.output_ids` (a reference, not a copy). The string is **rebuilt later in `_wait_one_response` once**, avoiding the O(n²) cost of `''.join(many_chunks)` at every step.
  - Only the final frame calls `state.get_text()` to materialize the full string.
- **non-stream**: only emit the full text+ids when `state.finished`; intermediate steps return `out_dict = None` and are not appended to `out_list`.

#### 2.6.2 `BatchTokenIDOutput` branch

Same as above but **without the text field**. Same incremental / non-incremental / final modes.

#### 2.6.3 `BatchEmbeddingOutput` branch

```python
out_dict = {
    "embedding": recv_obj.embeddings[i],
    "meta_info": meta_info,
}
if recv_obj.pooled_hidden_states is not None and recv_obj.pooled_hidden_states[i] is not None:
    out_dict["pooled_hidden_state"] = recv_obj.pooled_hidden_states[i]
```

Embedding tasks have no notion of "incremental"; one shot is the final result.

### 2.7 Timing and finish handling

```python
if state.time_stats.first_token_time == 0.0:
    state.time_stats.set_first_token_time()
```

The **first-token time is written exactly once** during the request's lifetime (used to compute TTFT).

```python
if state.finished:
    # write trace, e2e_latency
    # speculative-decoding extra metrics
    # metrics
    del self.rid_to_state[rid]
    if state.obj.lora_path:
        asyncio.create_task(self.lora_registry.release(state.obj.lora_id))
```

On finish:
- Record e2e latency and a detailed scheduler time-sample bundle.
- **Delete** the entry from `rid_to_state` (preventing memory leaks).
- Asynchronously release the LoRA refcount (does not block dispatch).

### 2.8 Enqueue and batched wake-up

```python
if out_dict is not None:
    state.out_list.append(out_dict)
    pending_notify[rid] = state

    if len(pending_notify) >= batch_notify_size:
        for s in pending_notify.values():
            s.event.set()
        pending_notify = {}
        await asyncio.sleep(0)
```

- Append `out_dict` to `state.out_list` for `_wait_one_response` to pick up.
- Once `pending_notify` reaches `batch_notify_size`, batch-set the events and **`await asyncio.sleep(0)`** to **yield voluntarily** — this lets the woken request coroutines run immediately rather than waiting until the entire for-loop finishes.
- This yield is the key SGLang optimization for **SSE first-token latency** under high concurrency: no single request is held back by a large batch.

### 2.9 Per-rid tail: optional metrics / dump

```python
if self.enable_metrics and state.obj.log_metrics:
    self.collect_metrics(state, recv_obj, i)
if self.dump_requests_folder and state.finished and state.obj.log_metrics:
    self.dump_requests(state, out_dict)
if self.crash_dump_folder and state.finished and state.obj.log_metrics:
    self.record_request_for_crash_dump(state, out_dict)
```

- Prometheus / custom metrics collection.
- On finish, optionally dump the entire request → useful for offline replay, debugging, analysis.
- Crash dump: keep the last N requests for post-mortem reproduction.

### 2.10 Tail

```python
for s in pending_notify.values():
    s.event.set()
```

Set events for whatever didn't reach `batch_notify_size` at loop exit.

```python
if (
    self.server_args.dp_size > 1
    and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
    and recv_obj.load is not None
):
    load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])
    self.send_to_scheduler.send_pyobj(load_update_req)
```

In DP (data-parallel) mode, **forward the latest load reported by this rank** back to the scheduler / router so the next DP routing decision sees fresh load info. Embedding does not carry this (DP load is driven by the generation path).

---

## 3. Overall data flow

```
DetokenizerManager / Scheduler                      handle_loop()
  └─ send_pyobj(BatchStrOutput|BatchTokenIDOutput|   └─ recv_pyobj()
                BatchEmbeddingOutput, batch=N)            │
                                                          ▼
                                        _handle_batch_output(recv_obj)
                                                          │
        ┌─────────────────────────────────────────────────┴─────────────────────────────┐
        │  for i, rid in enumerate(recv_obj.rids):                                      │
        │     state = rid_to_state[rid]                                                 │
        │     meta_info = {...common per-rid...}                                        │
        │     if BatchStrOutput   → accumulate delta text/ids, build out_dict           │
        │     if BatchTokenIDOutput → accumulate delta ids, build out_dict              │
        │     if BatchEmbeddingOutput → one-shot embedding out_dict                     │
        │     if state.finished: write metrics / del rid_to_state[rid]                  │
        │     state.out_list.append(out_dict); pending_notify[rid] = state              │
        │     if reached batch_notify_size: event.set() + asyncio.sleep(0)              │
        └─────────────────────────────────────────────────┬─────────────────────────────┘
                                                          ▼
                                Set events for any leftover pending_notify entries
                                                          ▼
                          DP mode: forward recv_obj.load to scheduler for load-aware routing
```

---

## 4. Design highlights & gotchas

1. **"One batch message → N request results."** This is the key to SGLang's throughput — the front and back ends are decoupled by 1:N cross-process messages, not 1:1.
2. **`pending_notify` batched wake-up + `await asyncio.sleep(0)`** balances fairness and event-loop jitter so that earlier rids don't have to wait until the entire for-loop completes.
3. **Non-incremental streaming defers string concatenation**: turns O(n²) into O(n).
4. **`del self.rid_to_state[rid]` must happen exactly once in the `state.finished` branch**; otherwise an orphan event might be set with no consumer.
5. **No special-case for abort here** — abort is signaled by the scheduler marking the request `finished` with `finish_reason.type="abort"`. This function still routes through the finished branch and yields normally to the client (which sees the abort reason).
6. **`BatchEmbeddingOutput` does not call `state.append_text` / extend `output_ids`** — it must not run the generation-side state machine.
7. **`load` is only forwarded on the generation path** — which is why DP routing is mainly tuned for generation load.
