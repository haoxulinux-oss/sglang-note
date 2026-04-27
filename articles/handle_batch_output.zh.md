# `_handle_batch_output()` 函数解析

**位置：** `python/sglang/srt/managers/tokenizer_manager.py:1628-1863`

这是 `handle_loop` 收到「批量输出」消息后的核心分发函数。负责：
- 把 scheduler/detokenizer 一个批内的多个 rid 结果**拆开**；
- 为每个 rid 构造 `out_dict`(将被 yield 给 HTTP 响应)；
- 唤醒对应的请求协程；
- 对 `finished` 请求做计时、metrics、清理 `rid_to_state`、释放 LoRA。

---

## 一、三类批量输出结构的含义

它们都是 `python/sglang/srt/managers/io_struct.py` 中的 dataclass，是 **scheduler/detokenizer → tokenizer_manager** 跨进程消息的标准载体。共同点：所有字段都是按 `rids` 索引对齐的列表。

### 1. `BatchStrOutput` —— 含文本的生成结果(最常见)

走 `/generate`、`/v1/chat/completions` 等需要解码成字符串的请求时，**DetokenizerManager 处理完 token→text 转换后**回传。

主要字段：

| 字段 | 含义 |
|---|---|
| `rids: List[str]` | 这个批里每个请求的 id |
| `output_strs: List[str]` | 本步增量生成的文本(delta) |
| `output_ids: List[List[int]]` | 本步增量生成的 token id |
| `finished_reasons: List[Optional[dict]]` | 为 None 表示还在跑；非 None 表示结束(type=stop/length/abort，含 status_code/message) |
| `prompt_tokens / completion_tokens / cached_tokens / reasoning_tokens` | 计费 / metrics 用 |
| `cached_tokens_details` | RadixCache / HiCache 命中分层细节 |
| `retraction_counts` | 因 KV 不足回退(preempt-and-retract)发生过的次数 |
| `output_hidden_states` | 若请求要求返回隐藏状态 |
| `routed_experts` | MoE 模型按需返回的专家路由信息 |
| `time_stats` | scheduler 端的时间采样(queue/prefill/decode 等) |
| `load` | DP 模式下这个 rank 的当前负载快照 |
| `dp_ranks` | 哪个 DP rank 服务了这个请求 |
| `customized_info` | 模型/插件自定义信息 |

语义：**「scheduler 一轮 batch 内所有产生新增 token 的请求」的合集**，文本侧已被 detokenizer 处理。

### 2. `BatchTokenIDOutput` —— 纯 token id(跳过 detokenize)

当 server 启动时 `--skip-tokenizer-init`，TokenizerManager 不持有 tokenizer，detokenizer 也不会把 id 转文本。此时 scheduler 直接回传 token id。

字段相比 `BatchStrOutput` **少一个 `output_strs`**，其它几乎一样。客户端要自行 detokenize(适合 token-in / token-out 的下游服务、benchmark、speculative decoding 编排层)。

### 3. `BatchEmbeddingOutput` —— Embedding 推理结果

`/encode`、`/v1/embeddings`、cross-encoder 等不做生成、只取池化向量的任务。每个请求只产生**一次性结果**。

主要字段：

| 字段 | 含义 |
|---|---|
| `rids: List[str]` | 请求 id |
| `embeddings: List[List[float]]` | 池化后的向量(或 `[CLS]` 向量等) |
| `pooled_hidden_states` | 可选：池化前的 hidden states |
| `prompt_tokens / finished_reasons / retraction_counts / time_stats` | 与生成类共有的计费/状态字段 |

它没有 `output_ids / completion_tokens / cached_tokens` 这类**生成类**字段，这也是函数内 `if not isinstance(recv_obj, BatchEmbeddingOutput): meta_info.update({...completion_tokens...})` 的判断依据。

---

## 二、函数逐段解析

### 1. 开头：批量唤醒缓冲

```python
pending_notify: dict[str, ReqState] = {}
batch_notify_size = self.server_args.batch_notify_size
```

不是收到一条就立刻 `state.event.set()`，而是攒一小批再批量 set，以降低 asyncio 唤醒抖动。`batch_notify_size` 是可调的(默认小批量)。

### 2. 主循环：按 rid 拆分

```python
for i, rid in enumerate(recv_obj.rids):
    state = self.rid_to_state.get(rid, None)
    if state is None:
        logger.error(...)
        continue
```

对 batch 中每一条按 `rid` 找到对应的 `ReqState`。找不到通常是请求已被 abort 并从 `rid_to_state` 删除——这种「孤儿」结果直接丢弃并打 error 日志。

### 3. 构造通用 `meta_info`

```python
meta_info = {
    "id": rid,
    "finish_reason": recv_obj.finished_reasons[i],
    "prompt_tokens": recv_obj.prompt_tokens[i],
    "weight_version": self.server_args.weight_version,
    "total_retractions": recv_obj.retraction_counts[i],
}
```

`weight_version` 用于热更新权重后客户端能识别「这个结果来自哪个权重版本」。`total_retractions` 在做稳定性监控时很关键——retraction 多通常意味着 KV 池过载。

### 4. 可选扩展字段

- `enable_metrics + time_stats`：把 scheduler 阶段的时间采样(recv→queue→prefill→decode→send 等)拍进 `meta_info`，最终 client 能看到端到端 + 各阶段耗时。
- `return_logprob`：调 `convert_logprob_style` 把 token 级 logprob、top-k logprob 补进 `meta_info`。如果 `return_text_in_logprobs=True` 且 server 没 skip tokenizer，还会把 token id 反 detokenize 成文本片段。
- 非 embedding：补 `reasoning_tokens / completion_tokens / cached_tokens(_details)`。其中 `cached_tokens_details` 反映 RadixCache 命中、HiCache L1/L2、disk 等的命中量。
- `output_hidden_states`、`routed_experts`、`customized_info`、`dp_ranks` 都是按需追加。

### 5. 标记是否完成

```python
state.finished = recv_obj.finished_reasons[i] is not None
```

**这是「请求是否结束」的单点写入**。后续 `_wait_one_response` 看的就是这个 flag。

### 6. 三类分支构造 `out_dict`

#### 6.1 `BatchStrOutput` 分支

```python
delta_text = recv_obj.output_strs[i]
delta_output_ids = recv_obj.output_ids[i]
state.append_text(delta_text)
state.output_ids.extend(delta_output_ids)
```

把这一步的 delta 累积到 `state` 里。然后分三种模式产出：

- **incremental streaming**(`incremental_streaming_output=True 且 stream=True`)：直接吐 delta 文本+id，并通过 `_slice_streaming_output_meta_info` 把 logprob 等切片到本步，更新 `last_output_offset`。客户端拿到的是真正的「增量帧」。
- **non-incremental streaming**(默认 stream 模式)：
  - 中间帧 `text=None, output_ids=state.output_ids(引用，不拷贝)`，**把字符串重建延迟到 `_wait_one_response` 那边一次性 join**。这是为了避免每步 `''.join(many_chunks)` 的 O(n²) 累积开销。
  - 结束帧才 `state.get_text()` 真正拼成完整字符串。
- **non-stream**：只在 `state.finished` 时返回完整文本+id；中间步骤 `out_dict = None`，根本不入 `out_list`。

#### 6.2 `BatchTokenIDOutput` 分支

和上面类似，**少了 text 字段**。同样有 incremental / non-incremental / 结束三种模式。

#### 6.3 `BatchEmbeddingOutput` 分支

```python
out_dict = {
    "embedding": recv_obj.embeddings[i],
    "meta_info": meta_info,
}
if recv_obj.pooled_hidden_states is not None and recv_obj.pooled_hidden_states[i] is not None:
    out_dict["pooled_hidden_state"] = recv_obj.pooled_hidden_states[i]
```

embedding 任务没有「增量」概念，一次出结果就是终态。

### 7. 计时与完成处理

```python
if state.time_stats.first_token_time == 0.0:
    state.time_stats.set_first_token_time()
```

整个请求生命周期里 **first token 时间只写一次**(用来算 TTFT)。

```python
if state.finished:
    # 写 trace、e2e_latency
    # speculative decoding 额外指标
    # metrics
    del self.rid_to_state[rid]
    if state.obj.lora_path:
        asyncio.create_task(self.lora_registry.release(state.obj.lora_id))
```

完成时：
- 记录 e2e latency、scheduler 阶段拼起来的详尽时间样本。
- 从 `rid_to_state` **删除条目**(防止内存泄漏)。
- 异步释放 LoRA 引用计数(不阻塞当前 dispatch)。

### 8. 入队 + 批量唤醒

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

- 把 `out_dict` 塞进 `state.out_list`，等 `_wait_one_response` 来取。
- 攒够 `batch_notify_size` 就批量 `event.set()`，并 `await asyncio.sleep(0)` **主动让出**——让被唤醒的请求协程能立刻被调度走(否则一直在 for 循环里，整批处理完才有机会跑)。
- 这个让出是 SGLang 对「**SSE 首 token 延迟**」的关键优化：高并发时不让任何单个请求被一个大 batch 拦住。

### 9. 每 rid 末尾：可选 metrics / dump

```python
if self.enable_metrics and state.obj.log_metrics:
    self.collect_metrics(state, recv_obj, i)
if self.dump_requests_folder and state.finished and state.obj.log_metrics:
    self.dump_requests(state, out_dict)
if self.crash_dump_folder and state.finished and state.obj.log_metrics:
    self.record_request_for_crash_dump(state, out_dict)
```

- Prometheus / 自定义 metrics 采集。
- 结束时可选 dump 整条请求 → 用于离线回放、纠错、分析。
- crash dump：保留最近 N 条请求，崩溃后可帮助复现。

### 10. 收尾

```python
for s in pending_notify.values():
    s.event.set()
```

把循环尾部还没攒满 `batch_notify_size` 的剩余 set 出去。

```python
if (
    self.server_args.dp_size > 1
    and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
    and recv_obj.load is not None
):
    load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])
    self.send_to_scheduler.send_pyobj(load_update_req)
```

DP(data parallel)模式下，把这个 rank 报的最新负载**回送**给 scheduler / 路由层。下次 DP 路由就有更新后的负载信息可用。embedding 没有这条(DP load 由生成路径驱动)。

---

## 三、整体数据流总结

```
DetokenizerManager / Scheduler                      handle_loop()
  └─ send_pyobj(BatchStrOutput|BatchTokenIDOutput|   └─ recv_pyobj()
                BatchEmbeddingOutput, batch=N)            │
                                                          ▼
                                        _handle_batch_output(recv_obj)
                                                          │
                          ┌───────────────────────────────┴──────────────────────────┐
                          │  for i, rid in enumerate(recv_obj.rids):                 │
                          │     state = rid_to_state[rid]                            │
                          │     meta_info = {...rid 共有...}                          │
                          │     if BatchStrOutput   → 累积 delta 文本/id, 构造 out_dict│
                          │     if BatchTokenIDOutput → 累积 delta id, 构造 out_dict   │
                          │     if BatchEmbeddingOutput → 一次性 embedding out_dict    │
                          │     if state.finished: 写 metrics / del rid_to_state[rid] │
                          │     state.out_list.append(out_dict); pending_notify[rid]  │
                          │     if 攒够 batch_notify_size: event.set() + sleep(0)     │
                          └───────────────────────────────┬──────────────────────────┘
                                                          ▼
                                       剩余 pending_notify.values().event.set()
                                                          ▼
                          DP 模式: 把 recv_obj.load 回送给 scheduler 做负载感知路由
```

---

## 四、设计亮点 & 易踩坑点

1. **「一个 batch 消息对应 N 个请求结果」**。这是 SGLang 高吞吐的关键——前后端解耦，每次跨进程通信不是 1:1，而是 1:N。
2. **`pending_notify` 批量唤醒 + `await asyncio.sleep(0)`**：在公平性和事件循环抖动之间取得平衡，避免「先收到结果的请求要等所有 rid 处理完才被唤醒」。
3. **non-incremental streaming 故意把字符串拼接延后**：把 O(n²) 变成 O(n)。
4. **`del self.rid_to_state[rid]` 必须只在 `state.finished` 分支里发生一次**，否则会出现孤儿唤醒事件再也没人接收的状况。
5. **abort 分支在这里不单独处理**——abort 是 scheduler 端把请求标 `finished + finish_reason.type="abort"`，本函数依然走 finish 分支正常 yield 给 client(client 就能看到 abort 原因)。
6. **`BatchEmbeddingOutput` 不走 `state.append_text` / `output_ids` 累积**——避免在 embedding 任务上跑生成侧状态机。
7. **`load` 字段只在生成路径回传**——这也是为什么 DP 路由逻辑主要照顾生成负载。
