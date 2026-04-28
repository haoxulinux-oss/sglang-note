# Scheduler 如何从 ZMQ 拿请求并跑推理

**位置：** `python/sglang/srt/managers/scheduler.py`

承接前一篇：`TokenizerManager.generate_request()` 已经把 `TokenizedGenerateReqInput` 通过 ZMQ `PUSH` socket 发出去了。本篇追踪它在 Scheduler 进程里走完的整条路径——从 ZMQ 出队，到送进 model.forward，再到把结果回推给 DetokenizerManager。

---

## 一 进程入口与 event loop 分派

`run_scheduler_process()` 是子进程入口(`scheduler.py:3751`)，它构造 `Scheduler` 实例后调用 `scheduler.run_event_loop()`：

```python
def run_event_loop(self) -> None:
    self.schedule_stream = self.device_module.Stream(priority=0)
    if self.device == "cpu":
        self.schedule_stream.synchronize = lambda: None
    with self.device_module.StreamContext(self.schedule_stream):
        dispatch_event_loop(self)
```

`dispatch_event_loop()` 根据 server_args 选择具体哪一种 loop(`scheduler.py:3665`)：

| 条件 | 选用的 loop |
|---|---|
| `enable_pdmux` | `event_loop_pdmux` |
| `pp_size > 1` | `event_loop_pp` |
| `enable_overlap` | `event_loop_overlap`(默认 GPU 情况，最常见) |
| 否则 | `event_loop_normal` |
| 加上 disagg PD 模式还有 6 个变种 | `event_loop_*_disagg_*` |

所有 loop 都遵循同一个骨架——「**收请求 → 拼 batch → 跑 forward → 处理结果**」，只是为了支持流水线并行 / overlap / PD 拆分而布置不同。下面以 `event_loop_normal` 为参考。

---

## 二 `event_loop_normal` 的四步循环

```python
@DynamicGradMode()
def event_loop_normal(self):
    while True:
        # ① 拉新请求
        recv_reqs = self.recv_requests()
        self.process_input_requests(recv_reqs)
        if self._engine_paused:
            self.cancel_bubble_timer()
            continue

        # ② 组下一批
        batch = self.get_next_batch_to_run()
        self.cur_batch = batch

        # ③ 跑这一批
        if batch:
            result = self.run_batch(batch)
            self.process_batch_result(batch, result)
        else:
            self.on_idle()

        # ④ 记录上一批
        self.last_batch = batch
```

注意这是**同步 `while True`**，不是 asyncio 协程——Scheduler 进程只跑这一条主线程，所有耗时(forward、batch 组装、采样、KV 分配)都在这条线上。设计原则：「**ZMQ 异步收，主循环串行处理**」。

---

## 三 ① 收请求：`recv_requests()`

> 详细分析见 [`Scheduler.recv_requests()` 解析](scheduler-recv-requests.zh.md)。

源码：`scheduler.py:1504-1659`。核心三部分：

### 3.1 非阻塞拉满 ZMQ buffer

```python
while True:
    try:
        if self.recv_limit_reached(len(recv_reqs)):
            break
        recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
    except zmq.ZMQError:
        break
    recv_reqs.append(recv_req)
```

- `self.recv_from_tokenizer` 是 `PULL` socket，对端是 TokenizerManager 的 `PUSH`(连同 RPC socket)。
- **`zmq.NOBLOCK`**：和 TokenizerManager 那边的 `await recv_pyobj()` 完全相反——这边是「能拿就拿，拿空就跳出」。因为 Scheduler 主循环是同步的，不能阻塞在 socket 上(否则 batch 永远跑不到)。
- `recv_limit_reached`(`max_recv_per_poll`)：单轮最多拉多少请求，避免一次涌入太多让 forward 推迟。
- 还从 `recv_from_rpc` 拉控制平面消息(权重更新、profile、abort 等)。

### 3.2 跨 rank 广播

`recv_requests` 由 `pp_rank=0 + attn_tp_rank=0 + attn_cp_rank=0` 的「主 rank」执行，其他 rank 通过 `broadcast_pyobj` 拿到同一份 `recv_reqs`(`scheduler.py:1607-1613`)。这样所有 TP/CP rank 在同一时刻看到完全一致的请求集，避免 batch 组装时分裂。

DP attention 模式下还会把 `work_reqs`(实际推理请求) 和 `control_reqs`(权重更新等) 分开广播——因为 DP 各分组接收的请求不一样，但控制消息要全部同步。

### 3.3 共享内存材料展开

```python
for req in recv_reqs:
    unwrap_shm_features(req)
```

多模态张量是用 POSIX shm 跨进程传的——`recv_pyobj` 反序列化只会拿到 `ShmPointerMMData` 元信息(防止 `pickle.dumps` 把整张图片塞进 ZMQ frame)；这里再 `shm_open` 把 tensor 拿回来。注意 multimodal + tp_size>1 时插了一个 `barrier(group=self.tp_cpu_group)`——避免源 rank 在其他 rank 完成 `shm_open` 之前就 `shm_unlink`(详见代码注释)。

---

## 四 ② 把请求接入队列：`process_input_requests()`

> 详细分析见 [`Scheduler.process_input_requests()` 解析](scheduler-process-input-requests.zh.md)。

```python
for recv_req in recv_reqs:
    output = self._request_dispatcher(recv_req)
    if output is not None:
        if not isinstance(output, RpcReqOutput):
            self.send_to_tokenizer.send_output(output, recv_req)
        else:
            self.recv_from_rpc.send_pyobj(output)
```

`_request_dispatcher` 是基于类型的派发器(`scheduler.py:1277-1339`)，把 30+ 种 `*ReqInput` 对应到具体 handler。我们关心的是 `TokenizedGenerateReqInput → handle_generate_request`。

### 4.1 `handle_generate_request()`

> 详细分析见 [`Scheduler.handle_generate_request()` 解析](scheduler-handle-generate-request.zh.md)。

关键步骤(`scheduler.py:1827-2023`)：

1. **构造 `Req`**：把 `TokenizedGenerateReqInput`(只有 token_ids、sampling_params、stream 标志等) 包成 Scheduler 内部的 `Req` 对象——这是承载请求生命周期的核心数据结构(prefix、kv、output token 缓冲、time_stats、finish reason 都挂在它上面)。
   ```python
   req = Req(recv_req.rid, recv_req.input_text, recv_req.input_ids,
             recv_req.sampling_params, ...)
   req.tokenizer = self.tokenizer
   ```
2. **多模态展开**：如果 `mm_inputs` 非 `None`，调 `pad_input_ids_func` 把单个 image token 展开成多个占位 token(让 vision encoder 的 patch embedding 有位置可放)，并算好 mrope 位置编码。
3. **校验 prompt 长度**：`validate_input_length(req, max_req_input_len, allow_auto_truncate)`——超长就 abort 或截断。
4. **设置 `max_new_tokens`**：取 `min(用户给的, max_req_len - len(input_ids) - 1)`。
5. **走结构化输出预编译**：如果带 grammar(JSON schema / EBNF)，先扔进 `grammar_manager` 队列编译；编译完才 `_add_request_to_queue`。
6. **入队**：
   ```python
   self._add_request_to_queue(req)
   ```

### 4.2 `_add_request_to_queue()`

> 详细分析见 [`Scheduler._add_request_to_queue()` 解析](scheduler-add-request-to-queue.zh.md)。

按 disagg 模式不同(`scheduler.py:2058-2080`)：

| 模式 | 落到哪个队列 |
|---|---|
| `NULL`(单机/colocate) | `self.waiting_queue.append(req)` |
| `PREFILL`(分离的 P 节点) | `self.disagg_prefill_bootstrap_queue.add(...)` |
| `DECODE`(分离的 D 节点) | `self.disagg_decode_prealloc_queue.add(...)` |

NULL 模式还会 `_prefetch_kvcache(req)`——尝试把已知的 prefix 从 HiCache 主动加载到 GPU(命中长 prefix 的请求受益巨大)。

到此为止 ① + ② 完成：`Req` 已经躺在 `waiting_queue` 里等着被打包成 batch。

---

## 五 ③ 组 batch：`get_next_batch_to_run()`

> 详细分析见 [`Scheduler.get_next_batch_to_run()` 解析](scheduler-get-next-batch-to-run.zh.md)。

源码：`scheduler.py:2302-2411`。这是 SGLang **continuous batching** 的核心调度器。整体策略：

```
last_batch (上一轮 prefill 的 chunk) ─┐
                                     ├─→ merge 进 running_batch
running_batch (decode 中的请求)  ─────┘
                                          │
                                          ▼
                              prefill 新请求? get_new_batch_prefill()
                                          │
                          ┌───────────────┴───────────────┐
                          有新 prefill                   只有 decode
                          ▼                              ▼
                   返回 prefill batch              返回 decode batch
                                                  (update_running_batch)
```

### 5.1 合并 `last_batch` 到 `running_batch`

如果上一轮跑的是 EXTEND(prefill)，那些 prefill 完成的请求要进入 decode 阶段——用 `merge_batch` 合到 `running_batch`(`scheduler.py:2335-2362`)：

```python
if self.last_batch and self.last_batch.forward_mode.is_extend():
    self.last_batch.filter_batch(...)
    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch
        else:
            self.running_batch.merge_batch(self.last_batch)
```

`chunked_req`(因为 chunked prefill 还没跑完的请求) 会先 stash 出来，下一轮再回到队列里继续 chunk。

### 5.2 优先组 prefill 新批：`get_new_batch_prefill()`

> 详细分析见 [`Scheduler.get_new_batch_prefill()` 解析](scheduler-get-new-batch-prefill.zh.md)。

`scheduler.py:2419` 起。简化逻辑：

1. 从 `waiting_queue` 里选请求(可能用 priority、Radix prefix 长度、长度阈值等多种排序)。
2. 用 `PrefillAdder` **逐个尝试加入**——每加一个就检查：
   - KV pool 还有没有空位(`token_to_kv_pool` available size)？
   - Tree cache(RadixCache)能否复用 prefix(命中越多，新分配越少)？
   - 总 token 数有没有超过 `chunked_prefill_size`？
3. 加不下了就停，剩下的 `Req` 留在 `waiting_queue`。
4. 如果当前 prefill 长度超过 chunk 阈值，**chunk 这个请求**(只放部分 token 到本批，剩余下次接着 prefill 同一个请求)。

返回的 `ScheduleBatch` 是个统一对象——它既能装 prefill 也能装 decode，区别在 `forward_mode`。

### 5.3 没新 prefill 就跑 decode

```python
if new_batch is not None:
    ret = new_batch                               # prefill 优先
else:
    if not self.running_batch.is_empty():
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch
    else:
        ret = None                                # 完全空闲
```

`update_running_batch()`(`scheduler.py:2669`)：清掉已完成的请求；当 KV 池吃紧时**主动 retract**(把某些请求踢出去回到 waiting，把它们的 KV 让出来)；构造 decode 用的 forward_batch 元信息。详见 [`Scheduler.update_running_batch()` 解析](scheduler-update-running-batch.zh.md)。

**优先级**: prefill > decode。这是为了让新请求尽快首 token 化(降低 TTFT)，代价是已运行请求的 ITL 偶尔会被 prefill 拖慢——这就是 chunked prefill 的动机。

---

## 六 ④ 跑 forward：`run_batch()`

> 详细分析见 [`Scheduler.run_batch()` 解析](scheduler-run-batch.zh.md)。

源码：`scheduler.py:2767-2923`。核心是：

```python
worker_batch = batch.get_model_worker_batch()
batch_result = self.model_worker.forward_batch_generation(worker_batch)
batch.output_ids = batch_result.next_token_ids
```

`get_model_worker_batch()` 把 `ScheduleBatch` 投影成 `ModelWorkerBatch`——纯数据 dataclass，过滤掉 scheduler-only 字段，只剩 forward 必需的那些(input_ids、positions、req_pool_indices、seq_lens、attn_backend metadata、sampling_info、return_logprob …)。

`forward_batch_generation` 进入 `tp_worker` → `model_runner` → `model.forward`，其中：

- 准备 `ForwardBatch`(包括 attn backend 选 `flashinfer` / `fa3` / `triton` / `torch_native` …)。
- CUDA Graph 模式下 replay 已 capture 的 graph(decode 阶段尤其常见)。
- `model.forward(...)` 算出 logits。
- `Sampler` 按 `sampling_info` 抽下一个 token(temperature / top-k / top-p / min-p / penalty / structured constraint)。
- 返回 `GenerationBatchResult{next_token_ids, logits_output, ...}`。

### 6.1 Overlap 调度的特殊点

`event_loop_overlap` 走的路径有一个关键差异(`scheduler.py:2799-2845`)：

```python
future_indices = self.future_map.alloc_future_indices(bs)
with self.forward_stream_ctx, ...:
    self.forward_stream.wait_stream(self.schedule_stream)
    self.future_map.resolve_future(model_worker_batch)
    batch_result = self.model_worker.forward_batch_generation(model_worker_batch)
    batch_result.copy_done = self.device_module.Event()
    self.future_map.store_to_map(future_indices, batch_result)
    batch_result.copy_to_cpu(...)
future_indices_or_next_token_ids = -future_indices.indices
batch.output_ids = future_indices_or_next_token_ids
```

要点：

- forward 在**独立的 CUDA stream**(`forward_stream`) 上发，scheduler 主线程立即返回，**不等 GPU 完成**就开始组下一批 batch、做 KV 分配。
- 但下一批的 `input_ids` 来自上一批的 `next_token_ids` —— 此刻 GPU 还没算完，所以用 **future indices** 做占位(负数索引)；等到真的需要这批 token 时，`future_map.resolve_future` 把占位换成实际值。
- 效果是 CPU 端的调度逻辑(组 batch、metadata 准备、KV 计算) 与 GPU 端的 forward **完全重叠**——这是 SGLang 比同类引擎吞吐高的关键之一。

### 6.2 `process_batch_result()`

> 详细分析见 [`Scheduler.process_batch_result()` 解析](scheduler-process-batch-result.zh.md)。

`scheduler.py:2950` 起。按 `forward_mode` 分派到 `process_batch_result_decode / _prefill / _disagg_prefill / _prebuilt / _idle`。每条路径里都会：

1. 把新 token append 到对应 `req.output_ids`。
2. 检查 EOS / stop string / max_new_tokens — 命中就 `req.finished_reason = ...`。
3. 把 finished 请求从 `running_batch` 摘出来，返还其 KV 占用回 `token_to_kv_pool`，从 RadixCache 增减引用计数。
4. 调 `self.stream_output(reqs, return_logprob)` 把这批输出发给 DetokenizerManager。

---

## 七 推回 DetokenizerManager：`stream_output_generation()`

> 详细分析见 [`Scheduler.stream_output()` / `stream_output_generation()` 解析](scheduler-stream-output.zh.md)。

`scheduler_output_processor_mixin.py:910-1217`。每个请求把 `output_ids[read_offset:]`(自上次发送以来新增的部分) 收集到几个并列 list 里，按 batch 组装成 `BatchTokenIDOutput`：

```python
self.send_to_detokenizer.send_output(
    BatchTokenIDOutput(
        rids=rids,
        finished_reasons=finished_reasons,
        decoded_texts=decoded_texts,
        decode_ids=decode_ids_list,
        read_offsets=read_offsets,
        output_ids=output_ids,
        ...
    )
)
```

注意：

- **同一个 batch 里的多个请求合并发**——一次 ZMQ send，N 个 rid。这就是上一篇 `_handle_batch_output` 看到的 fan-out 来源。
- 推回去的是 **token id**，不是文本——detokenize 由 DetokenizerManager 这个独立进程做(把 tokenizer 的 detokenize CPU 开销从 scheduler 进程剥离)。
- DetokenizerManager 算出文本后再 PUSH 给 TokenizerManager 的 PULL socket → 进入上一篇 `handle_loop` → fan-out 到 `rid_to_state[rid].out_list` → 唤醒 `_wait_one_response` → 流回 HTTP/SSE。
- Embedding 任务走 `stream_output_embedding` 直接发 `BatchEmbeddingOutput`，跳过 detokenize 那一跳。

---

## 八 完整链路总览

```
                Tokenizer Manager 进程
   HTTP /generate ─→ generate_request() 协程
                          │
                          ▼
                    PUSH (ZMQ)
                          │
══════════════════════════╪══════════════════════════════════════
                          ▼
                Scheduler 进程 (主循环 while True)
                          │
   ┌──────────────────────┴────────────────────────────────────┐
   │ ① recv_requests()  PULL+NOBLOCK, broadcast 到所有 rank      │
   │ ② process_input_requests() → handle_generate_request()    │
   │     ├─ 构造 Req                                            │
   │     ├─ 多模态展开 / prompt 校验 / max_new_tokens             │
   │     ├─ grammar 编译(异步)                                   │
   │     └─ _add_request_to_queue() → waiting_queue              │
   │ ③ get_next_batch_to_run()                                  │
   │     ├─ merge last_batch 到 running_batch                    │
   │     ├─ get_new_batch_prefill()  (prefill 优先)              │
   │     └─ update_running_batch()    (无新 prefill 时 decode)   │
   │ ④ run_batch()                                              │
   │     └─ tp_worker.forward_batch_generation                  │
   │            └─ model.forward + Sampler                      │
   │ ⑤ process_batch_result()                                   │
   │     ├─ append next_token_ids 到 req.output_ids              │
   │     ├─ 检查 finish 条件 / 释放 KV                            │
   │     └─ stream_output()                                      │
   │             └─ send_to_detokenizer.send_output(            │
   │                    BatchTokenIDOutput(...))                │
   └────────────────────────────────────────────────────────────┘
                          │
                          ▼
                    PUSH (ZMQ)
                          │
══════════════════════════╪══════════════════════════════════════
                          ▼
                Detokenizer 进程
              token id → 文本(差量解码) → BatchStrOutput
                          │
                          ▼
                    PUSH (ZMQ)
                          │
══════════════════════════╪══════════════════════════════════════
                          ▼
                Tokenizer Manager 进程
              handle_loop ─→ _handle_batch_output
                          ├─ rid_to_state[rid].out_list.append(out)
                          └─ state.event.set()
                          │
                          ▼
              generate_request() 协程被 .event.wait() 唤醒
                          │
                          ▼
                  yield → SSE → HTTP client
```

---

## 九 为什么这样设计

**1. 主循环是同步 `while True`，不是 asyncio。** Scheduler 进程主线程的工作量(打包 batch、CUDA 调度、Sampler) 是 CPU/GPU 串行的，没有 IO 等待。引入 asyncio 只会徒增 ctx switch 开销。ZMQ 收消息靠 `NOBLOCK` 轮询，避免阻塞——以**主循环空转一圈的时间**作为天然的轮询粒度，没空闲时一次拉一堆，空闲时也只是空转走完一圈而已。

**2. `recv_pyobj(zmq.NOBLOCK)` 配合 `broadcast_pyobj`。** 主 rank 拉，其他 rank 同步——保证 TP 内每个 rank 在同一物理时刻看到一致的请求集，下游 batch 组装才能保持对齐(否则不同 rank 组出不同 batch，立刻死锁)。

**3. continuous batching 的几个关键决策：**
- prefill 优先：缩短 TTFT。
- chunked prefill：超长 prompt 拆 chunk，避免单条长 prefill 阻塞所有 decode。
- retraction：KV 不够时主动踢请求回队列，让出空间给更紧迫的批。

**4. Overlap scheduler。** CPU 调度和 GPU forward 跨 stream 重叠——通过 `future_indices` 占位让下一批 batch 可以在上一批 forward 还没返回时就组装好，是 SGLang 高吞吐的关键之一。

**5. 独立 DetokenizerManager 进程。** 把 detokenize 这种 CPU 任务从 scheduler 主循环剥离，scheduler 只发 token id，CPU 负担最小化——尤其是在 vocab 很大的模型上 detokenize 不便宜，单独进程让 GPU 利用率更稳定。
