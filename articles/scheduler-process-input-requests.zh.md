# `Scheduler.process_input_requests()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:1690-1713`

**角色：** 把 `recv_requests()` 拉回来的 `*ReqInput` 列表逐个走 `_request_dispatcher` 派发到具体 handler。这是 scheduler 进程的「请求路由器」。

---

## 一 函数实现

```python
def process_input_requests(self, recv_reqs: List):
    now = time.monotonic()
    self.session_controller.maybe_reap(now)
    for recv_req in recv_reqs:
        # Skip health check when server is busy — ongoing requests already carry health info.
        if is_health_check_generate_req(recv_req) and not self.is_fully_idle(
            for_health_check=True
        ):
            self.return_health_check_ipcs.append(
                getattr(recv_req, "http_worker_ipc", None)
            )
            continue

        output = self._request_dispatcher(recv_req)
        if output is not None:
            if not isinstance(output, RpcReqOutput):
                self.send_to_tokenizer.send_output(output, recv_req)
            else:
                if self.recv_from_rpc is not None:
                    self.recv_from_rpc.send_pyobj(output)

    self._check_pending_flush()
    if self.external_corpus_manager is not None:
        self.external_corpus_manager.check_pending_load()
```

---

## 二 健康检查特殊处理

```python
if is_health_check_generate_req(recv_req) and not self.is_fully_idle(...):
    self.return_health_check_ipcs.append(getattr(recv_req, "http_worker_ipc", None))
    continue
```

健康检查走的是一个特殊的 `TokenizedGenerateReqInput`(rid 为 `HEALTH_CHECK_GENERATE_REQ_RID_PREFIX + ...`)。当 server 繁忙时：

- 不必真的把它丢进 waiting_queue 跑 forward——这会让健康检查被前面的长 prefill 阻塞，结果是「健康检查超时 → server 被外部判活失败 → 重启」。
- 直接把 `http_worker_ipc` 收集到 `return_health_check_ipcs`，等 `process_batch_result()` 后由 `maybe_send_health_check_signal()` 把 `HealthCheckOutput` 推回——告诉外部「在跑请求 = 我活着」。

`is_fully_idle(for_health_check=True)`：判断当前没有任何 prefill/decode 请求时返回 True，此时才让健康检查请求像普通请求一样走完整流程(测一遍真实推理路径)。

---

## 三 `_request_dispatcher` 类型派发

`_request_dispatcher` 是 `TypeBasedDispatcher`(`scheduler.py:1277-1339` 注册)，按 `type(recv_req)` 找对应 handler：

```python
self._request_dispatcher = TypeBasedDispatcher([
    (TokenizedGenerateReqInput, self.handle_generate_request),
    (TokenizedEmbeddingReqInput, self.handle_embedding_request),
    (BatchTokenizedGenerateReqInput, self.handle_batch_generate_request),
    (BatchTokenizedEmbeddingReqInput, self.handle_batch_embedding_request),
    (FlushCacheReqInput, self.flush_cache_wrapped),
    (ClearHiCacheReqInput, self.clear_hicache_storage_wrapped),
    (AbortReq, self.abort_request),
    (OpenSessionReqInput, self.open_session),
    (CloseSessionReqInput, self.close_session),
    (UpdateWeightFromDiskReqInput, self.update_weights_from_disk),
    (InitWeightsUpdateGroupReqInput, self.init_weights_update_group),
    (DestroyWeightsUpdateGroupReqInput, self.destroy_weights_update_group),
    (UpdateWeightsFromDistributedReqInput, self.update_weights_from_distributed),
    (UpdateWeightsFromTensorReqInput, self.update_weights_from_tensor),
    (UpdateWeightsFromIPCReqInput, self.update_weights_from_ipc),
    (GetWeightsByNameReqInput, self.get_weights_by_name),
    (ReleaseMemoryOccupationReqInput, self.release_memory_occupation),
    (ResumeMemoryOccupationReqInput, self.resume_memory_occupation),
    (CheckWeightsReqInput, self.check_weights),
    (SlowDownReqInput, self.slow_down),
    (ProfileReq, self.profile),
    (FreezeGCReq, self.handle_freeze_gc),
    (GetInternalStateReq, self.get_internal_state),
    (SetInternalStateReq, self.set_internal_state),
    (RpcReqInput, self.handle_rpc_request),
    (ExpertDistributionReq, self.expert_distribution_handle),
    (LoadLoRAAdapterReqInput, self.load_lora_adapter),
    (LoadLoRAAdapterFromTensorsReqInput, self.load_lora_adapter_from_tensors),
    (UnloadLoRAAdapterReqInput, self.unload_lora_adapter),
    (GetLoadsReqInput, self.get_loads),
    (PauseGenerationReqInput, self.pause_generation),
    (ContinueGenerationReqInput, self.continue_generation),
    (DumperControlReqInput, self.handle_dumper_control),
    (AddExternalCorpusReqInput, self.add_external_corpus),
    ...
])
```

按用途分三类：

| 类别 | 例子 | handler 行为 |
|---|---|---|
| **数据面**(高频) | `TokenizedGenerateReqInput`, `TokenizedEmbeddingReqInput` | 构造 `Req` → 入 waiting_queue，**立即返回 None**(不直接产生 output) |
| **控制面**(低频) | `FlushCacheReqInput`, `ProfileReq`, `OpenSessionReqInput` | 即时执行，**返回 `*Output`** 让调用方推回 |
| **权重管理** | `UpdateWeightFromDiskReqInput`, `LoadLoRAAdapterReqInput` | 调 model_runner / lora_manager 完成更新，返回 ack |

`TypeBasedDispatcher.__call__` 实现就是按类型查 dict 找对应 callable 调一下，没找到抛异常。

---

## 四 output 的回送路径

```python
output = self._request_dispatcher(recv_req)
if output is not None:
    if not isinstance(output, RpcReqOutput):
        self.send_to_tokenizer.send_output(output, recv_req)
    else:
        if self.recv_from_rpc is not None:
            self.recv_from_rpc.send_pyobj(output)
```

- **`output is None`**：常规情况(数据面请求)——`Req` 进队列等着被 batch 组进来跑，没有同步返回值。结果会通过 `stream_output` 异步推回。
- **`output` 不是 `RpcReqOutput`**：走标准回路 `send_to_tokenizer.send_output` 推给 TokenizerManager(它的 `_result_dispatcher` 处理)。`send_output` 会按 `recv_req.http_worker_ipc` 决定推给哪个 worker(多 worker 模式)。
- **`output` 是 `RpcReqOutput`**：走 RPC socket 回送(`recv_from_rpc.send_pyobj`)。RPC 路径独立于数据面 socket。

---

## 五 收尾：pending flush 与 corpus 加载

```python
self._check_pending_flush()
if self.external_corpus_manager is not None:
    self.external_corpus_manager.check_pending_load()
```

- `_check_pending_flush`(`scheduler.py:2984-3008`)：处理那些「需要等空闲才能执行的 flush_cache」——服务端忙时不能直接 flush(会破坏 running batch)，所以记一个 `_pending_flush`，每轮 input 处理完检查：当前是否完全空闲？是就 flush + ack。
- `external_corpus_manager.check_pending_load`：n-gram speculative decoding 用的外部语料异步加载状态机。

这两个 hook 都是**幂等的延迟任务**，借 input 处理这个时机点检查一下进度。

---

## 六 `session_controller.maybe_reap(now)`

```python
now = time.monotonic()
self.session_controller.maybe_reap(now)
```

`session_controller` 维护多轮会话的状态(prefill 后保留 KV，下一轮 user message 接着用)。`maybe_reap` 周期性回收：

- 已 close 但还在等当前轮跑完的 session。
- 超时无活动的 session(避免内存泄漏)。

放在 `process_input_requests` 开头是因为：
1. 每轮主循环都会调 process，自然形成一个轻量「定时器」。
2. 在创建新请求(可能 reuse session)之前完成清理，让 session 表保持干净。

---

## 七 设计要点小结

| 决策 | 原因 |
|---|---|
| 健康检查在 busy 时不入队 | 避免被前面的长 prefill 阻塞导致误判失活 |
| `TypeBasedDispatcher` 注册 30+ 类型 | 数据 / 控制 / 权重 / LoRA / profile 共享同一个入口，统一序列化 |
| 数据面 handler 返回 None | 真正结果通过 `stream_output` 异步推回，不在此处同步等 |
| RpcReqOutput 走独立 socket | RPC 控制流量与数据面解耦 |
| pending flush + corpus 检查放在循环末尾 | 借 process 的高频调度做轻量定时器，不需要单独线程 |

---

## 八 与上下文的衔接

```
recv_requests()  ─→  process_input_requests()  ─→  _request_dispatcher
                                                          │
                                                          ├─ handle_generate_request → _add_request_to_queue
                                                          ├─ flush_cache_wrapped → 返回 ack
                                                          └─ ... 其他 handler
```

`handle_generate_request` 等具体 handler 的细节见后续文章。
