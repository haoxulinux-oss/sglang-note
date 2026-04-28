# `Scheduler.process_batch_result()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:2950-2971`,具体子例程在 `scheduler_output_processor_mixin.py`。

**角色：** `run_batch` 之后的下一步——把 forward 产出的 `next_token_ids / embeddings / logprobs` 落到每个 `Req` 上,检查停止条件,释放 KV,最终把这一批输出推给 DetokenizerManager。

---

## 一 顶层分派

```python
def process_batch_result(
    self,
    batch: ScheduleBatch,
    result: Union[GenerationBatchResult, EmbeddingBatchResult],
):
    if batch.forward_mode.is_decode():
        self.process_batch_result_decode(batch, result)
    elif batch.forward_mode.is_extend():
        if batch.is_dllm():
            self.process_batch_result_dllm(batch, result)
        elif self.disaggregation_mode == DisaggregationMode.PREFILL:
            self.process_batch_result_disagg_prefill(batch, result)
        else:
            self.process_batch_result_prefill(batch, result)
    elif batch.forward_mode.is_prebuilt():
        self.process_batch_result_prebuilt(batch)
    elif batch.forward_mode.is_idle():
        self.process_batch_result_idle(batch, result)

    self.log_batch_result_stats(batch, result)
    self._maybe_clear_mm_inputs(batch)
    self.maybe_send_health_check_signal()
```

按 forward_mode 派发到 5 条子路径。最后三个 hook 是公共收尾。

---

## 二 5 种 forward_mode 的处理路径

| forward_mode | 路径 | 用途 |
|---|---|---|
| `DECODE` | `process_batch_result_decode` | 普通解码:每个 req 拼一个新 token,检查 EOS / stop |
| `EXTEND`(prefill) | `process_batch_result_prefill` | 处理 prefill 完毕的请求(下一轮通过 last_batch merge 进 running) |
| `EXTEND` + `is_dllm()` | `process_batch_result_dllm` | diffusion-LLM 的 prefill 结果(并行 token 解码,不一样的状态机) |
| `EXTEND` + `DisaggregationMode.PREFILL` | `process_batch_result_disagg_prefill` | PD 解耦 P 节点:把 KV 推给 D 节点,P 这边即可释放 |
| `is_prebuilt()` | `process_batch_result_prebuilt` | PD 解耦 D 节点:KV 已预接收,本批占位走完即可 |
| `is_idle()` | `process_batch_result_idle` | DP attention 的 idle batch:rank 没请求时陪跑,本身无输出 |

---

## 三 `process_batch_result_decode` 关键步骤(简化)

(具体源码在 `scheduler_output_processor_mixin.py:97` 起)

1. **同步等 GPU 拷贝**：`batch_result.copy_done.wait()`(overlap 模式),拿到 CPU 上的 `next_token_ids`。
2. **遍历 batch 内每个 req**:
   - `req.output_ids.append(next_token_id)`
   - `req.completion_tokens += 1`
   - 检查停止条件:EOS、stop string(用 incremental detokenize 后的尾部判定)、`max_new_tokens` 达到、grammar 匹配完毕。命中就 `req.finished_reason = FINISH_*`。
   - logprob 处理(如果 return_logprob)。
   - hidden_states / routed_experts 收集(如果 return_*)。
3. **释放完成的 req**:
   - `tree_cache.cache_finished_req(req)`:把已生成 token 注入 RadixCache(供后续相同 prompt 命中)。
   - 释放 `req_to_token_pool` slot 和 KV slot。
   - 从 `running_batch.reqs` 中移除(下一轮 `update_running_batch.filter_batch` 物理清理)。
4. **`stream_output(batch.reqs, batch.return_logprob)`**:把这一批新 token 推给 DetokenizerManager(详见 `stream_output_generation` 单独文章)。

---

## 四 `process_batch_result_prefill` 关键步骤

prefill 完成的请求要做几件 decode 路径不需要的事：

1. **incremental logprob 拼接**:prefill 阶段算了每个 input token 的 logprob(如果 return_logprob),把它们写入 `req.input_token_logprobs_*`。
2. **chunked prefill 状态更新**:如果是 chunked req 的中间轮次,只更新 `prefix_indices` 和 `extend_input_len`,**不**生成 next_token(还要继续 chunk)。最后一轮才走 sample 拿到 next_token。
3. **从 prefill 进入 decode 阶段**:filter 出真正完成 prefill 的请求(`is_chunked == 0`),把它们的状态切到 decode 模式——但物理 merge 要等下一轮 `get_next_batch_to_run` 用 `last_batch` 做。
4. **第一个 token 的停止检查**:有些极端 prompt 在 prefill 第一个 token 就触发 EOS,这里也要检查。
5. `stream_output`:把 prefill 完成的请求推回(包括 `prompt_logprobs` 等首次可用的元数据)。

---

## 五 `process_batch_result_disagg_prefill` 特殊点

PD 解耦的 P 节点完成 prefill 后:

1. 把 `prefill_logprobs / first_token` 推回给 client(让 client 立即看到 TTFT)。
2. 触发 KV 传输:把这个请求的 KV 通过 `transfer_backend`(NIXL/Mooncake/Mori) 推到对端 D 节点。
3. P 节点本地释放 KV slot。
4. 这个 req 在 P 节点的生命周期到此结束(D 节点接管后续 decode)。

---

## 六 公共收尾 hook

```python
self.log_batch_result_stats(batch, result)
self._maybe_clear_mm_inputs(batch)
self.maybe_send_health_check_signal()
```

### 6.1 `log_batch_result_stats`

按周期 log 一次 batch stats:吞吐、KV pool 使用率、cache hit rate、新增 retraction 数等。频率受 `decode_log_interval / prefill_log_interval` 控制。

### 6.2 `_maybe_clear_mm_inputs`

prefill 完成的请求,vision encoder 输出的 embedding tensor 已经写进 KV 不再需要,clear 掉以释放 GPU 内存(否则到 req finish 才 GC,会很久)。

### 6.3 `maybe_send_health_check_signal`

```python
def maybe_send_health_check_signal(self):
    if self.return_health_check_ipcs:
        self.send_to_tokenizer.send_output(
            HealthCheckOutput(
                http_worker_ipc=self.return_health_check_ipcs.popleft()
            )
        )
```

回应 `process_input_requests` 里被 busy-skip 的健康检查请求——告诉外部「我还活着,在跑请求」。每轮 process_batch_result 检查一次,逐个 pop。

---

## 七 设计要点小结

| 决策 | 原因 |
|---|---|
| forward_mode 5 路分派 | decode / prefill / dllm / disagg / idle 行为差异大 |
| `cache_finished_req` 写回 RadixCache | 让相同 prompt 的后续请求 prefix 命中 |
| 释放 KV 立刻发生(在 stream_output 之前) | 释放越早,下一轮 retract 概率越低 |
| chunked req 中间轮次不 sample | 还没 prefill 完整个 prompt,sample 没意义 |
| disagg prefill 推 KV 给 D 节点后释放 | P 节点 KV pool 周转才能高 |
| health check 在每轮收尾发 | 避免被前面的长 prefill 阻塞误判失活 |
| _maybe_clear_mm_inputs | 多模态 embedding 太大,prefill 完立即 free |

---

## 八 衔接

`stream_output_generation` 把这批输出包成 `BatchTokenIDOutput` 推给 DetokenizerManager,DetokenizerManager 检 token id → text 后再 PUSH 给 TokenizerManager 的 `handle_loop`,最后 fan-out 到各 `rid_to_state[rid]` 唤醒 `_wait_one_response` → SSE → HTTP client。
