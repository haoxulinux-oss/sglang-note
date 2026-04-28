# `Scheduler.run_batch()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:2767-2923`

**角色：** 把组好的 `ScheduleBatch` 真正喂给 model worker,触发一次 forward + sample,返回 `GenerationBatchResult`(或 embedding 任务的 `EmbeddingBatchResult`)。是 scheduler 主循环里**唯一会真正在 GPU 上跑计算**的地方。

---

## 一 函数签名

```python
def run_batch(
    self,
    batch: ScheduleBatch,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> Union[GenerationBatchResult, EmbeddingBatchResult]:
```

`pp_proxy_tensors` 只在 PP > 1 时由 `event_loop_pp` 传入(承载上一 stage 的 hidden states)。普通 path 是 None。

---

## 二 前置 hook

```python
self.forward_ct += 1
self._profile_batch_predicate(batch)
if self.forward_sleep_time is not None:
    logger.info(f"Scheduler.run_batch sleep {self.forward_sleep_time}s")
    time.sleep(self.forward_sleep_time)

if batch.forward_mode == ForwardMode.EXTEND:
    set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")

if batch.forward_mode.is_prebuilt():
    return self._run_batch_prebuilt(batch)
```

- `forward_ct` 计数,用于 retract 测试钩子等。
- `_profile_batch_predicate`:torch profiler hook,启用时按规则开/关 profile。
- `forward_sleep_time`:debug/测试用,人为减慢 forward。
- `EXTEND`(prefill) 的 `set_prefill_run_batch_start_time` 时间戳。
- `is_prebuilt()`:PD-disagg decode 节点的特殊路径——KV 已经从 prefill 节点传过来了,本轮不需要重新跑 forward,直接走占位流程。

---

## 三 三个 forward 路径

```python
if self.is_generation:
    if self.spec_algorithm.is_none() or self.enable_overlap:
        worker_batch_or_batch = batch.get_model_worker_batch()
    else:
        worker_batch_or_batch = batch  # spec v1 非 overlap 直接传 batch

    if self.enable_overlap:
        # Path A:overlap schedule
        ...
    elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
        # Path B:PD multiplexing 的 split prefill
        batch_result = self.tp_worker.forward_batch_split_prefill(batch)
        future_indices_or_next_token_ids = batch_result.next_token_ids
    else:
        # Path C:普通 forward(默认 GPU 路径之一)
        kwargs = ({"pp_proxy_tensors": pp_proxy_tensors}
                  if self.spec_algorithm.is_none() else {})
        with self.record_forward_metrics(batch):
            batch_result = self.model_worker.forward_batch_generation(
                worker_batch_or_batch, **kwargs
            )
        future_indices_or_next_token_ids = batch_result.next_token_ids
        self.update_cache_from_scheduler(batch, batch_result)
```

### 3.1 `get_model_worker_batch()`

把 ScheduleBatch 投影成 `ModelWorkerBatch`——一个纯数据 dataclass,只含 forward 必需的字段:

- `input_ids` / `positions` / `seq_lens`
- `req_pool_indices`(每个请求在 req_to_token_pool 里的 slot)
- attn backend 的 metadata(KV cache 索引、page table 等)
- `sampling_info`(temperature / top_k / top_p / 各种 penalty / structured constraint mask)
- `return_logprob` / `return_hidden_states` 标志
- `lora_id` 数组(LoRA batch 用)
- spec decode 信息(EagleDraftInput 等)

为什么投影:scheduler 内部对象有大量调度专用字段(time_stats、tree_cache_node、req 状态机) 不需要传给 worker;同时把 batch 对象拍平成 dataclass 可以跨进程 pickle / 跨 stream 复制更快。

注：spec v1 非 overlap 那一支直接传 batch(历史原因,TODO 中标注要统一)。

### 3.2 Path C:普通 forward(最常见)

```python
batch_result = self.model_worker.forward_batch_generation(worker_batch, **kwargs)
future_indices_or_next_token_ids = batch_result.next_token_ids
self.update_cache_from_scheduler(batch, batch_result)
```

- `model_worker.forward_batch_generation` 进入 `tp_worker → model_runner → model.forward → sampler`,拿到 `next_token_ids` 和 `logits_output`。
- `update_cache_from_scheduler`:把刚生成的 token KV 写回 RadixCache(让后续相同 prefix 的请求能命中)。

### 3.3 Path A:overlap schedule

```python
model_worker_batch = worker_batch_or_batch
self.record_batch_in_overlap(model_worker_batch)

# Sampling info will be modified during forward, so we store a copy.
model_worker_batch.sampling_info = (
    model_worker_batch.sampling_info.copy_for_forward()
)

bs = len(model_worker_batch.seq_lens)
future_indices = self.future_map.alloc_future_indices(bs)

with self.forward_stream_ctx, self.record_bubble_metrics(batch):
    self.forward_stream.wait_stream(self.schedule_stream)
    self.future_map.resolve_future(model_worker_batch)
    with self.record_forward_metrics(batch):
        batch_result = self.model_worker.forward_batch_generation(
            model_worker_batch
        )
    batch_result.copy_done = self.device_module.Event()
    if batch_result.delay_sample_func is None:
        self.future_map.store_to_map(future_indices, batch_result)
        batch_result.copy_to_cpu(return_logprob=batch.return_logprob)
    else:
        batch_result.future_indices = future_indices

future_indices_or_next_token_ids = -future_indices.indices
```

这是 SGLang 高吞吐的关键。要点：

- **独立 CUDA stream**:`forward_stream` 与 `schedule_stream` 分开。schedule_stream 跑 batch 组装、KV 元信息计算等 CPU/GPU 混合任务;forward_stream 跑 model.forward。两条 stream 可以**真正并发**(只在拷贝边界用 `wait_stream` 同步)。
- **future indices**:scheduler 主线程不等 forward 完成就开始组下一批。但下一批的 `input_ids` 来自这一批的 `next_token_ids`——此刻还没算出。解决:用 `future_indices`(负数索引) 作占位,组下一批时把 `input_ids = -future_indices.indices` 塞进去,真正用到时再 `resolve_future` 替换成实际 token。
- **`record_batch_in_overlap`**:把 `model_worker_batch` 留个引用在 `batch_record_buf[ct]`(双缓冲),防止 PyTorch GC 在 forward 还没完成时把 GPU tensor 释放掉。
- **sampling_info copy**:forward 过程中会原地修改 sampling_info(如 penalty 累计),copy 一份避免下一轮调度看到脏值。
- **`copy_to_cpu`**:把 `next_token_ids` 异步从 GPU 拷到 CPU,完成事件存在 `copy_done`,下游需要时 wait。
- **delay_sample_func**:spec v2 / 某些采样路径下,采样要在下一 forward 之后才完成(例如要看下一批的 grammar mask),这里只 alloc future,真正 sample 在 `launch_batch_sample_if_needed` 里做。

### 3.4 spec v2 special

```python
if batch.is_spec_v2:
    batch.spec_info = batch_result.next_draft_input
    batch.spec_info.future_indices = future_indices
    batch.seq_lens = batch_result.next_draft_input.new_seq_lens
```

spec v2 的下一轮 draft input 由 forward 产出,直接写回 batch 给下次用。

### 3.5 Path B:PD multiplexing split prefill

```python
elif self.enable_pdmux and batch.forward_mode.is_split_prefill():
    batch_result = self.tp_worker.forward_batch_split_prefill(batch)
    future_indices_or_next_token_ids = batch_result.next_token_ids
```

`pdmux` 模式让一台 GPU 上同时有 prefill 和 decode 实例,通过 stream multiplexing 切片跑。`split_prefill` 是把超长 prefill 切片让出 GPU 给 decode 用的特殊路径。

---

## 四 写回 batch 状态

```python
batch.output_ids = future_indices_or_next_token_ids

if batch.return_logprob:
    batch_result.extend_input_len_per_req = [
        req.extend_input_len for req in batch.reqs
    ]
    batch_result.extend_logprob_start_len_per_req = [
        req.extend_logprob_start_len for req in batch.reqs
    ]
else:
    batch_result.extend_input_len_per_req = None
    batch_result.extend_logprob_start_len_per_req = None
```

`batch.output_ids` 写成 future_indices(overlap) 或实际 token id(同步)——下一轮 `prepare_for_decode` 用它构造下一批的 input_ids。

`extend_input_len_per_req / extend_logprob_start_len_per_req` 记到 batch_result:**这两个字段会被后续 overlap schedule 修改**,但 output processing 需要 forward 时刻的值,所以快照一份。

---

## 五 Embedding 路径

```python
else:  # embedding or reward model
    model_worker_batch = batch.get_model_worker_batch()
    if self.enable_overlap:
        ...
        pooler_output = self.tp_worker.forward_batch_embedding(model_worker_batch)
        ret = EmbeddingBatchResult(
            embeddings=pooler_output.embeddings,
            pooled_hidden_states=pooler_output.pooled_hidden_states,
        )
        ret.copy_to_cpu()
    else:
        pooler_output = self.tp_worker.forward_batch_embedding(model_worker_batch)
        ret = EmbeddingBatchResult(...)
```

embedding 任务跑的是 encoder-style forward + pooling,没有 `next_token_ids` 概念,直接产出 `embeddings`。

---

## 六 后置 hook

```python
if batch.forward_mode == ForwardMode.EXTEND:
    set_time_batch(batch.reqs, "set_prefill_run_batch_end_time")

if (self.server_args.enable_dp_attention
    and self.server_args.elastic_ep_backend is not None):
    tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
    tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
    tp_active_ranks &= tp_active_ranks_cpu
    dp_active_ranks = tp_active_ranks.reshape(self.dp_size, -1).prod(axis=1)
    self.send_to_tokenizer.send_output(
        ActiveRanksOutput(status=dp_active_ranks.tolist())
    )

return ret
```

- prefill 结束时间戳。
- elastic EP backend(MoE 专家弹性扩缩):每轮报告一次哪些 DP rank 活跃,让 controller 决定是否调整 expert 分布。

---

## 七 设计要点小结

| 决策 | 原因 |
|---|---|
| ScheduleBatch → ModelWorkerBatch 投影 | 隔离调度状态与 forward 输入,跨 stream / 跨进程更高效 |
| overlap schedule 双 stream + future indices | CPU 调度与 GPU forward 完全并行,核心吞吐增益来源 |
| copy_to_cpu 异步 + copy_done 事件 | 避免主线程 sync 等 GPU,output processor 需要时再 wait |
| sampling_info copy_for_forward | forward 会原地 mutate(penalty 累计等),不能让调度看到脏值 |
| record_batch_in_overlap 双缓冲引用 | 防 GC 在 forward 未完成时释放 GPU tensor |
| split_prefill / pdmux | 同 GPU prefill+decode 复用,降低长 prompt 对 ITL 的冲击 |
| spec v2 写回 next_draft_input 到 batch | 下一轮 draft 用,避免重新构造 |

---

## 八 衔接

返回的 `batch_result` 立刻被 `process_batch_result(batch, result)` 处理:

- decode → `process_batch_result_decode` → 拼 token、检查 finish、`stream_output`。
- prefill → `process_batch_result_prefill` → 同上,加上 prefill→decode 状态切换。
- 其他 forward_mode 各有专门 handler。
