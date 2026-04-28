# `Scheduler.update_running_batch()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:2669-2756`

**角色：** decode 路径的入口准备——给 `running_batch` 做体检：清掉 finished 请求、KV 池满时主动 retract、最后 prepare decode tensor。在 `get_next_batch_to_run` 里只有「无新 prefill 可组」时才会调到它。

---

## 一 函数全貌

```python
def update_running_batch(self, batch: ScheduleBatch) -> Optional[ScheduleBatch]:
    """Update the current running decoding batch."""
    initial_bs = batch.batch_size()

    batch.filter_batch(v1_spec_info_filtered=True)
    if batch.is_empty():
        batch.batch_is_full = False
        return batch

    if self.enable_hierarchical_cache:
        self.tree_cache.flush_write_through_acks()

    # Check if decode out of memory
    if (kv_full_retract_flag := not batch.check_decode_mem()) or (
        TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
    ):
        ...  # retract 路径
    else:
        self.new_token_ratio = max(
            self.new_token_ratio - self.new_token_ratio_decay,
            self.min_new_token_ratio,
        )

    if batch.batch_size() < initial_bs:
        batch.batch_is_full = False

    if batch.is_empty():
        return batch

    batch.prepare_for_decode()
    return batch
```

---

## 二 ① filter_batch:清掉 finished

```python
initial_bs = batch.batch_size()
batch.filter_batch(v1_spec_info_filtered=True)
if batch.is_empty():
    batch.batch_is_full = False
    return batch
```

`filter_batch` 把 `req.finished()` 为 True 的请求物理移除——它们已经在上一轮 `process_batch_result` 里被打了 `finished_reason`,KV 也释放完了,这一轮不再参与。

`v1_spec_info_filtered=True`：spec decode v1 的 spec info 在外面已经过滤过,这里跳过重复处理。

如果全 finish 了直接返回(后面的 retract / prepare_for_decode 都没意义)。

---

## 三 ② HiCache write-through 回收

```python
if self.enable_hierarchical_cache:
    self.tree_cache.flush_write_through_acks()
```

HiCache 把 KV 写到 host RAM 时,GPU 端的拷贝句柄会被锁住(`lock_ref`) 防止 evict。一旦写入完成,这些 node 应该立刻变成 evictable——`flush_write_through_acks` 检查所有 ack,把完成的 node 释放出来。

放在 retract 之前:让 retract 有更多 evictable 选择。

---

## 四 ③ KV 池不足时 retract

```python
if (kv_full_retract_flag := not batch.check_decode_mem()) or (
    TEST_RETRACT and self.forward_ct % TEST_RETRACT_INTERVAL == 0
):
    old_available_tokens = self.token_to_kv_pool_allocator.available_size()
    old_ratio = self.new_token_ratio
    mamba_pool = getattr(self.tree_cache.req_to_token_pool, "mamba_pool", None)
    old_mamba_available = (
        mamba_pool.available_size() if mamba_pool is not None else None
    )
    retracted_reqs, new_token_ratio, reqs_to_abort = batch.retract_decode(
        self.server_args
    )
    ...
```

### 4.1 触发条件

- `check_decode_mem()`：检查本轮 decode 需要新分配 N 个 token(每个 running req 1 个,spec decode 多个) 的 KV slot,池里有没有这么多空间。返回 False(池不够) → 进入 retract 路径。
- `TEST_RETRACT`:测试钩子,周期性强制触发 retract,验证回退路径正确。

### 4.2 retract_decode

`batch.retract_decode(server_args)` 是核心:从 batch 里挑一些请求**踢出去回到 waiting_queue**,把它们的 KV slot 让给剩下的请求。返回三个东西：

- `retracted_reqs`：被踢回 waiting 的请求(下次重新调度,会从命中的 prefix 处续上)。
- `new_token_ratio`：动态调整的「预测每请求新生成 token 数」,影响 `check_decode_mem` 的判定阈值。
- `reqs_to_abort`：彻底放不下的请求(prompt + 已生成已经超过 max_req_len 等),直接 abort。

retract 选择策略(具体在 retract_decode 内部)考虑:

- 已生成 token 多的请求(踢回去重做的代价大,优先保留)? 或者反过来?——实际策略和 KV 占用、time_stats 综合算。
- 优先级模式下保留高优。

### 4.3 metrics + 重新入队

```python
self.num_retracted_reqs = len(retracted_reqs)
if self.enable_metrics and len(retracted_reqs) > 0:
    self.metrics_collector.increment_retracted_reqs(
        num_retracted_reqs=len(retracted_reqs),
        num_retracted_input_tokens=sum(len(r.origin_input_ids) for r in retracted_reqs),
        num_retracted_output_tokens=sum(len(r.output_ids) for r in retracted_reqs),
    )
self.new_token_ratio = new_token_ratio

for req in reqs_to_abort:
    abort_reason: FINISH_ABORT = req.to_finish
    self.send_to_tokenizer.send_output(
        AbortReq(finished_reason=abort_reason.to_json(), rid=req.rid),
        req,
    )
...
for req in retracted_reqs:
    self._add_request_to_queue(req, is_retracted=True)
```

- abort 的请求立刻通知 client。
- retracted 的请求**重新入队**(`is_retracted=True` 让 time_stats 记 `retract_time` 而不是初次入队时间)。它们会保留 `output_ids` 已生成的部分,下次被调度时,RadixCache 会命中 prompt + 已生成 prefix,重新 prefill 跳过这部分,然后接着 decode——**对 client 透明**,只是 latency 变高。

### 4.4 日志

```python
msg_prefix = (
    "KV cache pool is full. Retract requests. "
    if kv_full_retract_flag
    else "Testing retraction. "
)
msg_details = f"#retracted_reqs: {len(retracted_reqs)}, #new_tokens_gained: {new_token_gained}"
if mamba_num_gained is not None:
    msg_details += f", #mamba_num_gained: {mamba_num_gained}"
if kv_full_retract_flag:
    msg_details += f", #new_token_ratio: {old_ratio:.4f} -> {new_token_ratio:.4f}"
logger.warning(msg_prefix + msg_details)
```

KV 不足触发的 retract 是 warning 级别——这个事件本身不是 bug,但频繁发生意味着配置(KV 池太小 / max_running_requests 太大) 需要调整。

### 4.5 没 retract 时降低 new_token_ratio

```python
else:
    self.new_token_ratio = max(
        self.new_token_ratio - self.new_token_ratio_decay,
        self.min_new_token_ratio,
    )
```

`new_token_ratio` 是预测「每个 running 请求平均还要生成多少 token」的乘数,用来给 `check_decode_mem` 留缓冲。如果当前轮没 retract(说明预算够),让 ratio 慢慢衰减(更乐观),允许下一轮塞更多 prefill。retract 时反之上调。这是个**自适应反馈控制**,在抖动稳定性和吞吐之间动态找平衡。

---

## 五 ④ batch_is_full 重置 + prepare_for_decode

```python
if batch.batch_size() < initial_bs:
    batch.batch_is_full = False

if batch.is_empty():
    return batch

batch.prepare_for_decode()
return batch
```

- batch 缩小(filter 或 retract 引起) → `batch_is_full = False`,允许下一轮 `get_new_batch_prefill` 再塞新请求。
- 全空 → 直接返回 None-equivalent 的 empty batch。
- 否则 `prepare_for_decode`：构造 decode 用的 forward tensor:
  - `input_ids`:每个请求上一步生成的 token id(decode 一次只跑一个 token per req)。
  - `positions`:每个请求当前位置(seq_lens 累加)。
  - `seq_lens += 1`。
  - 准备 attn backend 的 metadata(KV cache 索引)。

注意 prefill 走的是 `prepare_for_extend`(在 `get_new_batch_prefill` 里),decode 走 `prepare_for_decode`——因为 forward_mode 不同,需要的 metadata 不同。

---

## 六 设计要点小结

| 决策 | 原因 |
|---|---|
| filter_batch 在 retract 之前 | 先把 finished 的释放,可能 retract 就不用做了 |
| HiCache write-through ack 回收前置 | 让 retract 有更多 evictable 选择 |
| retract 时 abort 真正放不下的 | 防止某些超长请求无限循环 retract→reschedule |
| new_token_ratio 自适应 | 根据是否 retract 动态调缓冲,平衡吞吐和稳定性 |
| retracted 请求重新入队对 client 透明 | RadixCache 记住已生成,后续被调度时 prefix 命中 |
| batch_is_full 在缩小时重置 | 允许新 prefill 立刻填补空缺,提高利用率 |

---

## 七 衔接

返回的 batch:

- empty → 上层 `get_next_batch_to_run` 把 `ret` 设成 None,本轮 idle。
- non-empty → 作为本轮 forward 的 decode batch,进 `run_batch`。
