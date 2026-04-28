# `Scheduler.get_next_batch_to_run()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:2302-2411`

**角色：** 主循环每轮的「调度器内核」——决定本轮 forward 跑什么。要么返回一个 prefill 批,要么返回一个 decode 批,要么返回 None(空闲)。

---

## 一 函数全貌

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    self._abort_on_waiting_timeout()
    self._abort_on_running_timeout()
    if self.dllm_config is not None:
        self.dllm_manager.filter_finished_reqs()

    # ① 把上一轮 prefill 完成的请求合并到 running_batch
    chunked_req_to_exclude = set()
    if self.dllm_config is not None and self.dllm_manager.any_staging_reqs():
        chunked_req_to_exclude.update(self.dllm_manager.staging_queue)
        for req in self.dllm_manager.staging_queue:
            self.stash_chunked_request(req)

    if self.chunked_req is not None:
        chunked_req_to_exclude.add(self.chunked_req)
        self.stash_chunked_request(self.chunked_req)

    # HiSparse 特殊处理(略)
    ...

    if (not self.enable_hisparse
        and self.last_batch
        and self.last_batch.forward_mode.is_extend()):
        if self.last_batch.chunked_req is not None:
            chunked_req_to_exclude.add(self.last_batch.chunked_req)
        if self.dllm_config is not None and self.last_batch.reqs:
            chunked_req_to_exclude.update(self.last_batch.reqs)

        last_bs = self.last_batch.batch_size()
        self.last_batch.filter_batch(
            chunked_req_to_exclude=list(chunked_req_to_exclude)
        )
        if self.last_batch.batch_size() < last_bs:
            self.running_batch.batch_is_full = False

        if not self.last_batch.is_empty():
            if self.running_batch.is_empty():
                self.running_batch = self.last_batch
            else:
                self.running_batch.merge_batch(self.last_batch)

    # ② prefill-only 批清理
    if self.running_batch.is_prefill_only:
        self.running_batch.filter_batch()
        if self.running_batch.is_empty():
            self.running_batch.batch_is_full = False

    # ③ 优先组 prefill 新批
    if self.dllm_config is not None:
        new_batch = self.get_new_batch_dllm()
    else:
        new_batch = self.get_new_batch_prefill()

    # ④ 给 spec + dp-attn 做 mlp_sync 准备
    need_mlp_sync = self.require_mlp_sync
    if need_mlp_sync and not self.spec_algorithm.is_none():
        new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
        need_mlp_sync = new_batch is None

    # ⑤ 选 prefill 还是 decode
    if new_batch is not None:
        ret = new_batch
    else:
        if (
            not self.running_batch.is_empty()
            and not self.running_batch.is_prefill_only
        ):
            self.running_batch = self.update_running_batch(self.running_batch)
            ret = self.running_batch if not self.running_batch.is_empty() else None
        else:
            ret = None

    ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)
    ret = self._maybe_prepare_ngram_embedding(ret)

    if ret:
        set_schedule_time_batch(ret)

    return ret
```

整体五步:**timeout 检查 → 合并上一批 → 优先 prefill → 否则 decode → DP/spec 同步处理**。

---

## 二 ① 超时治理

```python
self._abort_on_waiting_timeout()
self._abort_on_running_timeout()
```

每轮调度前先扫两个队列：

- **waiting_timeout**：`waiting_queue` 里有没有等了太久(超过 `SGLANG_REQ_WAITING_TIMEOUT`)的请求,直接 abort。防止过载时旧请求无限挂在队列里(client 早已超时)。
- **running_timeout**：`running_batch` 里有没有 forward 跑了太久(超过 `SGLANG_REQ_RUNNING_TIMEOUT`)的请求,标记 abort。

放在 `get_next_batch_to_run` 开头是个好位置：每轮都会过一次,频次足够;同时早于 batch 组装,abort 的请求不会被打包进新批徒劳跑一次。

---

## 三 ② 合并 last_batch 到 running_batch

这是 continuous batching 的关键缝合点。

### 3.1 chunked_req 排除集

```python
chunked_req_to_exclude = set()

if self.chunked_req is not None:
    chunked_req_to_exclude.add(self.chunked_req)
    self.stash_chunked_request(self.chunked_req)
```

`chunked_req` 是「当前正在被 chunked prefill 拆分的那一个 request」。它在上一轮被部分 prefill 了,但还没跑完——这一轮要把它**单独拎出来**,不参与 merge,因为它要继续走 prefill 路径(下一轮的 `_get_new_batch_prefill_raw` 会用 `adder.add_chunked_req` 把它接回去)。

`stash_chunked_request` 把它从 `last_batch.reqs` 中暂存到 scheduler 的 `chunked_req` 字段(其实就是个 noop,因为已经在那个字段上了),后面 `filter_batch` 把它从 `last_batch` 物理移除。

### 3.2 last_batch 是 EXTEND(prefill) 时合并

```python
if (not self.enable_hisparse
    and self.last_batch
    and self.last_batch.forward_mode.is_extend()):
    ...
    last_bs = self.last_batch.batch_size()
    self.last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))
    if self.last_batch.batch_size() < last_bs:
        self.running_batch.batch_is_full = False

    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch
        else:
            self.running_batch.merge_batch(self.last_batch)
```

含义：上一轮跑的是 prefill,那些请求 prefill 完成,这一轮要进入 decode 阶段——把它们 merge 到 `running_batch`。

- `filter_batch` 把 chunked_req 过滤出去(已经在排除集)。
- 如果 last_batch 缩小了(说明 chunked_req 被拿走了),`running_batch.batch_is_full` 重置——给后续 prefill 留点空间。
- 然后 merge 到 running_batch(空就直接赋值,非空就 `merge_batch` 拼起来)。

为什么 last_batch 是 DECODE 就不 merge？因为 DECODE 的 batch 已经在 `running_batch` 里跑了,本来就是同一个对象。EXTEND 是「新 prefill 出来的请求第一次入 running」的边界,只在这个时刻 merge。

### 3.3 prefill-only 清理

```python
if self.running_batch.is_prefill_only:
    self.running_batch.filter_batch()
    if self.running_batch.is_empty():
        self.running_batch.batch_is_full = False
```

prefill-only 模式(embedding / classifier 等不需要 decode 的任务):请求 prefill 完就结束,不进入 decode。这里把 finished 请求过滤掉,让 `/v1/loads` 等 endpoint 看到的 `num_running_reqs` 准确。

放在 `last_batch` 块外面是为了即使流量停了(`last_batch is None`)也能清理掉残留的 finished 请求。

---

## 四 ③ 优先组 prefill 新批

```python
if self.dllm_config is not None:
    new_batch = self.get_new_batch_dllm()
else:
    new_batch = self.get_new_batch_prefill()
```

调 `get_new_batch_prefill()`(详见单独文章) ——它内部用 `PrefillAdder` 从 `waiting_queue` 一个个挑请求,直到 KV 池或 token 上限拦不下来为止。

返回 `None` 表示没有新 prefill 可组(队列空 / 池满 / 都不能放下来)。

dllm_config 是 diffusion-LLM(扩散语言模型) 的特殊路径,普通 LLM 走 `get_new_batch_prefill`。

---

## 五 ④ DP attention + spec decode 的 mlp_sync

```python
need_mlp_sync = self.require_mlp_sync
if need_mlp_sync and not self.spec_algorithm.is_none():
    new_batch = self.maybe_prepare_mlp_sync_batch(new_batch)
    need_mlp_sync = new_batch is None
```

`require_mlp_sync = enable_dp_attention`——DP attention 把 attn 部分各 DP rank 独立跑,但 MLP 部分要做一次 all-gather 同步。这要求**所有 DP rank 在同一时刻一致地跑 prefill 或一致地跑 decode**(不能 rank0 跑 prefill 而 rank1 跑 decode,会死锁在 all-gather 上)。

`maybe_prepare_mlp_sync_batch` 做的事:统一各 rank 的「本轮跑啥」决策。返回 None 表示「prefill 没准备好,等下面 decode 路径」。

注释说:这个分支只对 spec decode 生效——不带 spec 的话,后面有另一处 `maybe_prepare_mlp_sync_batch` 兜底。

---

## 六 ⑤ 选 prefill 还是 decode

```python
if new_batch is not None:
    ret = new_batch                              # prefill 优先
else:
    if (not self.running_batch.is_empty()
        and not self.running_batch.is_prefill_only):
        self.running_batch = self.update_running_batch(self.running_batch)
        ret = self.running_batch if not self.running_batch.is_empty() else None
    else:
        ret = None                               # 没 prefill 也没 decode
```

策略：**prefill 优先**。这是降低 TTFT(time to first token) 的核心决策——新请求一旦能组进 batch,立刻 prefill,不让它在队列里多等一轮 decode。

代价：已经在 decode 的请求 ITL(inter-token latency) 偶尔会被新 prefill 拖慢。chunked prefill 就是为了限制这个代价——超长 prompt 拆 chunk,单次 prefill 占用的 forward 时间有上限。

`update_running_batch`(见单独文章) 处理 decode 路径：filter finished、retract on OOM、prepare_for_decode。

---

## 七 ⑥ 后处理 hooks

```python
ret = self.maybe_prepare_mlp_sync_batch(ret, need_sync=need_mlp_sync)
ret = self._maybe_prepare_ngram_embedding(ret)

if ret:
    set_schedule_time_batch(ret)
```

- `maybe_prepare_mlp_sync_batch(ret, need_sync=...)`：DP attention 的 idle batch 兜底——如果某些 rank 没请求要跑(`ret is None`),要给它构造一个 idle batch 让它陪跑同步通信。
- `_maybe_prepare_ngram_embedding`：n-gram speculative decoding 的预备步骤(给 batch 注入 n-gram 候选)。
- `set_schedule_time_batch`：给本批每个 req 打 `schedule_time` 时间戳。

---

## 八 设计要点小结

| 决策 | 原因 |
|---|---|
| timeout 检查放在调度开头 | 频次足够,且能避免 abort 请求被打包进新批 |
| chunked_req 单独 stash | 它要继续走 prefill 路径,不能进 decode merge |
| last_batch is_extend 才 merge | EXTEND→DECODE 是请求生命周期的关键边界 |
| prefill 优先 decode | 降低 TTFT,代价由 chunked prefill 限制 |
| DP attention 强制 mlp_sync | 各 DP rank 必须一致地跑同种 forward,否则 all-gather 死锁 |
| idle batch 兜底 | DP rank 没请求时也要陪跑通信原语 |

---

## 九 衔接

返回的 `batch`:

- 不是 None → 主循环调 `run_batch(batch)` 跑 forward,然后 `process_batch_result(batch, result)` 处理输出。
- 是 None → 主循环调 `on_idle()`(空闲自检 + KV pool 状态重置)。

无论哪种,本轮结尾都把 `self.last_batch = batch`,下一轮 `get_next_batch_to_run` 就能从 `last_batch` 接着合并。
