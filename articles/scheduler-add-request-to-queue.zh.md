# `Scheduler._add_request_to_queue()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:2058-2080`

**角色：** 把已经构造好的 `Req` 按 disaggregation 模式落到对应的等待队列里。这是请求从「构造完毕」到「等待调度」的最后一跳。

---

## 一 函数实现

```python
def _add_request_to_queue(self, req: Req, is_retracted: bool = False):
    if self.disaggregation_mode == DisaggregationMode.NULL:
        if not self._set_or_validate_priority(req):
            return
        if self._abort_on_queued_limit(req):
            return
        self._prefetch_kvcache(req)
        self.waiting_queue.append(req)
        req.time_stats.set_wait_queue_entry_time()
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        self._prefetch_kvcache(req)
        self.disagg_prefill_bootstrap_queue.add(
            req, self.model_config.num_key_value_heads
        )
        req.time_stats.set_prefill_bootstrap_queue_entry_time()
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
        if not is_retracted:
            req.time_stats.set_decode_prealloc_queue_entry_time()
        else:
            req.time_stats.set_retract_time()
    else:
        raise ValueError(f"Invalid {self.disaggregation_mode=}")
```

---

## 二 三种 disaggregation 模式

`disaggregation_mode` 由 `--disaggregation-mode` server arg 决定：

| 模式 | 含义 | 落到哪 |
|---|---|---|
| `NULL` | 单机 / colocate(prefill + decode 同节点) | `self.waiting_queue` |
| `PREFILL` | PD 解耦的 prefill 节点(只跑 prefill,把 KV 传给 decode 节点) | `self.disagg_prefill_bootstrap_queue` |
| `DECODE` | PD 解耦的 decode 节点(只跑 decode,从 prefill 节点接 KV) | `self.disagg_decode_prealloc_queue` |

PD 解耦的好处：prefill 是 compute-bound,decode 是 memory-bound,放在不同节点用不同硬件配比能提高整体利用率。

---

## 三 NULL 模式(最常见)

```python
if not self._set_or_validate_priority(req):
    return
if self._abort_on_queued_limit(req):
    return
self._prefetch_kvcache(req)
self.waiting_queue.append(req)
req.time_stats.set_wait_queue_entry_time()
```

四个步骤：

### 3.1 优先级校验

`_set_or_validate_priority(req)`(`scheduler.py:2082-2105`)：

- 启用了优先级调度(`enable_priority_scheduling=True`)但用户没传 priority → 给个默认值(最低优先级)。
- **没启用**优先级调度但用户传了 priority,且配置 `abort_on_priority_when_disabled` → 直接 abort(防止 client 错以为 priority 生效)。
- 否则放行。

返回 False 表示请求被 abort,直接 return。

### 3.2 队列长度上限

`_abort_on_queued_limit(req)`(`scheduler.py:2107-2154`)：

如果 `len(waiting_queue) >= max_queued_requests`：

- **无优先级调度**：直接拒绝新请求(返回 abort 信息给 client,reason=`The request queue is full`)。
- **有优先级调度**：找队列里**优先级最低**的现存请求,如果新请求优先级**严格更高**,就 abort 那个旧请求换它的位置,否则拒绝新请求(`The request is aborted by a higher priority request`)。

返回 True 表示「这个新请求被 abort 了」,直接 return。这个机制让 server 在过载时用 priority 自动维持 SLO,不会无限堆积请求耗内存。

### 3.3 KV cache 预取

`_prefetch_kvcache(req)`：在 HiCache 启用时,主动把这个 request 的已知 prefix(根据 `routing_key` / `extra_key` 命中的) 从 L2/L3(host RAM / SSD) 提前加载到 L1(GPU HBM)。等到这个请求真的被 prefill 调度时,KV 已经在 GPU 上,prefill 直接跳过这部分 token。

预取是异步的,不会阻塞 `_add_request_to_queue`——只是发出加载请求。`get_new_batch_prefill` 在挑选请求时会通过 `check_prefetch_progress` 判断是否完成。

### 3.4 入队 + 打时间戳

```python
self.waiting_queue.append(req)
req.time_stats.set_wait_queue_entry_time()
```

`time_stats` 在多个关键节点打时间戳(进队 / 出队 / forward 开始 / 完成),最终回流给 client 做 latency profiling。

---

## 四 PREFILL 模式

```python
elif self.disaggregation_mode == DisaggregationMode.PREFILL:
    self._prefetch_kvcache(req)
    self.disagg_prefill_bootstrap_queue.add(
        req, self.model_config.num_key_value_heads
    )
    req.time_stats.set_prefill_bootstrap_queue_entry_time()
```

落到 `disagg_prefill_bootstrap_queue`——这个队列里的请求要先和对端 decode 节点完成 bootstrap 握手(交换 KV 传输地址、配对 `bootstrap_room`),才会从这里转移到正式的 prefill waiting 队列。

之所以要 bootstrap：PREFILL 节点和 DECODE 节点是独立进程(常常在不同机器),需要建立 RDMA / NIXL / Mooncake 之类的 KV 传输通道。

`num_key_value_heads` 传进去是因为 KV 传输容量按 head 数计算。

---

## 五 DECODE 模式

```python
elif self.disaggregation_mode == DisaggregationMode.DECODE:
    self.disagg_decode_prealloc_queue.add(req, is_retracted=is_retracted)
    if not is_retracted:
        req.time_stats.set_decode_prealloc_queue_entry_time()
    else:
        req.time_stats.set_retract_time()
```

落到 `disagg_decode_prealloc_queue`——decode 节点收到请求后,要在本地预分配好 KV slot 才能开始接收 prefill 节点传来的 KV。这个队列管理 prealloc 的进度。

### 5.1 `is_retracted` 参数

这是唯一一处用到 `is_retracted` 参数的地方。来源：`update_running_batch` 在 KV 不够时会主动 retract(把某些 decode 中的请求踢回队列让出 KV),那些被踢回的请求在重新入队时,`time_stats` 应该记 `retract_time`(而不是当成新请求记 `decode_prealloc_queue_entry_time`)——这样 latency profiling 才能正确反映「这个请求被 retract 了 N 次」的事实。

注意：retract 路径只在 NULL 和 DECODE 模式下出现,PREFILL 节点不需要 retract(prefill 都是一次性跑完的)。

---

## 六 设计要点小结

| 决策 | 原因 |
|---|---|
| 按 disagg 模式走不同队列 | 三种角色的请求生命周期不同,前置条件也不同 |
| NULL 模式做优先级 + 队列长度校验 | 过载时优雅降级,通过优先级维持 SLO |
| PD 模式不在此处做优先级检查 | 优先级在 prefill/decode 节点各自的队列处理器里检查 |
| `_prefetch_kvcache` 入队前发 | 越早发起预取,等到真调度时 hit 概率越高 |
| `is_retracted` 区分时间戳 | latency profiling 需要分清「初次入队」与「被踢回重排」 |
| 时间戳在每次入队都打 | time_stats 完整覆盖请求一生中所有等待节点 |

---

## 七 衔接

入队后:

- NULL → `waiting_queue` → `get_new_batch_prefill` 取出来组 prefill batch。
- PREFILL → `disagg_prefill_bootstrap_queue` → bootstrap 完成 → 转 prefill waiting queue。
- DECODE → `disagg_decode_prealloc_queue` → prealloc 完成 + KV 接收完 → 加入 decode running batch。
