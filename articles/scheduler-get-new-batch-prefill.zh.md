# `Scheduler.get_new_batch_prefill()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:2419-2667`

**角色：** 从 `waiting_queue` 里挑请求组成一个 prefill `ScheduleBatch`。这是 SGLang continuous batching + RadixCache + chunked prefill 的核心实现位置。

---

## 一 顶层封装

```python
def get_new_batch_prefill(self) -> Optional[ScheduleBatch]:
    prefill_delayer_single_pass = None
    if self.prefill_delayer:
        max_pool_usage = self.get_pool_stats().get_max_pool_usage()
        prefill_delayer_single_pass = PrefillDelayerSinglePassExecutor(
            self.prefill_delayer, token_usage=max_pool_usage
        )

    ret = self._get_new_batch_prefill_raw(
        prefill_delayer_single_pass=prefill_delayer_single_pass
    )

    if self.prefill_delayer:
        prefill_delayer_single_pass.finalize(actual_prefill=ret is not None)

    return ret
```

`prefill_delayer` 是个可选机制：**在 KV 池接近满时主动拒绝新 prefill**——让 decode 把 KV 释放出来再说,避免「prefill 一进来就触发 retract」的抖动。`SinglePassExecutor` 是单次决策的有状态对象,根据当前池占用打分,决定是否本轮跳过 prefill。

主体逻辑在 `_get_new_batch_prefill_raw`。

---

## 二 提早返回的几个守门条件

```python
def _get_new_batch_prefill_raw(self, prefill_delayer_single_pass):
    # ① grammar 编译完成的请求重新入队
    if self.grammar_manager.has_waiting_grammars():
        ready_grammar_requests = self.grammar_manager.get_ready_grammar_requests()
        for req in ready_grammar_requests:
            self._add_request_to_queue(req)

    # ② HiCache 事件检查
    if self.enable_hierarchical_cache:
        self.tree_cache.check_hicache_events()

    # ③ 优先级抢占重置 batch_is_full
    if self.enable_priority_preemption:
        self.running_batch.batch_is_full = False

    # ④ 队列空 & 没 chunked_req → 直接退
    if (self.running_batch.batch_is_full or len(self.waiting_queue) == 0) \
       and self.chunked_req is None:
        return None

    running_bs = len(self.running_batch.reqs)
    if (self.get_num_allocatable_reqs(running_bs) <= 0
        and self.chunked_req is None
        and not self.enable_priority_preemption):
        self.running_batch.batch_is_full = True
        return None
```

- ① **grammar ready 入队**：从 grammar 编译队列里把已编译完的请求拿出来扔进 waiting_queue,本轮就有机会上车。
- ② **HiCache 事件**：处理 KV 写回 host RAM / SSD 的回调,腾 evictable 出来。
- ③ **优先级抢占**：如果允许抢占,重置 `batch_is_full` 让 PrefillAdder 有机会用高优请求踢掉低优的。
- ④ **真没东西可跑**:空。

`get_num_allocatable_reqs(running_bs) = pp_max_micro_batch_size - running_bs`(PP > 1 时还要乘 req_to_token_pool 容量)——本轮还能塞多少个新请求。<= 0 就 `batch_is_full = True` 退出。

---

## 三 优先级排序 + 动态 chunk size

```python
self.policy.calc_priority(self.waiting_queue, self.running_batch)

if TEST_RETRACT and running_bs > TEST_RETRACT_NO_PREFILL_BS:
    return None

chunked_prefill_size = self.chunked_prefill_size
if self.chunked_req is not None and self.enable_dynamic_chunking:
    history_len = len(self.chunked_req.prefix_indices)
    dynamic_size = self.predict_next_chunk_size(history_len)
    if dynamic_size is not None:
        chunked_prefill_size = dynamic_size
```

- `policy.calc_priority` 按 schedule policy(`fcfs` / `lpm` / `random` / 优先级 …) 给 waiting_queue 排序。常见配置 `lpm`(longest prefix match) 把 prefix 命中长的请求排前面——它们 prefill 实际计算少。
- **TEST_RETRACT**：测试钩子,人为让 running batch > 阈值时不 prefill,触发 retract 路径。
- **dynamic chunking**：chunked prefill 的 chunk size 默认是固定值(`chunked_prefill_size`),但启用动态 chunking 时,会根据 chunked_req 已 prefill 的长度预测下一个 chunk 大小——通常越往后 chunk 可以越大(因为 attn 计算复杂度对 prefix 增长不敏感)。

---

## 四 PrefillAdder：核心打包器

```python
adder = PrefillAdder(
    self.page_size,
    self.tree_cache,
    self.token_to_kv_pool_allocator,
    self.running_batch,
    self.new_token_ratio,
    self.max_prefill_tokens,
    chunked_prefill_size,
    running_bs if self.is_mixed_chunk else 0,
    self.priority_scheduling_preemption_threshold,
    max_prefill_bs=self.max_prefill_bs,
    max_running_requests=self.max_running_requests,
    prefill_max_requests=self.server_args.prefill_max_requests,
    prefill_delayer_single_pass=prefill_delayer_single_pass,
    dllm_config=self.dllm_config,
)

if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)
```

`PrefillAdder` 是「在 KV 池预算内尽量塞请求」的状态机,它知道：

- 每加一个请求,这个请求能命中 RadixCache 多少 prefix(这部分不消耗新 KV)。
- 还需要新分配多少 KV slot(`page_size` 取整后)。
- 当前 KV 池剩余多少。
- 总 prefill token 数有没有超 `chunked_prefill_size`(决定是否要 chunk)。

`add_chunked_req(self.chunked_req)`：上一轮被拆掉的 chunked 请求要先回到这一批,继续 prefill 它剩下的 chunk(用最高优先级)。

---

## 五 主循环：从 waiting_queue 一个个尝试加入

```python
for req in self.waiting_queue:
    # 5.1 LoRA batch 校验
    if self.enable_lora and req.lora_id not in running_loras:
        if self.enable_lora_overlap_loading:
            res = self.lora_overlap_loader.try_overlap_load_lora(
                req.lora_id, running_loras
            )
            if not res:
                continue
        else:
            new_lora_set = {req.lora_id} | running_loras
            if not self.tp_worker.model_runner.lora_manager.validate_lora_batch(
                new_lora_set
            ):
                continue

    # 5.2 batch 满判定
    running_bs = len(self.running_batch.reqs)
    if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
        self.running_batch.batch_is_full = True
    if self.disaggregation_mode == DisaggregationMode.PREFILL:
        if len(adder.can_run_list) >= self.req_to_token_pool.available_size():
            self.running_batch.batch_is_full = True

    if self.running_batch.batch_is_full:
        if (not self.enable_priority_preemption
            or not adder.preempt_to_schedule(req, self.server_args)):
            break

    # 5.3 HiCache 预取检查
    if self.enable_hicache_storage:
        prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
        if not prefetch_done:
            continue
        req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)

    # 5.4 真正尝试加入
    req.init_next_round_input(self.tree_cache)
    res = adder.add_one_req(
        req,
        has_chunked_req=(self.chunked_req is not None),
        truncation_align_size=self.truncation_align_size,
    )

    if self.enable_lora:
        running_loras.add(req.lora_id)

    if res != AddReqResult.CONTINUE:
        if res == AddReqResult.NO_TOKEN:
            ...
            self.running_batch.batch_is_full = True
        # mamba state 回滚(避免泄漏)
        added = len(adder.can_run_list) > 0 and req is adder.can_run_list[-1]
        if not added and req.mamba_pool_idx is not None:
            self.tree_cache.req_to_token_pool.mamba_pool.free(
                req.mamba_pool_idx.unsqueeze(-1)
            )
            req.mamba_pool_idx = None
        break
```

逐一拆解：

### 5.1 LoRA 同批校验

LoRA 模型把多个 adapter 的请求合到一个 batch 跑(BGMV/SGMV 算子),但同一 batch 内的 adapter 总数有上限。如果当前请求的 adapter 没在 running 集合,要先校验加入后是否仍合法,合法才允许。

`enable_lora_overlap_loading`：异步加载 LoRA weight 的优化——一次加载一个 adapter,与计算重叠(否则要等所有 adapter 加载完才能开始 forward)。

### 5.2 batch 满判定

两层 cap：
- `get_num_allocatable_reqs`:`pp_max_micro_batch_size - running_bs`,主要受 PP 微 batch 大小限制。
- DisaggMode.PREFILL：还要看 `req_to_token_pool` 是否有空 slot(prefill 节点的 KV 一旦被占用,要等 KV 传给 decode 节点才能释放,容易紧)。

满了之后,默认 break 退出。但启用 priority preemption 时,会调 `adder.preempt_to_schedule`——尝试用当前高优请求踢掉一个低优的(从 running 或 adder 已加入的列表里抢)。

### 5.3 HiCache 预取检查

`enable_hicache_storage`:KV 写到磁盘做 L3 cache。`_add_request_to_queue` 时已经发起异步预取,这里检查进度：

- **没完成 → continue**：暂时跳过这个请求,先组其他能跑的。
- **完成 → 拿到 storage_hit_length**：表示从磁盘加载了 N 个 token 的 KV 进 GPU,这些 token 不需要重新 prefill。

### 5.4 真正尝试加入

`req.init_next_round_input(tree_cache)` 算这个请求本轮要 prefill 的 token 范围:

- `prefix_indices`:RadixCache 命中的 prefix(这部分 KV 已存在,跳过)。
- `extend_input_len`:这一轮要 prefill 的 token 数。
- 如果是 chunked prefill 的中间轮次,只 prefill 一个 chunk。

`adder.add_one_req` 返回三种结果之一:

| `AddReqResult` | 含义 | 行为 |
|---|---|---|
| `CONTINUE` | 加成功,可以继续试下一个 | 循环继续 |
| `NO_TOKEN` | KV 池剩余不够 | 标 batch_is_full + break |
| 其他(`OTHER`) | 其他限制(prefill_max_requests / max_prefill_tokens / chunked_prefill_size 满) | break |

**Mamba state 回滚**：Mamba 模型每个请求要预分配一个 mamba state slot(在 `_add_request_to_queue` 之前的某处)。如果这个请求最终没被 add(`adder.can_run_list[-1]` 不是它),mamba slot 必须 free,否则会泄漏。

---

## 六 收尾:更新队列状态与构造 batch

```python
can_run_list: List[Req] = adder.can_run_list
if len(can_run_list) == 0:
    return None

can_run_set = set(can_run_list)
self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_set]
if adder.preempt_list:
    for req in adder.preempt_list:
        self._add_request_to_queue(req)

if adder.new_chunked_req is not None:
    assert self.chunked_req is None
    self.chunked_req = adder.new_chunked_req

if self.chunked_req is not None:
    self.chunked_req.is_chunked += 1

self.adder = adder
self.can_run_list = can_run_list
self.running_bs = len(self.running_batch.reqs)

set_time_batch(can_run_list, "set_forward_entry_time")

new_batch = ScheduleBatch.init_new(
    can_run_list,
    self.req_to_token_pool,
    self.token_to_kv_pool_allocator,
    self.tree_cache,
    self.model_config,
    self.enable_overlap,
    self.spec_algorithm,
    chunked_req=self.chunked_req,
)
self.max_prefill_bs = max(self.max_prefill_bs, len(can_run_list))
if self.enable_hierarchical_cache:
    new_batch.hicache_consumer_index = (
        self.tree_cache.ready_to_load_host_cache()
    )

new_batch.prepare_for_extend()
new_batch.prefill_stats = PrefillStats.from_adder(...)
```

要点：

- **从 waiting_queue 物理移除**已加入的请求(用 set diff,避免 O(n²))。
- **`adder.preempt_list`**：被抢占的请求重新入队(可能下一轮再上车)。
- **`new_chunked_req`**：本轮新产生的 chunked 请求(input 太长被拆),记到 scheduler 状态。
- **`is_chunked += 1`**:每被 chunk 一次计数 +1,用于 metrics。
- **`ScheduleBatch.init_new`**:把 can_run_list + KV/Cache 引用 + spec/forward 配置打包成 ScheduleBatch 对象。
- **`prepare_for_extend`**：把请求各自的 `prefix_indices / extend_input_len` 等汇总成 batch 级 tensor(`input_ids / positions / req_pool_indices / seq_lens`),为 forward 做好准备。

---

## 七 mixed-style chunked prefill

```python
if (self.is_mixed_chunk
    and not self.running_batch.is_empty()
    and not (new_batch.return_logprob or self.running_batch.return_logprob)
    and new_batch.input_embeds is None):
    self.running_batch.filter_batch(v1_spec_info_filtered=True)
    if not self.running_batch.is_empty():
        self.running_batch.prepare_for_decode()
        new_batch.mix_with_running(self.running_batch)
        new_batch.decoding_reqs = self.running_batch.reqs
    self.running_batch = ScheduleBatch(reqs=[], batch_is_full=...)
else:
    new_batch.decoding_reqs = None
```

mixed chunked prefill 的核心思想:**同一次 forward 内同时跑 prefill + decode**——把 running_batch(decode 中的请求) 拼到 new_batch(prefill 请求) 里,attn backend 用 unified 模式同时处理两种 forward_mode。

好处：不需要等 prefill 跑完才能 decode,降低 ITL 抖动。代价：实现复杂(部分功能不兼容,如 logprob、input_embeds)。

执行后 `running_batch` 清空(请求都进了 new_batch),下一轮 `last_batch=new_batch`,merge 时再分回。

---

## 八 设计要点小结

| 决策 | 原因 |
|---|---|
| PrefillAdder 状态机式打包 | 把「KV 预算 / chunk size / LoRA / RadixCache」多种约束统一到一个 add_one_req 接口 |
| RadixCache 命中 → 不重复 prefill | 多轮对话和热点 prompt 几乎零成本 |
| HiCache prefetch 异步检查 | 避免阻塞主循环;预取没完成的请求暂时跳过 |
| chunked prefill 上限 | 控制单次 prefill 占用 forward 时间,降低对 decode 的 ITL 影响 |
| dynamic chunking | 已 prefill 越长,下一 chunk 可以越大(attn cost 不敏感) |
| LoRA batch 校验 | 同 batch adapter 数量有限,避免 BGMV 越界 |
| mixed chunked prefill | 进一步把 prefill 和 decode 在 forward 层叠加,消除 prefill→decode gap |

---

## 九 衔接

返回的 `new_batch`(or None) 回到 `get_next_batch_to_run`:

- 不是 None → 作为本轮 forward 的 batch。
- 是 None → 上层考虑跑 decode(`update_running_batch`)。
