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

### 3.3 Path A:overlap schedule(GPU 部署的默认路径,务必理解)

> 这一节面向初学者重点展开。`enable_overlap=True` 是 GPU 部署默认值,你的请求大概率走这条路径。

#### 3.3.1 它要解决的问题:同步路径 GPU 闲置

先看 Path C(同步) 的时间轴。每轮主循环 CPU 和 GPU 必须**串行**:

```
轮 1: │CPU 组batch│GPU forward (1)               │CPU 处理结果│
轮 2:                                            │CPU 组batch│GPU forward (2)              │CPU 处理结果│
轮 3:                                                                                       │CPU 组batch│ ...
                                                                                                          ─► 时间
                                                                                              ▲
                                                              在「CPU 组batch」期间,GPU 完全闲置!
```

观察:

- 一次 `model.forward` 在 GPU 上需要几毫秒(decode 阶段尤其短)。
- 但 CPU 组下一个 batch(收 ZMQ、分配 KV slot、构造 attn metadata、launch CUDA kernel)也要 **几百微秒到几毫秒**。
- 这两段**互相等待**——decode 阶段 CPU 时间甚至能占总时间 30-50%。

**目标**:让 CPU 组 batch 的时间 **隐藏在** GPU forward 的时间里,GPU 永远满载。

```
理想效果:
GPU stream:    │forward (1)    │forward (2)    │forward (3)    │  ← GPU 不停
CPU stream:    │组2│发起2│组3│发起3│组4│发起4│           ← CPU 提前组下一批
                            ─► 时间
```

总耗时从 `T_cpu + T_gpu` 降为 `max(T_cpu, T_gpu)`。

---

#### 3.3.2 核心难题:下一批的 input_ids 还不存在

让两条 stream 并行听起来很简单,但有个**鸡生蛋**问题:

**轮 N+1 的 `input_ids` 是轮 N 生成的 next_token_ids**(自回归)。

但在 overlap 模式下,scheduler 想在轮 N 的 forward **还没跑完** 就开始组轮 N+1 的 batch——此时 `next_token_ids` 还在 GPU 里没算出来,CPU 端拿不到。

```
   t=0  CPU: 组 batch N
   t=1  CPU: 发起 forward N → GPU 开始算
   t=2  CPU: 想组 batch N+1, 但 input_ids = ??? 还没算出来
                                                ↑ 难题
```

---

#### 3.3.3 解决方案:`future_indices` 占位

SGLang 的核心 trick:**用「未来位置编号」代替具体值**。

打个比方——你在快递柜寄件:

> 寄件时柜子还没有新包裹,但你预先**抢一个空格子的编号**(比如「第 7 号格子」)。回头别人来取件时,只要给「第 7 号格子」这个编号,系统就能把已经放进去的包裹拿出来——不用再问「具体是什么包裹」。

`future_indices` 就是这个编号:

```python
future_indices = self.future_map.alloc_future_indices(bs)  # 分配 batch_size 个空格子编号
```

机制:

- **轮 N 一开始**:scheduler 在 `future_map`(GPU 上的一段连续显存) 里**预订 batch_size 个空槽位**,得到 `future_indices`(比如 [7, 8, 3, 5])。
- **batch.output_ids 写成负数索引**:`-7, -8, -3, -5`(负号是约定:这些不是真 token,是 future 占位符)。
- **CPU 立刻继续组下一批**:把这些负数当作 input_ids 直接用。
- **轮 N forward 完成后**:GPU 上 `future_map[7], future_map[8], ...` 自动写入实际生成的 token id。
- **轮 N+1 forward 开始前**:GPU 上 `resolve_future` 算子扫描 input_ids,看到负数 `-7` 就到 `future_map[7]` 取真实 token——**这一切在 GPU 上完成,CPU 不参与**。

```python
# 关键三行
future_indices = self.future_map.alloc_future_indices(bs)        # ① 抢编号
self.future_map.resolve_future(model_worker_batch)               # ③ GPU 内部:负数→真值
batch_result.copy_to_cpu(return_logprob=...)                     # ④ next_token 异步拷 CPU
self.future_map.store_to_map(future_indices, batch_result)       # ② 把 forward 输出写回编号
future_indices_or_next_token_ids = -future_indices.indices       # 给下一轮用的占位
```

---

#### 3.3.4 CUDA stream 是什么

要让 CPU 调度和 GPU forward 并行,得引入「**多条 GPU 指令流**」概念。

**类比**:GPU 像一个工厂,有多条**流水线**(stream)。同一条流水线上的指令严格按顺序跑;不同流水线可以**真正同时**执行(GPU 硬件支持多 stream 调度)。

SGLang 用两条 stream:

| stream | 跑什么 | 默认 |
|---|---|---|
| `schedule_stream` | 调度相关:KV 元信息计算、batch tensor 拷贝、`future_map.alloc/resolve` | scheduler 主线程默认在这条上发 |
| `forward_stream` | 真正的 model.forward(matmul、attention、LayerNorm 等) | 用 `with self.forward_stream_ctx:` 切换过去 |

**关键操作 `forward_stream.wait_stream(schedule_stream)`**:让 forward stream「**等到 schedule stream 跑到这一刻**」再开始——这是必需的同步点,因为 forward 要用 schedule stream 上准备的 KV slot 和 metadata。

但其他时刻两条 stream 完全独立——这就是「**overlap**」的物理基础。

---

#### 3.3.5 一轮 overlap 的完整时序

```
schedule_stream:  │组 batch N│alloc_fut│发起 fwd N│---组 batch N+1---│alloc_fut│发起 fwd N+1│
                                            ↓ wait_stream
forward_stream:                              │resolve_future│model.forward N            │store_to_map│copy_to_cpu│
                                                                                                             ↓ event(copy_done)
CPU 主线程:        ─继续组下一批,不等 GPU──────────────────────────────────────────►

                 ──────────────────────────────────时间─────────────────────────────────►
```

红色解释:

- **alloc_fut**:在 future_map 里预订槽位(只是在 GPU 上选几个 index,几乎零成本)。
- **发起 fwd N**:在 schedule_stream 上 launch 几个 CUDA kernel 把任务送到 forward_stream(launch 是异步的,CPU 不阻塞)。
- **resolve_future**:把上一轮的负数占位符在 GPU 上替换成真实 token id(forward N-1 已写过 future_map)。
- **model.forward N**:真正的几千个 kernel,占用大部分时间。
- **store_to_map**:把 forward 输出写回 future_map,供下一轮 resolve 用。
- **copy_to_cpu + copy_done**:next_token_ids 异步从 GPU 拷到 CPU(`copy_to_cpu` 立即返回,完成时间通过 `copy_done` event 通知)。

CPU 主线程从「发起 fwd N」之后就**不等了**,立即去组 batch N+1——只要 forward N 在 GPU 上算的同时,CPU 把下一批准备好,流水线就不停。

---

#### 3.3.6 几个辅助机制(防出错)

##### `record_batch_in_overlap`:防 GC

```python
self.record_batch_in_overlap(model_worker_batch)
```

问题:`model_worker_batch` 是个 dataclass,持有大量 GPU tensor 引用。CPU 主线程发起 forward 后立即返回,Python 的引用计数掉到 0,**PyTorch 可能在 forward 还没跑完就 GC 掉这些 tensor**——GPU 读到野指针就崩溃。

解决:把 `model_worker_batch` 多塞到一个**双缓冲队列** `batch_record_buf[0..1]` 里,保证至少存活到下一轮被覆盖时(那时 forward 早跑完了)。

##### `sampling_info.copy_for_forward()`:防脏读

```python
model_worker_batch.sampling_info = model_worker_batch.sampling_info.copy_for_forward()
```

问题:forward 过程中 sampler 会**原地修改** `sampling_info`(比如 frequency penalty 把已生成 token 的频率累计上去)。如果不 copy,CPU 主线程在轮 N+1 组 batch 时读到的 sampling_info 已经被轮 N 改脏了。

解决:每轮 fork 一份副本给 forward 用,scheduler 端的原版不动。

##### `copy_done` event:异步等数据

```python
batch_result.copy_done = self.device_module.Event()
batch_result.copy_to_cpu(return_logprob=...)
```

`copy_to_cpu` 把 next_token_ids 从 GPU 拷到 CPU 也是异步操作,**调用立即返回**,实际拷贝在 stream 上排队。`Event` 是个 GPU 同步信号:谁要用这批 next_token_ids 时,先 `event.synchronize()` 等一下,确保拷贝完成。

##### `delay_sample_func`:延迟采样

```python
if batch_result.delay_sample_func is None:
    self.future_map.store_to_map(future_indices, batch_result)
    batch_result.copy_to_cpu(...)
else:
    batch_result.future_indices = future_indices
```

某些场景(spec v2、grammar 约束) **采样要等到下一 forward 之后** 才能完成——比如 grammar mask 依赖下一轮的状态。这种情况:本轮先只跑 forward 把 logits 算出来,**不立即 sample**,把 future_indices 存着,等到 `launch_batch_sample_if_needed` 时(下一轮处理上一批结果时)再 sample。

##### `update_cache_from_scheduler`:在 Path A 不需要

Path C 末尾要 `update_cache_from_scheduler(batch, batch_result)` 把生成 token 的 KV 写回 RadixCache。Path A 不需要,因为这一步**通过 `process_batch_result` 在异步路径上完成**(那时 copy_done 已经触发)。

---

#### 3.3.7 一句话理解

> **overlap schedule = 用「期货」(future_indices) 让 CPU 不必等 GPU**。
>
> 调度时 CPU 给每个 next_token 抢一个 GPU 槽位编号占位,batch.output_ids 写成负数指向这些编号。CPU 立刻去组下一批,把负数当真实 token 用。GPU 上 forward 跑完后把真值写到那个编号;下一轮 forward 开始前 GPU 内部用 `resolve_future` 把负数换成真值——CPU 全程不阻塞,**GPU 永远满载**,这就是 SGLang 高吞吐的核心机制。

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
