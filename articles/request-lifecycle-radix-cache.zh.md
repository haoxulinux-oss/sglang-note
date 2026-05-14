# 一个请求在 RadixCache 视角下的完整生命周期(初学者向)

> 📖 **关联阅读**:
> - 先读:[RadixCache 数据结构详解](radix-cache-structure.zh.md)——本文假设你已经知道 TreeNode、`match_prefix`、`split_node`、`lock_ref` 等概念
> - 父篇:[`FlashInferAttnBackend.forward_extend()` 详解](flashinfer-forward-extend.zh.md) §八 「命中的前缀什么时候参与 attention」
> - 上游入口:[`handle_generate_request()` 解析](scheduler-handle-generate-request.zh.md)、[`_add_request_to_queue()` 解析](scheduler-add-request-to-queue.zh.md)

代码位置:涉及 `scheduler.py` / `schedule_policy.py` / `schedule_batch.py` / `radix_cache.py` / `flashinfer_backend.py`,具体行号在每个阶段给出。

---

## 一 跟踪场景设定

为了把所有变化具象化,我用一个完整可重现的场景:

**Llama-3-8B,page_size=16,KV pool size=65536。系统已运行一段时间,树里已经有共享前缀**。

```
当前 RadixCache 状态:

       root (lock_ref=1)
         │
         └── node_X
              key=[A,B,C,...,X1] (1024 个 token,共 64 页)
              value=[slot 100..1123] (1024 个物理 slot)
              lock_ref=0  ← 没人在用,可被淘汰
              children={}
```

**新请求 R 到来**,prompt = `[A,B,C,...,X1, Y1,Y2,...,Y100]`(共 1124 个 token,前 1024 个和 node_X 一样,后 100 个全新),`max_new_tokens=50`。

下面先看一张**总流程图**,建立整体框架感,再逐阶段展开。

---

## 二 总函数调用流程图(先看这张图)

下图把请求 R 从入队到结束的所有关键调用画成一张图。**每个 T0~T7 阶段在图上都有明确的位置**——后面 §三~§十 各小节就是把每个阶段在这张图上展开讲解。

```
═══════════════════════════════════════════════════════════════════════════════
                          ★ T0 阶段:请求入队 ★
═══════════════════════════════════════════════════════════════════════════════

  TokenizerManager(另一进程)
    │  zmq 发送 TokenizedGenerateReqInput
    ↓
  Scheduler.recv_requests()                           ← scheduler.py:event_loop 内
    └─ Scheduler.process_input_requests()
        └─ Scheduler.handle_generate_request()        ← scheduler.py
              │
              │  构造 Req 对象,初始化各字段(req_pool_idx=None 等)
              ↓
        Scheduler._add_request_to_queue(req)          ← scheduler.py
              │
              │  self.waiting_queue.append(req)        ★ T0 完成,req 等候在队列里
              ↓
        (返回主循环 event_loop_normal)

═══════════════════════════════════════════════════════════════════════════════
                ★ T1~T3 阶段:scheduler 主循环组 batch ★
═══════════════════════════════════════════════════════════════════════════════

  Scheduler.event_loop_normal()                       ← scheduler.py:1384
    └─ Scheduler.get_next_batch_to_run()              ← scheduler.py:2302
         │
         │  分流到 prefill 或 decode
         ↓
         Scheduler.get_new_batch_prefill()            ← scheduler.py:2419
              │
              │  ★ T1 阶段:match_prefix(由调度策略触发)★
              │
              ├─ self.policy.calc_priority(self.waiting_queue, self.running_batch)
              │       │                                       ← scheduler.py:2474
              │       │  (实例:SchedulePolicy)
              │       │
              │       └─ SchedulePolicy.calc_priority()       ← schedule_policy.py:117
              │              │  if isinstance(policy, CacheAwarePolicy):
              │              ↓
              │              SchedulePolicy._compute_prefix_matches(waiting_queue, policy)
              │                     ← schedule_policy.py:185
              │                     │  (对 waiting_queue 全员批量做 prefix 匹配)
              │                     │
              │                     └─ self.tree_cache.match_prefix(MatchPrefixParams(...))
              │                            ← radix_cache.py:398
              │                            └─ _match_prefix_helper()    ← radix_cache.py:693
              │                                  (可能触发 _split_node()) ← radix_cache.py:719
              │                            返回 (device_indices, last_node, host_hit_length)
              │                     │
              │                     └─ 写到 r.prefix_indices / r.last_node / r.host_hit_length
              ↓
              ┌──────── PrefillAdder.add_one_req() 主流程 ──────┐
              │                                                │
              │  ★ T2 阶段:admit + lock + alloc ★            │
              │                                                │
              │  PrefillAdder.add_one_req(req)                 │  ← schedule_policy.py:780+
              │       └─ with self._lock_node(req.last_node):  │  ← schedule_policy.py:664
              │              │   inc_lock_ref(临时护栏,with 退出会 dec)
              │              ↓
              │              tree_cache.inc_lock_ref(node)     │  ← radix_cache.py:637
              │              │
              │              self.can_run_list.append(req)     │
              │              self._req_inc_lock_ref(req)       │  ★ 长期锁(给请求持有)
              │              │   (在 with 内升级成长期锁,详见 §五)
              └────────────────────────────────────────────────┘
              │
              ↓
              ReqToTokenPool.alloc(reqs)              ← memory_pool.py:156
                  └─ 给 req 分配 req_pool_idx
              ↓
              ScheduleBatch.prepare_for_extend()      ← schedule_batch.py:1598
                  │
                  ├─ token_to_kv_pool_allocator.alloc(extend_num_tokens)
                  │     ↑ 分配新物理 KV slot
                  │
                  └─ req_to_token_pool.write(...)
                        ↑ 填好 req_to_token[req_pool_idx][0:1124] 映射
              ↓
              ★ T3 阶段:prepare ForwardBatch ★

              ScheduleBatch.get_model_worker_batch()  ← schedule_batch.py:2431
                  └─ ModelWorkerBatch
                       ├─ extend_prefix_lens          ★ ★ 关键字段 ★ ★
                       ├─ extend_seq_lens
                       ├─ seq_lens
                       ├─ out_cache_loc
                       ├─ req_pool_indices
                       └─ ...
              ↓
              返回新 batch 给主循环

═══════════════════════════════════════════════════════════════════════════════
                   ★ T4 阶段:worker / kernel 实际执行 ★
═══════════════════════════════════════════════════════════════════════════════

  Scheduler.run_batch(batch)                          ← scheduler.py:2767
    └─ self.tp_worker.forward_batch_generation(model_worker_batch)
                                                      ← tp_worker.py:443
         │
         ├─ ForwardBatch.init_new(...)               ← forward_batch_info.py
         │     │  把 ModelWorkerBatch 转成 ForwardBatch
         │     │  挂上 token_to_kv_pool / attn_backend
         │     │  extend_prefix_lens 传进来
         │     ↓
         │
         ├─ model_runner.forward(forward_batch)       ← model_runner.py:2896
         │     │
         │     └─ FlashInferAttnBackend.init_forward_metadata(forward_batch)
         │     │       ← flashinfer_backend.py:433
         │     │  (执行一次 plan,绑定 paged_kv_indices 等元数据)
         │     │
         │     ↓
         │     for layer_id in range(num_layers):     ← 32 层循环
         │         │
         │         └─ LlamaForCausalLM.forward()      ← llama.py:510
         │              └─ LlamaModel.forward()        ← llama.py:366
         │                   └─ LlamaDecoderLayer.forward()  ← llama.py:303
         │                        └─ LlamaAttention.forward() ← llama.py:220
         │                             ├─ qkv_proj + split + RoPE
         │                             └─ RadixAttention.forward(q, k, v, forward_batch)
         │                                  ← radix_attention.py:99
         │                                  │
         │                                  └─ forward_batch.attn_backend.forward(...)
         │                                      ↓
         │                                      FlashInferAttnBackend.forward_extend(q, k, v, layer, ...)
         │                                          ← flashinfer_backend.py:775
         │                                          │
         │                                          ├─ token_to_kv_pool.set_kv_buffer()
         │                                          │     ← memory_pool.py:1022
         │                                          │  写新 K, V 到 slot [2000..2099]
         │                                          │
         │                                          ├─ prefill_wrapper_ragged.forward(q, k_new, v_new)
         │                                          │  → o1, s1   (对本批新 K/V 算 attention)
         │                                          │
         │                                          ├─ prefill_wrapper_paged.forward(q, full_cache)
         │                                          │  → o2, s2   (对 prefix K/V 算 attention)
         │                                          │
         │                                          └─ merge_state(o1, s1, o2, s2)
         │                                              → 最终 attention 输出
         │
         └─ model_runner.sample(logits)               ← model_runner.py:3070
              (采样 next_token_ids 返回给 scheduler)

═══════════════════════════════════════════════════════════════════════════════
              ★ T5 阶段:prefill 完成,新 token 挂回 RadixCache ★
═══════════════════════════════════════════════════════════════════════════════

  Scheduler.process_batch_result(batch, result)       ← scheduler.py:2950
    └─ Scheduler.process_batch_result_prefill()       ← scheduler_output_processor_mixin.py:126
         │
         │  if req 已完成 (EOS / max_new_tokens):
         │      → 跳到 T7
         │  else (req 还要继续 decode):
         │      ↓
         └─ tree_cache.cache_unfinished_req(req)      ← scheduler_output_processor_mixin.py:200
                ↓
                RadixCache.cache_unfinished_req(req)  ← radix_cache.py:535
                     │
                     ├─ token_ids = req.fill_ids       (本批所有 token,1124 个)
                     ├─ kv_indices = req_to_token_pool.req_to_token[5][:1124]
                     ↓
                     ├─ self.insert(InsertParams(key, value))  ← radix_cache.py:468
                     │     └─ _insert_helper()                  ← radix_cache.py:749
                     │          创建新节点 node_Y(挂在 node_X 下)
                     ↓
                     ├─ token_to_kv_pool_allocator.free(重复 slot)
                     ├─ self.match_prefix(...)         (重新匹配以拿新 indices)
                     ├─ req_to_token_pool.write(...)   (更新映射)
                     ↓
                     ├─ self.dec_lock_ref(req.last_node)   ← 释放对旧 last_node 的锁
                     └─ self.inc_lock_ref(new_last_node)   ← ★ last_node 切换到 node_Y

═══════════════════════════════════════════════════════════════════════════════
                ★ T6 阶段:decode 阶段循环 50 次 ★
═══════════════════════════════════════════════════════════════════════════════

  (回到主循环,req 进入 running_batch)
  Scheduler.event_loop_normal()       ← 又一轮
    └─ Scheduler.update_running_batch(batch)         ← scheduler.py:2669
         └─ ScheduleBatch.prepare_for_decode()       ← schedule_batch.py:2179
              └─ token_to_kv_pool_allocator.alloc(1)     ★ 每轮分配 1 个 slot
              └─ req_to_token_pool.req_to_token[5][i] = new_slot

    └─ Scheduler.run_batch(batch)                    ← forward_mode = DECODE
         └─ ... (同 T4 但走 FlashInferAttnBackend.forward_decode)
              ← flashinfer_backend.py:889

    └─ Scheduler.process_batch_result_decode()       ← scheduler_output_processor_mixin.py:390
         │  (decode 阶段不 cache_unfinished_req,树不动)
         │  累积 output_ids
         ↓
  循环 50 次,直到 EOS 或达到 max_new_tokens

═══════════════════════════════════════════════════════════════════════════════
            ★ T7 阶段:请求结束,KV 完全归还 RadixCache ★
═══════════════════════════════════════════════════════════════════════════════

  Scheduler.process_batch_result_decode()
    └─ (检测到 req.finished_reason)
    └─ release_kv_cache(req, self.tree_cache)        ← mem_cache/common.py:479
         │
         ├─ tree_cache.cache_finished_req(req)       ← radix_cache.py:488
         │     ↓
         │     ├─ insert(全部 fill_ids + output_ids 进树)
         │     │    创建 node_Z(挂在 node_Y 下)
         │     │
         │     ├─ token_to_kv_pool_allocator.free(重复 slot)
         │     ├─ token_to_kv_pool_allocator.free(未对齐尾部)
         │     │
         │     └─ self.dec_lock_ref(req.last_node)
         │          沿 node_Y → node_X → root 一路 dec_ref
         │          ★ 所有节点 lock_ref 归零,可被淘汰
         │
         ├─ req_to_token_pool.free(req_pool_idx)     ← 归还 req 的 slot
         │
         └─ scheduler 把 req 从 running_batch 移出,把结果 stream 给 detokenizer
```

**几个关键观察**:

1. **T1 / T2 紧挨着**——`match_prefix` 完后立刻进 admit + lock,中间不释放,**避免在临界区中被 evict**
2. **T3 / T4 / T5 一气呵成**——一次 `run_batch` 调用内完成「prepare → forward → cache_unfinished_req」三件事
3. **T4 才是真正接触 GPU 的地方**——前面 T0~T3 都是 CPU 端的元数据准备
4. **T6 是循环**——每生成一个 token 都走一遍 prepare_for_decode → forward_decode,**RadixCache 树形完全不变**
5. **T5 和 T7 都调 RadixCache.insert**——但 T5 是「中途同步」(请求还没结束),T7 是「最终归档」

下面 §三~§十 把每个阶段在这张图上展开,讲清每个调用的输入输出和关键变量。

---

## 三 阶段 T0:请求进入 scheduler

**入口**:`Scheduler.handle_generate_request()`(`scheduler.py`)

```python
# 把 GenerateReqInput 包成 Req 对象
req = Req(
    rid=...,
    origin_input_ids=[A,B,C,...,X1, Y1,...,Y100],   # 长度 1124
    output_ids=[],
    sampling_params=...,
    ...
)

# Req 关键字段初始化(schedule_batch.py:614 及周边)
req.fill_ids = []                       # 还没拼好,初始为空
req.prefix_indices = empty(0)            # 还没匹配前缀
req.last_node = None
req.host_hit_length = 0
req.req_pool_idx = None                  # ★ 还没分配 req slot
req.kv_committed_len = 0
req.cache_protected_len = 0
req.extend_input_len = 0

# 入 waiting_queue
self._add_request_to_queue(req)
```

**此时 RadixCache 完全不知道这个请求存在**——树没变化,`lock_ref` 没变。

### 关键变量快照(T0)

```
req.req_pool_idx        = None
req.prefix_indices      = tensor([])
req.last_node           = None
req.fill_ids            = []

RadixCache:
  node_X.lock_ref       = 0  (没变)
  evictable_size_       = 1024
  protected_size_       = 0
```

---

## 四 阶段 T1:scheduler 取出请求,做 `match_prefix`

**入口**:`SchedulePolicy._compute_prefix_matches()`(`schedule_policy.py:185`,**带下划线的私有方法**)。`Scheduler.get_new_batch_prefill()` 在 `scheduler.py:2474` 调 `self.policy.calc_priority(self.waiting_queue, self.running_batch)`,后者再进 `_compute_prefix_matches` 给本批 `waiting_queue` 全员做 prefix 匹配。

调用链:

```
Scheduler.get_new_batch_prefill()       ← scheduler.py:2419
   └─ self.policy.calc_priority()       ← schedule_policy.py:117 (SchedulePolicy 实例)
        └─ self._compute_prefix_matches(waiting_queue, policy)
                                        ← schedule_policy.py:185
```

```python
for r in waiting_queue:
    prefix_ids = r.origin_input_ids + r.output_ids
    # prefix_ids = [A,B,C,...,X1, Y1,...,Y100]  长度 1124

    match_result = self.tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=prefix_ids, extra_key=...))
    )
    r.prefix_indices, r.last_node, ..., r.host_hit_length = (
        match_result.device_indices,    # ★ 命中的物理 slot 索引
        match_result.last_device_node,  # ★ 树里最后命中的节点
        ...
    )
```

**`match_prefix` 内部发生的事**(回顾 [RadixCache 结构](radix-cache-structure.zh.md) §四):

1. 从 root 出发,沿 children 走
2. 找到 `node_X`,匹配前 1024 token——**完全匹配 `node_X.key`**(`prefix_len == len(node.key)`)
3. 继续往下找子节点,但 `node_X.children == {}`——退出循环

**返回**:
- `device_indices` = `node_X.value` = `tensor([100, 101, ..., 1123])`(1024 个物理 slot)
- `last_device_node` = `node_X`

### RadixCache 树的变化:**无**

`match_prefix` 在这个例子里**没有触发 split**(因为 `prefix_len == len(node.key)` 完美匹配)。树形完全不变。

> 如果 prompt 只匹配 node_X 的前 800 token(不到完整 1024),则会触发 `_split_node`,把 node_X 切成两段——见 [RadixCache 结构](radix-cache-structure.zh.md) §4.1。

### 关键变量快照(T1)

```
req.req_pool_idx        = None        (还没分配)
req.prefix_indices      = tensor([100, 101, ..., 1123])  ← 1024 个 slot
req.last_node           = node_X      ← ★ 指向树里的节点
req.fill_ids            = []          (还没设置)

RadixCache:
  node_X.lock_ref       = 0   ← ★ 还没 inc,因为 match_prefix 只是「查」
  node_X.last_access_time = now()  ← 更新了
  evictable_size_       = 1024
  protected_size_       = 0
```

**关键观察**:`match_prefix` **不会** inc lock_ref——只是查询。这意味着**理论上 node_X 在 match 和 admit 之间还可能被 evict**——所以 admit 是一个紧凑的临界区(见下一阶段)。

---

## 五 阶段 T2:`PrefillAdder.add_one_req` admit + 锁住 node + 分配新 slot

**入口**:`PrefillAdder.add_one_req()`(`schedule_policy.py:780+`)

```python
# 计算 extend_input_len = 总长度 - 已命中前缀长度
real_input_tokens = req.extend_input_len - req.host_hit_length    # 见 §5.1
real_input_tokens = self.ceil_paged_tokens(real_input_tokens)
prefix_len = len(req.prefix_indices)        # 见 §5.1,这两个数不是一回事

# 关键:进入临界区,锁住 node_X
with self._lock_node(req.last_node):       # ★ inc_lock_ref(node_X)
    # 此时 node_X.lock_ref 至少为 1,不会被 evict

    # 检查 KV pool 容量够不够
    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN

    # 给请求分配 req_pool_idx
    self.can_run_list.append(req)
    self._req_inc_lock_ref(req)             # ★ 长期锁(详见 §5.2 双层锁机制)
```

### 5.1 `len(req.prefix_indices)` 和 `req.host_hit_length` 是同一回事吗?

**不是**。它们是「两种不同来源的前缀命中」,数值上**独立、叠加**。

| 字段 | 在哪 | 表征 |
|---|---|---|
| **`req.prefix_indices`** | **device 命中**——KV 数据已经在 GPU HBM 里 | int64 张量,**`len(...)` = device 命中 token 数** |
| **`req.host_hit_length`** | **host 命中**——KV 备份在 host RAM(HiCache 第 2 层),还没拉回 GPU | 整数,host 命中的额外 token 数 |

`match_prefix` 同时返回两者(`schedule_policy.py:204`):

```python
(r.prefix_indices, r.last_node, r.last_host_node, r.host_hit_length) = (
    match_result.device_indices,        # ← prefix_indices = device 命中索引
    match_result.last_device_node,
    match_result.last_host_node,
    match_result.host_hit_length,       # ← host 命中长度(超出 device 部分)
)
```

完整 prompt 的命中视图:

```
                                完整 prompt(1124 token)
   ├────────────────────────────────────────────────────────────────┤

   ├─ device 命中(1024) ─┤─ host 命中(50) ─┤─ 需要重新 prefill(50) ─┤
   │   prefix_indices       │   host_hit_length│   real_input_tokens     │
   │   GPU 上的 slot 索引   │   在 RAM 里,等会儿 │   完全新算              │
   │                        │   init_load_back   │
   │                        │   拉回 GPU         │
```

#### 那 4 行代码的含义对照(以 device=1024, host=50 为例)

```python
real_input_tokens = req.extend_input_len - req.host_hit_length    # 100 - 50 = 50
                  = "extend 范围内,扣除 host 能补的,真正要 GPU 重算的 token 数"
real_input_tokens = self.ceil_paged_tokens(real_input_tokens)     # page_align 到 64

prefix_len = len(req.prefix_indices)                              # 1024
           = "device 上已经占的 slot 数"
           = 后面 _update_prefill_budget(prefix_len, ...) 和 req.cache_protected_len = prefix_len 用
```

> **每一行用途不同**:
> - `real_input_tokens` 算「还要算多少 token」——`extend_input_len` 已经扣过 device 命中,再扣 host 命中
> - `prefix_len` 算「device 上已经占了多少 slot」——决定 KV cache 保护边界

#### host 命中什么时候变成 device 命中

紧接着 `add_one_req` 后面(`schedule_policy.py:830`):

```python
if req.host_hit_length > 0:
    new_indices, req.last_node = self.tree_cache.init_load_back(...)   # ★ host→device DMA
    req.prefix_indices = torch.cat([req.prefix_indices, new_indices])  # 合并到 device 索引
    req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
    prefix_len = len(req.prefix_indices)                                # ← 现在变成 1074
    req.cache_protected_len = prefix_len
```

`init_load_back` 内部从 host RAM 把 KV 拷回 GPU,新分配的 device slot 索引拼到 `prefix_indices` 后面。**load-back 之后,attention kernel 不再区分「device 原生」和「host 拉回」**,统一看 `extend_prefix_lens = 1074`。

#### 5.1.1 `req.host_hit_length` 是哪里赋值的

整个生命周期里 `req.host_hit_length` 在 **两处**被设置,**两处都来自 `tree_cache.match_prefix()` 的返回值**:

##### 位置 ①:`SchedulePolicy._compute_prefix_matches`(`schedule_policy.py:204-214`)

scheduler 主循环每轮组 batch 之前批量做(用于排队优先级排序):

```python
for r in waiting_queue:
    prefix_ids = r.origin_input_ids + r.output_ids
    match_result = self.tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=prefix_ids, extra_key=...))
    )
    (r.prefix_indices, r.last_node, r.last_host_node, r.host_hit_length) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.host_hit_length,           # ★ 第一次赋值
    )
```

##### 位置 ②:`Req.init_next_round_input`(`schedule_batch.py:1002-1015`)

`Scheduler.get_new_batch_prefill` 在 `scheduler.py:2559` 对**每个 req** 调一次 `req.init_next_round_input(self.tree_cache)`,再做最终匹配(因为前面 `_compute_prefix_matches` 之后,树可能被别的 req 改过,需要重新对齐):

```python
def init_next_round_input(self, tree_cache=None, ...):
    self.fill_ids = self.origin_input_ids + self.output_ids
    ...
    match_result = tree_cache.match_prefix(
        MatchPrefixParams(key=RadixKey(token_ids=token_ids, ...))
    )
    (self.prefix_indices, self.last_node, self.last_host_node,
     self.host_hit_length, self.mamba_branching_seqlen) = (
        match_result.device_indices,
        match_result.last_device_node,
        match_result.last_host_node,
        match_result.host_hit_length,           # ★ 第二次赋值(最终生效的)
        ...,
    )
```

**两处的差别**:
- ① 是「整批 waiting_queue 的预 match」,目的是排序(LPM / DFS 等策略需要知道每个 req 的前缀长度)
- ② 是「真正决定 admit 时用的 match」,**这次的结果才是后面 `add_one_req` 看到的值**

> 初学者只需记住:**`host_hit_length` 的来源永远是 `tree_cache.match_prefix()` 的返回值**,具体由 `HiRadixCache.match_prefix` 算出来。

##### `match_result.host_hit_length` 又是怎么算的

源头在 `HiRadixCache.match_prefix`(`hiradix_cache.py:1239-1242`):

```python
host_hit_length = 0
last_host_node = last_node
while last_node.evicted:                            # 节点的 device KV 已被淘汰
    host_hit_length += len(last_node.host_value)    # 但 host 上还有备份
    last_node = last_node.parent
```

**算法**:从 device 匹配到的最深节点开始,**向 root 方向走**,遇到「device 已淘汰但 host 还有备份」(`node.evicted == True` 且 `node.host_value is not None`)的节点,把它的长度累加进 `host_hit_length`。一旦遇到既不在 device 也不在 host 的节点就停止。

普通 `RadixCache`(无 HiCache)永远返回 `host_hit_length=0`(`radix_cache.py:1227`),所以**没开 HiCache 时,这个字段恒为 0**——这种情况下「单卡 device 命中」就是全部。

#### 5.1.2 一句话区分两个字段

> **`len(req.prefix_indices)` = device 命中数**(GPU 上现成的 KV slot 数);**`req.host_hit_length` = host 命中数**(在 RAM 里、等会儿 `init_load_back` 才拉回 GPU 的)。两者来源都是 `match_prefix` 的返回值,**数值上独立、叠加,不是同一个东西**——只有当 HiCache 未启用或恰好无额外命中时 `host_hit_length=0`,此时仅 `prefix_indices` 一项代表前缀。

### 5.2 `_lock_node` 的双层锁机制——两次 `inc_lock_ref` 为什么必要

T2 阶段有**两个 `inc_lock_ref` 调用**:

```python
with self._lock_node(req.last_node):       # ① 临时锁(护栏)
    # ...各种容量检查、可能 return...
    self.can_run_list.append(req)
    self._req_inc_lock_ref(req)             # ② 长期锁(请求持有)
```

#### `_lock_node` 是 contextmanager,with 退出会自动 dec

源码(`schedule_policy.py:664`):

```python
@contextmanager
def _lock_node(self, last_node: TreeNode):
    try:
        self.tree_cache.inc_lock_ref(last_node)    # ← 进 with 时 +1
        yield None
    finally:
        self.tree_cache.dec_lock_ref(last_node)    # ← 退出 with 时 -1(对称)
```

**净效果**:`with` 块内 `inc + dec = 0`。

#### 两次 inc 的角色对比

| | ① `_lock_node` 的 inc | ② `_req_inc_lock_ref` 的 inc |
|---|---|---|
| **作用域** | 仅 `with` 块内(几行代码) | **请求整个生命周期**(T2 → T7) |
| **配对的 dec** | 块退出时 `finally` 自动 dec | T5 `cache_unfinished_req` 或 T7 `cache_finished_req` 里手动 dec |
| **目的** | **临界区护栏**——防止"容量检查的几 μs 内"node 被 evict 抽走 | **持久锁**——只要请求活着,prefix 路径就不能被 evict |

#### lock_ref 时间线

```
T1  match_prefix              node_X.lock_ref = 0
T2  进入 with _lock_node      node_X.lock_ref = 1   ← ① 临时 +1
        |
        | 做容量检查
        | 检查通过 → 调 _req_inc_lock_ref(req)
        | node_X.lock_ref = 2  ← ② 长期 +1
        |
T2  退出 with                  node_X.lock_ref = 1   ← ① 临时 -1 配对完成,长期锁仍在
        |
T3-T6 整个请求生命周期         node_X.lock_ref = 1   ← 持久保护
        |
T5  cache_unfinished_req      切换 last_node → node_Y(锁跟着搬,详见 §八)
T7  cache_finished_req        dec_lock_ref(node_Y) → 一路 dec 到 root
                              所有 ancestor 节点 lock_ref 归零
```

#### 为什么不能只用 ② 长期锁

如果省去 ① 只留 ②,任何在 ② 之前的提前 return / exception 都会**导致泄漏**(请求没被 admit,但也没人去 dec)。`_lock_node` 用 `with` + `try/finally` 提供**异常安全**:

```python
with self._lock_node(req.last_node):
    if not_enough_memory:
        return AddReqResult.NO_TOKEN   # ← 提前 return,finally 跑 dec,锁干净释放
    if some_assert_fails:
        raise ...                       # ← 抛异常,finally 跑 dec

    self._req_inc_lock_ref(req)         # ← 只有走到这一行,临时锁才"升级"成长期锁
```

**关键不变量**:**只有当代码走到 `_req_inc_lock_ref(req)` 这行,才升级成长期锁**——之前任何路径(return / exception),① + finally 的净效果都是 0,不会泄漏。

### 5.3 `inc_lock_ref` 触发的链路(`radix_cache.py:637`):

```python
def inc_lock_ref(self, node):  # node = node_X
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)   # 1024
            self.protected_size_ += len(node.key)   # 1024
        node.lock_ref += 1
        node = node.parent
```

**从 node_X 一路 inc 到 root**——保证 node_X 整条路径都被保护。**关键效果**:

```
node_X.lock_ref:    0  →  1     (从 evictable 移到 protected)
evictable_size_:    1024 → 0
protected_size_:    0   → 1024
```

### 5.4 接下来:`req.req_pool_idx` 的分配

`PrefillAdder` 把 req 加入 `can_run_list` 后,主循环里 `Scheduler.get_new_batch_prefill()` 调用 `ReqToTokenPool.alloc()`(`memory_pool.py:156`)。这一步既简单又关键,先把这个池子的整体结构搞清楚。

#### 5.4.1 `ReqToTokenPool` 是什么样的数据结构

类定义在 `memory_pool.py:127`,**整个池子只有两个核心字段**:

```python
class ReqToTokenPool:
    def __init__(self, size, max_context_len, device, ...):
        self.size = size                                       # 池子最多能装多少个 req
        self.max_context_len = max_context_len                 # 每个 req 最多支持多长上下文
        self.device = device

        self.req_to_token = torch.zeros(                       # ★ 主存储:二维查表
            (size, max_context_len), dtype=torch.int32, device=device
        )
        self.free_slots = list(range(size))                    # ★ 空闲行号列表
```

##### `self.req_to_token` —— 「**req idx → 它的每个 token 在 KV 池的物理 slot 编号**」二维查表

```
形状: [size, max_context_len]    例如 [4096, 32768]
dtype: int32
device: cuda

视图:
                        token 位置 0  1  2  3  4  5  ...  max_context_len-1
                                 ┌──┬──┬──┬──┬──┬──┬───┬──┐
   req_to_token[0]:  req slot 0 │  │  │  │  │  │  │   │  │
   req_to_token[1]:  req slot 1 │  │  │  │  │  │  │   │  │
   req_to_token[2]:  req slot 2 │  │  │  │  │  │  │   │  │
   ...
   req_to_token[5]:  req slot 5 │100│101│..│1123│2000│..│..│  ← R 的映射:第 0 个 token
                                 └──┴──┴──┴──┴──┴──┴───┴──┘     对应物理 KV slot 100,
                                                                  第 1 个对应 101,...
```

**每一行就是一个 req 的「token → 物理 KV slot」映射表**。形状 `[size, max_context_len]` = `[池子容量, 单个 req 最大 token 数]`,典型 size=4096 / max_context_len=32k → 4096×32768×4B ≈ 512 MB。

##### `self.free_slots` —— 「**池子里哪几行是空的**」

```
初始: [0, 1, 2, 3, 4, ..., size-1]      所有行都空着
alloc 后: 从前面取 → free_slots[i:]    例如 alloc 5 行,变成 [5, 6, ..., size-1]
free 后:  append 到末尾 → free_slots+[id]  归还的行号加回去
```

是个 **Python list**(不是 set),**FIFO 复用顺序**(先 alloc 的行先归还、先归还的先复用)。`available_size() = len(self.free_slots)`。

#### 5.4.2 `alloc` 源码逐行解释

```python
def alloc(self, reqs: list[Req]) -> Optional[List[int]]:
    # ① 找出 batch 里「已经持有 req_pool_idx」的请求 —— 它们不需要新分配
    reusing = [i for i, r in enumerate(reqs) if r.req_pool_idx is not None]
    # 注意:这里 reusing 存的是 reqs 列表里的 **下标** (0/1/2/...),不是 req_pool_idx 本身

    # ② 断言:只有「chunked prefill 中途继续」或「decode 已经committed KV」的 req 才允许复用
    assert all(
        reqs[i].is_chunked > 0 or reqs[i].kv_committed_len > 0 for i in reusing
    ), "reusing request must be chunked or have committed KV"

    # ③ 真正需要新分配的请求数
    need_size = len(reqs) - len(reusing)

    # ④ 容量不够,失败
    if need_size > len(self.free_slots):
        return None

    # ⑤ 从空闲池前面拿 need_size 个行号
    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]     # 把这几个从空闲池移除

    # ⑥ 给那些 req_pool_idx 为 None 的请求逐个赋值
    offset = 0
    for r in reqs:
        if r.req_pool_idx is None:
            r.req_pool_idx = select_index[offset]
            offset += 1
    return [r.req_pool_idx for r in reqs]
```

#### 5.4.3 三个关键概念详解

**① `reusing` 是什么**

「**已经有 req_pool_idx 的 req 的下标列表**」。什么场景会出现?

- **chunked prefill 跨 chunk 继续**:R 第一轮跑了前 5000 个 token,scheduler 把它存在 waiting_queue 等下一轮跑剩下的 5000 个,**req_pool_idx 一直保留**(已分配过)。下一轮 `alloc()` 看到 `r.req_pool_idx is not None`,直接复用同一行,不重新分配
- **decode 阶段** 其实不再调 `alloc()`(req 在 prefill 时已经拿到 row),但 chunked prefill 是个混合状态,需要这种"中途复用"
- **某些 disagg 模式**:从 prefill 实例传过来的 req 可能已经预填好 req_pool_idx

> **直观理解**:`reusing` = "这一批里有几个 req 已经占着 req_to_token 的某一行了,本次不用再给它们找位置"。

**② `need_size` 代表什么**

```
need_size = len(reqs) - len(reusing)
         = "本批 req 总数" - "已经占着行的 req 数"
         = "真正需要新分配的行数"
```

举例:本批 8 个 req,其中 2 个是 chunked-prefill 续接(已有 req_pool_idx),→ `need_size = 6`,只用从 `free_slots` 取 6 个新行号。

**③ `self.free_slots` 的作用**

它是 **"行号空闲池"**——`req_to_token` 那张大表里**有 size 行,哪些行没人用,哪些行被占了**,就靠这个 list 维护:

```
              ┌────────────┐
              │ req_to_token │  shape: [size=4096, max_context_len=32768]
              │ (整张大表)   │
              └────────────┘
                    │
                    ↓ free_slots 记录"未被占用"的行号
              [3, 8, 17, 22, 88, ..., 4095]
                ↑
                alloc(...) 从这里取走 → 这些行就归请求所有了
                free(req)  把行号还回来 → 行可被下一个请求复用
```

**注意**:`free()` 之后 `req_to_token[req_pool_idx]` 那一行**数据不清零**——下次复用前 `alloc` 也不清零,等到新 req 写进自己的 prefix_indices / out_cache_loc 时直接覆盖。"行内容"是垃圾,但**反正只有当前持有者会去读它**,无所谓。

#### 5.4.4 完整数据流总览

```
ReqToTokenPool 池
  ├─ self.req_to_token        ← 二维大表 [size, max_context_len] int32 on GPU
  │     ├─ row 0  [_______________________...]
  │     ├─ row 1  [_______________________...]
  │     ├─ row 2  [100, 101, ..., 1123, 2000, ..., 2099, _____...]  ← R 用了 row 2 (req_pool_idx=2)
  │     ├─ row 3  [_______________________...]
  │     ├─ row 4  [5000, 5001, ..., 5500, _____...]                   ← 别的请求 R' 用了 row 4
  │     ├─ ...
  │     └─ row size-1
  │
  └─ self.free_slots          ← Python list,记录哪些 row 空着
        例如 [3, 5, 6, 7, ..., size-1]   (0, 1, 2, 4 已被占用)


alloc / free 的工作循环:
                   ┌───── PrefillAdder admit 请求 ────┐
                   │ ReqToTokenPool.alloc(reqs)        │
                   │   1. 看哪些 req 已有 req_pool_idx │
                   │   2. need_size = 总数 - 已有的    │
                   │   3. 从 free_slots[:need_size] 取 │
                   │   4. 把 req.req_pool_idx 设上     │
                   └───────────────────────────────────┘
                              │
                              ↓
                   ┌───── 请求活着的整个生命周期 ─────┐
                   │ req_to_token[req_pool_idx] 装着  │
                   │ 这个请求每个 token 的物理 KV 索引 │
                   │  - prefix 部分 = node_X 的 slot   │
                   │  - extend 部分 = 新分配的 slot    │
                   │  - decode 时每步追加 1 个新 slot  │
                   └───────────────────────────────────┘
                              │
                              ↓
                   ┌───── 请求结束 cache_finished_req ┐
                   │ ReqToTokenPool.free(req)         │
                   │   self.free_slots.append(...)    │
                   │   req.req_pool_idx = None        │
                   └───────────────────────────────────┘
```

#### 5.4.5 我们的例子(R 拿到 req_pool_idx=5)

```
alloc 之前:
  free_slots = [5, 12, 17, 88, ..., size-1]      (假设 0-4 都被占了)
  req_to_token[5] = [全 0]                       (空闲行内容是垃圾)

alloc 之后:
  reusing = []                                   (R 是新来的)
  need_size = 1
  select_index = [5]
  free_slots = [12, 17, 88, ..., size-1]         (5 被取出)
  R.req_pool_idx = 5
  req_to_token[5] 还是全 0(下一步 prepare_for_extend 才写)
```

**这一步只是「占了一行,记下行号」,行内容还没填**。下一节 §5.5 才把 prefix slot 索引和新分配的 extend slot 索引写进 `req_to_token[5]`。

### 5.5 紧接着:把命中的前缀 slot 写进映射表

`prepare_for_extend` 阶段(`schedule_batch.py:1600+`),把 `req.prefix_indices` 拷到 `req_to_token_pool.req_to_token[req_pool_idx]` 的前 1024 行:

```python
# 简化伪代码
req_to_token_pool.req_to_token[5][0:1024] = req.prefix_indices  # 1024 个 slot 编号
```

**这一刻 R 的 req_to_token 表前 1024 个槽位指向 node_X 的物理 slot**——R 和"已经离开的 R1"实际上**共享了同一份 KV 数据**,零拷贝。

### 5.6 同时:为 100 个新 token 分配新物理 slot

```python
# token_to_kv_pool_allocator.alloc(100)
new_slots = tensor([2000, 2001, ..., 2099])    # 100 个新 slot
req_to_token_pool.req_to_token[5][1024:1124] = new_slots
```

`req.fill_ids` 也被设好:`fill_ids = origin_input_ids + output_ids = [A,B,C,...,X1, Y1,...,Y100]`,长度 1124。

### 5.7 关键变量快照(T2)

```
req.req_pool_idx        = 5
req.prefix_indices      = tensor([100, 101, ..., 1123])     ← 1024 个
req.last_node           = node_X
req.fill_ids            = [A,B,...,X1, Y1,...,Y100]          ← 长度 1124
req.extend_input_len    = 100                                 ← 1124 - 1024
req.cache_protected_len = 1024

req_to_token_pool.req_to_token[5]:
  [0:1024]   = [100, 101, ..., 1123]      ← 命中前缀的 slot(共享 node_X)
  [1024:1124] = [2000, 2001, ..., 2099]   ← 新分配的 100 个 slot

RadixCache:
  node_X.lock_ref       = 1               ← 已被 R 引用
  evictable_size_       = 0
  protected_size_       = 1024
```

---

## 六 阶段 T3:`new_batch.prepare_for_extend` 拼 ForwardBatch

**入口**:`ScheduleBatch.prepare_for_extend()`(`schedule_batch.py:1577+`)

把 batch 里所有请求的 token、KV 索引、长度信息拼成一组扁平张量,送进 worker。重点关注 R 的贡献:

```python
input_ids = [r.fill_ids[len(r.prefix_indices):] for r in reqs]
# 对 R:fill_ids[1024:1124] = [Y1, Y2, ..., Y100]  长度 100
# ★ 注意只取了 prefix 之后的部分——前缀对应的 Q 不用算

seq_lens = [len(r.fill_ids) for r in reqs]          # R: 1124(总长度)
prefix_lens = [len(r.prefix_indices) for r in reqs]  # R: 1024(已命中前缀长度)
extend_lens = [r.extend_input_len for r in reqs]     # R: 100(本批需算的)

# 写到 ModelWorkerBatch / ForwardBatch
forward_batch.extend_prefix_lens = tensor([1024, ...])  # ★ 关键字段!
forward_batch.seq_lens             = tensor([1124, ...])
forward_batch.req_pool_indices     = tensor([5, ...])
forward_batch.out_cache_loc        = tensor([2000, 2001, ..., 2099, ...])
```

### 关键字段含义对照表

| 字段 | R 的值 | 含义 |
|---|---|---|
| `input_ids` | `[Y1, ..., Y100]` | **只装本批新算的 100 个 token**(prefix 部分不算 Q) |
| `seq_lens` | `1124` | 总历史长度(prefix + new) |
| `prefix_lens` / `extend_prefix_lens` | `1024` | **告诉 attention kernel:"前 1024 token 在 cache 里,直接读"** |
| `extend_lens` / `extend_seq_lens` | `100` | 本批新 token 数 |
| `req_pool_indices` | `5` | req_to_token 表的行号 |
| `out_cache_loc` | `[2000..2099]` | 100 个新 token 的物理 KV slot 编号(写入位置) |

**`extend_prefix_lens` 就是父篇 [`forward_extend`](flashinfer-forward-extend.zh.md) §八 反复提到的 "kernel 怎么知道 prefix 有多长" 的来源**。

### 关键变量快照(T3)

```
ForwardBatch:
  forward_mode           = EXTEND
  input_ids              = [..., Y1, Y2, ..., Y100, ...]    (本批所有 req 拼起来)
  req_pool_indices       = [..., 5, ...]
  seq_lens               = [..., 1124, ...]
  extend_prefix_lens     = [..., 1024, ...]
  extend_seq_lens        = [..., 100, ...]
  out_cache_loc          = [..., 2000, ..., 2099, ...]
```

RadixCache 树形**不变**;`node_X.lock_ref` 仍然是 1。

---

## 七 阶段 T4:worker / attention kernel 实际执行

**入口**:`forward_batch_generation` → `model_runner.forward` → `LlamaDecoderLayer.forward` → `LlamaAttention.forward` → `RadixAttention.forward` → `FlashInferAttnBackend.forward_extend`(详见 [forward 详解](flashinfer-forward-extend.zh.md))。

按典型路径(use_ragged=True + extend_no_prefix=False)走分支 C:

```python
# (1) 对本批 100 个新 token 跑 ragged attention
o1, s1 = prefill_wrapper_ragged.forward_return_lse(q, k, v, causal=True, ...)
# q.shape = [100, 32, 128],对本批 100 个 query
# k, v 是本批新算的 K、V(还没写进 cache)

# (2) 对 1024 个 prefix 跑 paged attention
o2, s2 = prefill_wrapper_paged.forward_return_lse(
    q,
    forward_batch.token_to_kv_pool.get_kv_buffer(layer_id),  # 拿到 [size+page_size, head_num, head_dim] 全局 buffer
    causal=False, ...
)
# paged wrapper 从 plan 阶段就知道:本 req 的 KV 跨度 = seq_lens[i] = 1124,占用 page = ...
# wrapper 通过 req_to_token_pool.req_to_token[5][0:1024] 找到那 1024 个物理 slot
# 这些 slot 早就装着 node_X 的 KV 数据,直接读

# (3) LSE 合并
o, _ = merge_state(o1, s1, o2, s2)

# (4) 写新 K, V 到 cache
forward_batch.token_to_kv_pool.set_kv_buffer(layer, out_cache_loc, k, v, ...)
# 即:k_buffer[layer_id][2000..2099] = k,同时 v_buffer 同样写入
```

### 关键观察:此时 KV cache 池里的物理布局

```
k_buffer[layer_id]:
  ...
  slot 100   ← node_X 的第 0 个 token,R 通过 req_to_token 共享读
  slot 101   ← node_X 的第 1 个 token
  ...
  slot 1123  ← node_X 的第 1023 个 token
  ...
  slot 2000  ← R 新算的第 0 个 token (Y1)
  slot 2001  ← R 新算的第 1 个 token (Y2)
  ...
  slot 2099  ← R 新算的第 99 个 token (Y100)
```

**R 跨越了不连续的 slot 区间 [100..1123] + [2000..2099]**——这就是 paged KV cache 的灵活性。

### RadixCache 树形:**仍然不变!**

虽然新 K、V 已经被算出来并写进 cache 池,**树里还没出现 [Y1..Y100] 这段路径**——这要等 R 跑完 prefill 之后才插入。

### 关键变量快照(T4 末尾)

```
req.req_pool_idx        = 5
req.prefix_indices      = tensor([100..1123])         ← 还是这个,没变
req.fill_ids            = [A..X1, Y1..Y100]
req.kv_committed_len    = 1124  (此处可能更新,取决于实现)

req_to_token_pool.req_to_token[5]:
  [0:1124] = [100..1123, 2000..2099]   ← 完整映射

k_buffer / v_buffer:
  slot 2000..2099 = R 算出的新 K/V

RadixCache:
  node_X.lock_ref       = 1            ← 还没变
  evictable_size_       = 0
  protected_size_       = 1024
```

---

## 八 阶段 T5:prefill 完成 → `cache_unfinished_req` 把新 token 挂到树里

**入口**:`Scheduler.process_batch_result_prefill()` → `scheduler_output_processor_mixin.py:200` 调 `tree_cache.cache_unfinished_req(req)`。

```python
def cache_unfinished_req(self, req: Req, chunked=False):
    token_ids = req.fill_ids                                  # [A..X1, Y1..Y100]
    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]
    # kv_indices = [100..1123, 2000..2099]

    radix_key = RadixKey(token_ids).page_aligned(self.page_size)
    values = kv_indices[: len(radix_key)].to(dtype=torch.int64, copy=True)

    # ★ insert——但已经存在的部分不会重复创建,会沿树往下走
    result = self.insert(InsertParams(key=radix_key, value=values, ...))
    new_prefix_len = result.prefix_len    # 1024(已存在的部分 = node_X)

    # 释放重复的物理 slot(node_X 已经持有了 [100..1123],不需要再占一份)
    self.token_to_kv_pool_allocator.free(
        kv_indices[req.cache_protected_len : new_prefix_len]
    )
    # cache_protected_len 之前已经标记为受保护,这里 free 的是中间产生的重复

    # 重新 match_prefix,拿到包含新插入节点的完整 indices
    match_result = self.match_prefix(MatchPrefixParams(key=radix_key))
    new_indices, new_last_node = match_result.device_indices, match_result.last_device_node

    # 把 req_to_token 映射表换成「树里的物理 slot」(因为新节点 [Y1..Y100] 已经被树拥有)
    self.req_to_token_pool.write(
        (req.req_pool_idx, slice(req.cache_protected_len, len(new_indices))),
        new_indices[req.cache_protected_len :],
    )

    req.cache_protected_len = len(new_indices)
    self.dec_lock_ref(req.last_node)     # 释放 node_X 上的 lock(从 inc 到 root)
    self.inc_lock_ref(new_last_node)     # 锁定新的 last_node(包含 Y1..Y100 的那个新节点)
    req.last_node = new_last_node        # ★ last_node 切换!
    req.prefix_indices = new_indices
```

### `insert` 内部发生的事

`_insert_helper` 沿树往下走:
1. 从 root 走到 `node_X`,**前 1024 token 完全匹配**(`prefix_len == len(node_X.key)`),跳过
2. 剩余 100 个 token(Y1..Y100)在 `node_X.children` 里没匹配
3. **创建新叶子节点 `node_Y`**,挂在 `node_X` 下:

```
       root
         │
         └── node_X (lock_ref=1)        ← 旧 lock
              │
              └── node_Y (lock_ref=0)    ← ★ 新建
                  key=[Y1,...,Y100]      (但 page_aligned 可能只取前 96 个)
                  value=[2000..2095]      ← 96 个 slot,page_size=16 对齐
                  lock_ref=0
                  parent=node_X
```

### lock_ref 切换

```python
self.dec_lock_ref(req.last_node)    # 旧 last_node = node_X,dec
self.inc_lock_ref(new_last_node)    # 新 last_node = node_Y,inc
```

**net 效果**:`node_X.lock_ref` 不变(从 1 dec 到 0,又因为 node_Y 的 inc 走到 root 路径上经过 node_X,inc 回 1)。**`node_Y.lock_ref` = 1**。

### 关键变量快照(T5)

```
req.last_node           = node_Y       ← ★ 切换
req.prefix_indices      = tensor([100..1123, 2000..2095])  ← 1024 + 96 = 1120
req.cache_protected_len = 1120

RadixCache:
  node_X.lock_ref       = 1            ← 还在(R 引用着 node_Y,而 node_Y 是 node_X 的子)
  node_Y.lock_ref       = 1            ← 新增
  evictable_size_       = 0
  protected_size_       = 1120
```

> **page_size=16 引起的「尾巴」**:1124 不是 16 的倍数,page_aligned 后变 1120。剩下的 4 个 slot([2096..2099])**暂时还在 req_to_token 表里(`req.prefix_indices` 含它们),但没挂到树里**——下次 cache_unfinished_req 或 cache_finished_req 时再处理。

---

## 九 阶段 T6:decode 阶段 — 每步分配一个 slot 但 RadixCache 不变

R 进入 decode 阶段,每轮生成 1 个 token,前后 50 个 token(`max_new_tokens=50`)。

每一轮 decode 时:
1. `token_to_kv_pool_allocator.alloc(1)` 拿到一个新 slot,例 `2200`
2. `req_to_token_pool.req_to_token[5][1124] = 2200`
3. attention kernel 读 cache + 写新 K/V 到 slot 2200
4. **RadixCache 树形不变**——decode 期间不往树里 insert(每 token 都插太贵)

每次扩张 `req.fill_ids` 和 `req_to_token_pool.req_to_token[5]` 的有效长度:

```
T6 第 1 步:fill_ids 长 1125,req_to_token[5][1124] = 2200
T6 第 2 步:fill_ids 长 1126,req_to_token[5][1125] = 2201
...
T6 第 50 步:fill_ids 长 1174,req_to_token[5][1173] = 2249
```

50 轮后 R 生成完毕(碰到 EOS 或达到 max_new_tokens)。

### 关键变量快照(T6 末尾)

```
req.fill_ids            = [A..X1, Y1..Y100, Z1..Z50]   长 1174
req.output_ids          = [Z1..Z50]
req.kv_committed_len    = 1174

req_to_token_pool.req_to_token[5][0:1174] = [100..1123, 2000..2099, 2200..2249]

RadixCache:
  node_X.lock_ref       = 1   (仍引用 node_Y)
  node_Y.lock_ref       = 1
```

---

## 十 阶段 T7:请求结束 → `cache_finished_req`

**入口**:`Scheduler` 检测到 R 结束(EOS / 长度上限),调 `release_kv_cache`(`mem_cache/common.py:479`)→ `tree_cache.cache_finished_req(req)`。

```python
def cache_finished_req(self, req: Req, is_insert=True):
    kv_committed_len = req.pop_committed_kv_cache()    # 1174

    token_ids = (req.origin_input_ids + req.output_ids)[:kv_committed_len]
    # token_ids = [A..X1, Y1..Y100, Z1..Z50]  长 1174

    kv_indices = self.req_to_token_pool.req_to_token[
        req.req_pool_idx, : len(token_ids)
    ]
    # [100..1123, 2000..2099, 2200..2249]

    radix_key = RadixKey(token_ids).page_aligned(self.page_size)
    # page_size=16,1174 向下对齐到 1168
    # radix_key 长度 = 1168

    values = kv_indices[: 1168]    # 取前 1168 个 slot

    # ★ insert
    result = self.insert(InsertParams(key=radix_key, value=values, ...))
    new_prefix_len = result.prefix_len    # 1120(已存在 node_X + node_Y)

    # 释放重复的物理 slot
    self.token_to_kv_pool_allocator.free(
        kv_indices[req.cache_protected_len : new_prefix_len]
    )

    # 释放未对齐的尾部(1168 之后那几个 slot)
    self.token_to_kv_pool_allocator.free(kv_indices[1168:])

    # 释放 req 在树上的锁
    self.dec_lock_ref(req.last_node)
```

### `insert` 内部:在 node_Y 下挂一个新子节点

`_insert_helper` 沿树走:
1. root → node_X(完全匹配前 1024)
2. node_X → node_Y(完全匹配 1024..1120)
3. 剩余 [Z1..Z48] 48 个 token 没匹配(48 是 1168-1120),作为新叶子挂上:

```
       root
         │
         └── node_X (lock_ref=2)
              │
              └── node_Y (lock_ref=2)
                   │
                   └── node_Z (lock_ref=0)   ← ★ 新建
                       key=[Z1..Z48]
                       value=[2200..2247]
                       lock_ref=0
```

### `dec_lock_ref` 释放 R 的引用

从 `req.last_node = node_Y` 一路 dec 到 root,**每个节点的 lock_ref 减 1**:

```
node_Y.lock_ref:  2 → 1   (但 R 是唯一引用者,所以是 1 → 0)
node_X.lock_ref:  2 → 1   (R 释放,但 R 此前在 node_Y 的链上又 inc 过一次,所以净效果)
```

> 实际上 R 只占了「node_Y 这一条 inc 链」上的引用,dec 后 node_Y 和 node_X 都从 1 变 0(因为 R 是唯一引用者)。

### 关键变量快照(T7)

```
req.req_pool_idx        = None       (被释放,通过 ReqToTokenPool.free 还回)

RadixCache:
  node_X (lock_ref=0)    ← R 走了,无人引用
   │
   ├── node_Y (lock_ref=0)
   │    │
   │    └── node_Z (lock_ref=0)
   │
   └── (其他可能的兄弟节点)

  evictable_size_       = 1024 + 96 + 48 = 1168
  protected_size_       = 0
```

**R 留下的全部 KV 数据(包括 prompt 部分和生成部分)都进了 RadixCache,变成可复用的前缀**。如果下一个请求的 prompt 是 `[A..X1, Y1..Y100, Z1..Z48, 别的]`,会**完美命中 1168 个 token 的前缀**,只需算新增部分。

---

## 十一 全程时间线总览

```
时间     │  RadixCache 树形                              │  关键变量
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T0 入队   │  root─node_X(lock=0)                          │  req: 几乎全空
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T1 match  │  无变化                                       │  prefix_indices=node_X.value
         │                                              │  last_node=node_X
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T2 admit  │  root─node_X(lock=1)  ← inc                  │  req_pool_idx=5
         │                                              │  分配新 slot [2000..2099]
         │                                              │  req_to_token[5][0:1124] 填好
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T3 batch  │  无变化                                       │  forward_batch.extend_prefix_lens=[1024]
         │                                              │  out_cache_loc=[2000..2099]
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T4 fwd    │  无变化                                       │  k_buffer[2000..2099]=new K
         │                                              │  v_buffer[2000..2099]=new V
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T5 cache_ │  root─node_X(lock=1)                          │  last_node=node_Y(切换)
unfinish │      └─node_Y(lock=1)  ← 新建并 inc           │  prefix_indices 更新
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T6 decode │  无变化                                       │  每轮 alloc 1 个新 slot
×50      │                                              │  fill_ids 增长
─────────┼──────────────────────────────────────────────┼─────────────────────────────
T7 finish │  root─node_X(lock=0)                          │  req 释放
         │      └─node_Y(lock=0)                         │  evictable_size_ += 1168
         │           └─node_Z(lock=0)  ← 新建,叶子        │
```

---

## 十二 关键变量的传递路径(回答用户原问题)

> **`forward_batch.extend_prefix_lens` 看起来是一个关键成员,它记录了一个 req 的前缀有多长**

正是。它的来源传递链:

```
Req.origin_input_ids + output_ids                                       (用户输入)
        │
        ↓  tree_cache.match_prefix()                                     T1 阶段
Req.prefix_indices = tensor([slot_0, slot_1, ..., slot_{prefix_len-1}])  (物理 slot 索引)
Req.last_node                                                            (树里的位置)
        │
        ↓  ScheduleBatch.prepare_for_extend()                            T3 阶段
[len(r.prefix_indices) for r in reqs]
        │
        ↓
ScheduleBatch.prefix_lens
        │
        ↓  schedule_batch.py:2438
ScheduleBatch.extend_prefix_lens
        │
        ↓  ModelWorkerBatch 转 ForwardBatch                              进入 worker
forward_batch.extend_prefix_lens
        │
        ↓  FlashInferAttnBackend.init_forward_metadata                   T4 阶段开始
indices_updater_prefill.update(..., prefix_lens=...)
        │
        ↓
prefill_wrapper.plan(...) / begin_forward(...)
        │
        ↓  CUDA kernel 读 paged_kv_indices 找到 prefix 物理 slot
        ↓  q 对 prefix K/V 跑 attention
```

---

## 十三 一句话总结

> **一个请求的完整生命周期是 7 个阶段**:T0 入队 → T1 `match_prefix` 拿到 `prefix_indices` 和 `last_node` → T2 `inc_lock_ref` 锁住前缀 + 分配 `req_pool_idx` + alloc 新 slot → T3 `prepare_for_extend` 算出 `extend_prefix_lens` 等张量 → T4 attention kernel 跑「split-K」并写新 K/V 到 cache 池 → T5 `cache_unfinished_req` 把新 token 挂回树(创建新节点,切换 `last_node`) → T6 decode 阶段每步 alloc 1 个新 slot 但**树不变** → T7 `cache_finished_req` 插入最终段并 `dec_lock_ref` 释放,**留下的 KV 全部成为可复用前缀**。`RadixCache` 的树形变化主要发生在 T2(分配)、T5(`cache_unfinished_req` 插入)、T7(`cache_finished_req` 插入)三个时刻,其他阶段只是 `lock_ref` 计数变化或物理 slot 池的 alloc/free。
