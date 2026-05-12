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

下面我们逐阶段看 RadixCache 和关键变量怎么变。

---

## 二 阶段 T0:请求进入 scheduler

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

## 三 阶段 T1:scheduler 取出请求,做 `match_prefix`

**入口**:`PrefillAdder.compute_prefix_matches()`(`schedule_policy.py:185`),scheduler 每轮主循环组 batch 之前会批量做。

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

## 四 阶段 T2:`PrefillAdder.add_one_req` admit + 锁住 node + 分配新 slot

**入口**:`PrefillAdder.add_one_req()`(`schedule_policy.py:780+`)

```python
# 计算 extend_input_len = 总长度 - 已命中前缀长度
prefix_len = len(req.prefix_indices)       # 1024
real_input_tokens = req.extend_input_len - req.host_hit_length  # 1124-1024 = 100

# 关键:进入临界区,锁住 node_X
with self._lock_node(req.last_node):       # ★ inc_lock_ref(node_X)
    # 此时 node_X.lock_ref 至少为 1,不会被 evict

    # 检查 KV pool 容量够不够
    if total_tokens >= self.rem_total_tokens:
        return AddReqResult.NO_TOKEN

    # 给请求分配 req_pool_idx
    self.can_run_list.append(req)
    self._req_inc_lock_ref(req)             # ★ 第二次 inc(为了请求长期持有)
```

### `inc_lock_ref` 触发的链路(`radix_cache.py:637`):

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

### 接下来:`req.req_pool_idx` 的分配

`PrefillAdder` 把 req 加入 `can_run_list` 后,主循环里 `Scheduler.get_new_batch_prefill()` 调用 `ReqToTokenPool.alloc()`:

```python
# memory_pool.py: ReqToTokenPool.alloc
def alloc(self, reqs):
    need_size = len(reqs) - len(reusing)
    select_index = self.free_slots[:need_size]
    self.free_slots = self.free_slots[need_size:]
    for r in reqs:
        if r.req_pool_idx is None:
            r.req_pool_idx = select_index[offset]    # ★ 分配 req_pool_idx
            offset += 1
```

这一步 `req.req_pool_idx` 被赋值(假设拿到 5),`req_to_token_pool.req_to_token[5]` 是一行长度 `max_context_len` 的 int 数组,**这一行就是「请求 R 用了哪些物理 KV slot」的映射表**。

### 紧接着:把命中的前缀 slot 写进映射表

`prepare_for_extend` 阶段(`schedule_batch.py:1600+`),把 `req.prefix_indices` 拷到 `req_to_token_pool.req_to_token[req_pool_idx]` 的前 1024 行:

```python
# 简化伪代码
req_to_token_pool.req_to_token[5][0:1024] = req.prefix_indices  # 1024 个 slot 编号
```

**这一刻 R 的 req_to_token 表前 1024 个槽位指向 node_X 的物理 slot**——R 和"已经离开的 R1"实际上**共享了同一份 KV 数据**,零拷贝。

### 同时:为 100 个新 token 分配新物理 slot

```python
# token_to_kv_pool_allocator.alloc(100)
new_slots = tensor([2000, 2001, ..., 2099])    # 100 个新 slot
req_to_token_pool.req_to_token[5][1024:1124] = new_slots
```

`req.fill_ids` 也被设好:`fill_ids = origin_input_ids + output_ids = [A,B,C,...,X1, Y1,...,Y100]`,长度 1124。

### 关键变量快照(T2)

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

## 五 阶段 T3:`new_batch.prepare_for_extend` 拼 ForwardBatch

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

## 六 阶段 T4:worker / attention kernel 实际执行

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

## 七 阶段 T5:prefill 完成 → `cache_unfinished_req` 把新 token 挂到树里

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

## 八 阶段 T6:decode 阶段 — 每步分配一个 slot 但 RadixCache 不变

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

## 九 阶段 T7:请求结束 → `cache_finished_req`

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

## 十 全程时间线总览

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

## 十一 关键变量的传递路径(回答用户原问题)

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

## 十二 一句话总结

> **一个请求的完整生命周期是 7 个阶段**:T0 入队 → T1 `match_prefix` 拿到 `prefix_indices` 和 `last_node` → T2 `inc_lock_ref` 锁住前缀 + 分配 `req_pool_idx` + alloc 新 slot → T3 `prepare_for_extend` 算出 `extend_prefix_lens` 等张量 → T4 attention kernel 跑「split-K」并写新 K/V 到 cache 池 → T5 `cache_unfinished_req` 把新 token 挂回树(创建新节点,切换 `last_node`) → T6 decode 阶段每步 alloc 1 个新 slot 但**树不变** → T7 `cache_finished_req` 插入最终段并 `dec_lock_ref` 释放,**留下的 KV 全部成为可复用前缀**。`RadixCache` 的树形变化主要发生在 T2(分配)、T5(`cache_unfinished_req` 插入)、T7(`cache_finished_req` 插入)三个时刻,其他阶段只是 `lock_ref` 计数变化或物理 slot 池的 alloc/free。
