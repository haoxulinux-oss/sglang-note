# RadixCache 数据结构详解(初学者向)

> 📖 **关联阅读**:
> - 配套篇:[一个请求在 RadixCache 视角下的完整生命周期](request-lifecycle-radix-cache.zh.md) — 看完本文再去看那一篇
> - 父篇:[`FlashInferAttnBackend.forward_extend()` 详解](flashinfer-forward-extend.zh.md) §八 — 解释了「prefix 不会被复制进 q/k/v,一直在 KV cache 池里」
> - 横向:[KV cache 存什么、HBM 是什么](kvcache-prefetch-and-storage.zh.md)

代码位置:`python/sglang/srt/mem_cache/radix_cache.py`

---

## 一 一句话定位

**RadixCache 不是一个 KV 存储池,而是一棵「前缀树索引」**——它的节点不存 K、V 数据本身,而是存「这一段 token 对应到 `token_to_kv_pool` 的哪些物理 slot」的指针(`value` 字段是 int64 索引张量)。**真正的 K、V 数据在 `MHATokenToKVPool`(`memory_pool.py:764`,见父篇 §四)**。

把 RadixCache 想象成图书馆的「**索引卡片柜**」,卡片记录"`AAA + BBB` 这段 token 的 KV 存在 K 池的第 [102, 103, ..., 250] 号 slot",但真正的内容不在卡片柜里。

---

## 二 RadixCache 的两个核心类

### 2.1 `TreeNode`(`radix_cache.py:211`)——树节点

```python
class TreeNode:
    def __init__(self, id=None, priority=0):
        self.children = defaultdict(TreeNode)    # 子节点 dict,key 是 child 的「首页」token id
        self.parent: TreeNode = None             # 父节点
        self.key: RadixKey = None                # 本节点存的 token id 序列(变长)
        self.value: Optional[torch.Tensor] = None # ★ 这段 token 对应的 KV 物理 slot 编号
        self.lock_ref = 0                        # ★ 引用计数:>0 表示被请求引用着,不能淘汰
        self.last_access_time = time.monotonic() # LRU 用的时间戳
        self.hit_count = 0                       # 这个节点被命中多少次
        self.host_value: Optional[torch.Tensor] = None  # HiCache:host RAM 上的副本索引
        self.hash_value: Optional[List[str]] = None     # 持久化用的 hash 标识
        self.priority = priority                 # 优先级感知淘汰用
        self.id = TreeNode.counter               # 唯一 ID(调试用)
```

**关键字段速查表**:

| 字段 | 含义 | 例 |
|---|---|---|
| `key` | 本节点持有的 token id 序列(`RadixKey` 包装) | `[7000, 9001, 12, 305, ...]`(可能上千个) |
| `value` | 这段 token 在 `token_to_kv_pool` 里的物理 slot 编号 | `tensor([102, 103, 104, ..., 250])` |
| `children` | 子节点字典,key 是「子节点首页」的特征 | `{(101, 305): child_a, (101, 99): child_b}` |
| `lock_ref` | 被多少请求占用——`>0` 不能 evict | `2` 表示两个请求正在用这条路径 |
| `last_access_time` | LRU 排序键 | `1730000000.123` |
| `parent` | 父节点 | (除 root 外不为 None) |

### 2.2 `RadixKey`(`radix_cache.py:71`)——token 序列的包装

为什么不直接用 list/tensor?——因为 RadixCache 需要支持:
- **`extra_key`** namespace(LoRA / 采样盐值 / 缓存版本):同样的 token 序列但 `extra_key` 不同 → 视作不同节点(隔离)
- **eagle bigram view**(投机解码用):需要把相邻 token 配对成 bigram 再算 hash
- **页对齐切割**(`page_size > 1` 时切到整页边界,避免半页节点)

`RadixKey` 提供:
- `__len__` / `__iter__` / `__getitem__` 等列表操作
- `.match(other, page_size)` — 计算两个 key 的最长公共前缀长度
- `.child_key(page_size)` — 取首页作为子节点 dict 索引
- `.page_aligned(page_size)` — 长度向下对齐到页边界
- `.hash_page(start, end, prior_hash)` — 算页 hash(HiCache 用)

> 初学者**只需要把 RadixKey 当成 "带额外属性的 token id 列表"**,不用纠结 bigram view、extra_key 这些细节。

---

## 三 整棵树长什么样(以一个例子直观感受)

假设系统刚启动,空树:

```
                      ┌─────────────────┐
                      │  root_node       │
                      │  key=[]          │
                      │  value=[]        │
                      │  lock_ref=1      │  ← root 永远 lock_ref=1,从不被淘汰
                      │  children={}     │
                      └─────────────────┘
```

**T1:第一个请求 R1 进来**,prompt = `"You are a helpful assistant. <user>: 介绍下罗马"`,假设 tokenize 后 = `[A,B,C,D,E,F,G,H,I,J]`(共 10 个 token)。R1 跑完 prefill 时:

```
              root (lock_ref=1, key=[])
                 │
                 │ children[(A,)]
                 ↓
              ┌───────────────────────┐
              │ node_1                 │
              │ key=[A,B,C,D,E,F,G,H,I,J] │
              │ value=[102,103,...,111]   │  ← 10 个物理 KV slot
              │ lock_ref=1 (R1 引用着)   │
              │ children={}              │
              └───────────────────────┘
```

**T2:第二个请求 R2 进来**,prompt = `"You are a helpful assistant. <user>: 介绍下中国"`,tokenize = `[A,B,C,D,E,F,G,H,I,K,L]`(前 9 个和 R1 一样,后面 `K,L` 不同)。**关键:R2 来时 scheduler 调 `match_prefix`,会发生节点分裂**——`node_1` 在「前 9 个 token 后」被切成两段:

```
              root
                 │
                 ↓
              ┌───────────────────┐
              │ node_1A (新建)     │
              │ key=[A,B,C,D,E,F,G,H,I] │  ← 共享前缀
              │ value=[102,...,110]      │
              │ lock_ref=2 (R1 + R2 都引用) │
              └───────────────────┘
                 │              │
                 │              │
       children[(J,)]    children[(K,L)]
                 │              │
                 ↓              ↓
        ┌─────────────┐  ┌────────────┐
        │ node_1B      │  │ node_2      │
        │ key=[J]      │  │ key=[K,L]   │
        │ value=[111]  │  │ value=[120,121] │
        │ lock_ref=1   │  │ lock_ref=1  │
        │ (R1)         │  │ (R2)        │
        └─────────────┘  └────────────┘
```

**T3:R1 和 R2 都结束**,scheduler 调 `cache_finished_req` → `dec_lock_ref`:

```
              root  (lock_ref=1)
                 │
                 ↓
              node_1A (lock_ref=0)  ← 可被 evict 了
                 │           │
                 ↓           ↓
            node_1B          node_2
            (lock_ref=0)     (lock_ref=0)
```

**T4:第三个请求 R3 进来**,prompt 前缀和 R1 完全相同 `[A,B,C,D,E,F,G,H,I,J]`,**直接命中,零拷贝复用**:

```
              root  (lock_ref=1)
                 │
                 ↓
              node_1A (lock_ref=1)  ← R3 占用
                 │           │
                 ↓           ↓
            node_1B          node_2
            (lock_ref=1)     (lock_ref=0)
              (R3)
```

R3 的 `prefix_indices = [102,103,...,110, 111]`,**省掉 10 个 token 的 prefill 计算**。

---

## 四 关键算法:`match_prefix`(`radix_cache.py:398`)

输入:`RadixKey(token_ids=[...])`
输出:`MatchResult(device_indices, last_device_node, ...)`

**算法骨架**(`_match_prefix_helper`,`:693`):

```python
def _match_prefix_helper(self, node: TreeNode, key: RadixKey):
    value = []
    child_key = key.child_key(self.page_size)   # 取 key 首页作为 dict 索引

    while len(key) > 0 and child_key in node.children:
        child = node.children[child_key]
        prefix_len = child.key.match(key, page_size=self.page_size)   # 算和 child.key 的公共前缀长度

        if prefix_len < len(child.key):
            # ★ 部分命中:把 child 在 prefix_len 处切开
            new_node = self._split_node(child.key, child, prefix_len)
            value.append(new_node.value)
            node = new_node
            break
        else:
            # ★ 完全命中 child,继续往下
            value.append(child.value)
            node = child
            key = key[prefix_len:]              # 剥掉已匹配的部分
            child_key = key.child_key(self.page_size) if len(key) else None

    return value, node
```

**两种情况**:
1. **完全匹配某个 child,继续往下**:把 child 整个吃掉,继续匹配下一层
2. **只匹配 child 的前几个 token**:**调 `_split_node` 把 child 切开**——前缀部分变成新节点 `new_node`(R1 + R2 共享),剩余部分挂在 `new_node` 下

返回 `value` 是匹配过的所有节点的 `value` 列表,最后 `torch.cat(value)` 拼成完整的 KV 物理 slot 索引张量。

### 4.1 `_split_node` 详解(关键操作)

`_split_node`(`:719`)就是上面 T2 例子里把 `node_1` 切成 `node_1A` 和 `node_1B` 的实现:

```python
def _split_node(self, key, child, split_len):
    new_node = TreeNode(priority=child.priority)
    new_node.children = {key[split_len:].child_key(...): child}    # ★ child 变成 new_node 的子节点
    new_node.parent = child.parent
    new_node.lock_ref = child.lock_ref                              # ★ 继承 child 的 lock_ref
    new_node.key = child.key[:split_len]                            # 前 split_len 个 token
    new_node.value = child.value[:split_len].clone()                # 对应物理 slot 前半段

    child.parent = new_node                                         # ★ 改 child 的 parent 指针
    child.key = child.key[split_len:]                               # 剩余 token
    child.value = child.value[split_len:].clone()                   # 对应物理 slot 后半段

    new_node.parent.children[key.child_key(...)] = new_node         # 替换 parent → child 的连接
    return new_node
```

**关键不变量**:
- `child` 不被删除,只是变小
- `new_node` 接管 child 原来在父节点 children dict 里的位置
- `lock_ref` 直接继承——split 不改变"被引用的事实"

---

## 五 关键算法:`insert`(`radix_cache.py:468`)

请求结束时调用,把请求积累的 token / KV slot 写回树。

**算法骨架**(`_insert_helper`,`:749`):

```python
def _insert_helper(self, node, key, value, priority=0, chunked=False):
    total_prefix_length = 0
    child_key = key.child_key(self.page_size)

    while len(key) > 0 and child_key in node.children:
        node = node.children[child_key]
        prefix_len = node.key.match(key, page_size=self.page_size)
        total_prefix_length += prefix_len
        key = key[prefix_len:]                        # 剥掉重叠部分
        value = value[prefix_len:]                    # 同步剥掉

        if prefix_len < len(node.key):
            # 已有节点和新 key 只部分重合,切开
            new_node = self._split_node(node.key, node, prefix_len)
            node = new_node
        # else:完全重合,继续往下

    if len(key):
        # ★ 剩下 key 在树里没匹配,作为新叶子挂上
        new_node = TreeNode(priority=priority)
        new_node.parent = node
        new_node.key = key
        new_node.value = value.clone()
        node.children[child_key] = new_node
        self.evictable_size_ += len(key)

    return total_prefix_length
```

**两个动作**:
1. 沿树往下走,遇到部分重叠就 split,跳过已存在的 token
2. 剩下的 token 作为新叶子节点挂上去

返回 `total_prefix_length`(已存在的 token 数)——**caller 用这个数字知道哪些 slot 是和 cache 重复的、可以释放**。

---

## 六 lock_ref:防止"正在用的节点被淘汰"

### 6.1 引用计数语义

```python
def inc_lock_ref(self, node):
    while node != self.root_node:
        if node.lock_ref == 0:
            self.evictable_size_ -= len(node.key)
            self.protected_size_ += len(node.key)
        node.lock_ref += 1
        node = node.parent
```

**从该节点一路 inc 到 root**——保证整条路径上的祖先都不会被 evict。

- `lock_ref == 0`:节点没人用,**可被淘汰**(`evictable`)
- `lock_ref > 0`:节点有请求在用,**受保护**(`protected`)

`dec_lock_ref` 是对称操作。

### 6.2 evictable_size_ vs protected_size_

两个全局计数器:

```
total_kv_in_tree   = evictable_size_ + protected_size_
```

- `evictable_size_`:树里所有 `lock_ref == 0` 节点 key 的总长度——**可释放的 KV 量**
- `protected_size_`:树里所有 `lock_ref > 0` 节点 key 的总长度——**正在被请求引用的 KV 量,绝对不能动**

当 `token_to_kv_pool_allocator` 显存告急,scheduler 会调 `tree_cache.evict(num_tokens)` 释放 `evictable_size_` 里的内容。

### 6.3 evictable_leaves(`radix_cache.py:362`)

`evictable_leaves` 是一个 set,**存所有「lock_ref==0 且没有任何 lock_ref>0 后代」的叶子节点**——这是淘汰的候选名单。`evict` 从这里取 leaf,按 LRU/LFU/FIFO 等策略选一个释放。释放后如果父节点也变叶子且可淘汰,加入候选名单(`evict` 函数里的 `heapq.heappush`)。

> 注意:**只能淘汰叶子**——不能从树中间挖一个节点出来,否则破坏树结构和共享性。

---

## 七 evict 算法(`radix_cache.py:608`)

```python
def evict(self, params):
    num_tokens = params.num_tokens
    leaves = list(self.evictable_leaves)
    eviction_heap = [
        (self.eviction_strategy.get_priority(node), node) for node in leaves
    ]
    heapq.heapify(eviction_heap)

    num_evicted = 0
    while num_evicted < num_tokens and len(eviction_heap):
        _priority, x = heapq.heappop(eviction_heap)
        # ★ 真正释放 KV cache 池里的物理 slot
        self.token_to_kv_pool_allocator.free(x.value)
        num_evicted += len(x.value)
        self._delete_leaf(x)

        # 如果父节点现在变叶子且可淘汰,加入候选
        if len(x.parent.children) == 0 and x.parent.lock_ref == 0:
            heapq.heappush(eviction_heap, (..., x.parent))

    return EvictResult(num_tokens_evicted=num_evicted)
```

**两步**:
1. `token_to_kv_pool_allocator.free(x.value)` —— 把物理 slot 还回空闲池
2. `_delete_leaf(x)` —— 从树里摘掉这个叶子节点

支持的淘汰策略(`__init__` 里根据 `eviction_policy` 选):**LRU / LFU / FIFO / MRU / FILO / Priority / SLRU**。默认 LRU。

---

## 八 RadixCache 和其他组件的关系图

```
                  ┌─────────────────────────────────────────┐
                  │                Scheduler                  │
                  │                                           │
                  │   waiting_queue: [req1, req2, ...]        │
                  │   running_batch: [req3, req4, ...]        │
                  └─────────────────────────────────────────┘
                            │       │              │
                  match_prefix    cache_unfinished_req
                            │       │              │
                            ↓       ↓              ↓
                  ┌─────────────────────────────────────────┐
                  │            RadixCache (索引层)            │
                  │                                           │
                  │   root  ──┬── node_A (lock_ref=2)          │
                  │           │      │                          │
                  │           │      ├── node_B (lock_ref=1)    │
                  │           │      └── node_C (lock_ref=0)    │
                  │           │                                 │
                  │           └── node_D (lock_ref=0)           │
                  │                                              │
                  │   .value 字段都是 int64 索引张量             │
                  │   evictable_size_, protected_size_         │
                  └─────────────────────────────────────────┘
                            │                       │
                            │ node.value(索引)       │ evict
                            ↓                       ↓ free()
                  ┌─────────────────────────────────────────┐
                  │     token_to_kv_pool_allocator           │
                  │                                           │
                  │   free_slots: [3, 8, 17, 22, ...]         │  ← 空闲物理 slot
                  │   .alloc(n) / .free(indices)              │
                  └─────────────────────────────────────────┘
                            │
                            ↓ 分配的物理 slot 索引
                  ┌─────────────────────────────────────────┐
                  │       MHATokenToKVPool (数据层)           │
                  │                                           │
                  │   k_buffer[layer]: [size+page_size, 8, 128] │
                  │   v_buffer[layer]: [size+page_size, 8, 128] │
                  │                                           │
                  │   .set_kv_buffer(layer, loc, k, v)        │  ← 写新 KV
                  │   .get_kv_buffer(layer_id) → (k_buf, v_buf) │  ← 读 cache
                  └─────────────────────────────────────────┘
                            │
                            ↓ attention kernel 读 K, V
                  ┌─────────────────────────────────────────┐
                  │     FlashInferAttnBackend.forward_extend │
                  └─────────────────────────────────────────┘
```

**三层职责清晰**:
- **RadixCache**:**「逻辑层 / 索引层」**——记录"哪段 token 对应哪些 slot",维护树形结构、lock_ref、淘汰策略
- **token_to_kv_pool_allocator**:**「分配器」**——管理空闲 slot 列表,提供 alloc / free 接口
- **MHATokenToKVPool**:**「物理层 / 数据层」**——真正持有 GPU 显存上的 K、V 张量

---

## 九 关键计数器和指标

| 指标 | 含义 |
|---|---|
| `self.evictable_size_` | 树里所有 `lock_ref==0` 节点 key 长度之和——可淘汰的 KV 量 |
| `self.protected_size_` | 树里所有 `lock_ref>0` 节点 key 长度之和——被请求占用的 KV 量 |
| `total_size()` | 树里所有节点的 key 长度之和(`evictable + protected`) |
| `token_to_kv_pool_allocator.available_size()` | 物理 slot 池剩余空闲量(**不包括树里 evictable 的部分**) |
| `available_size() + evictable_size_` | 真正能分配给新请求的容量(必要时 evict) |

---

## 十 一句话总结

> **`RadixCache` 是一棵 token id 前缀树,节点存的不是 K、V 数据本身,而是「这段 token 在 `MHATokenToKVPool` 里的物理 slot 索引张量(node.value)」**。`match_prefix` 沿树往下匹配并按需 split 节点(分裂共享前缀),`insert` 把请求结束时积累的 token+slot 挂回树,`lock_ref` 引用计数保护「正在被请求引用」的路径不被淘汰,`evict` 按 LRU 等策略从叶子开始释放 `evictable_size_` 区域。**RadixCache 本身只是「索引」,真正的 K、V 数据在 `MHATokenToKVPool` 里;两者通过 `token_to_kv_pool_allocator` 连接**。

下一篇:[一个请求在 RadixCache 视角下的完整生命周期](request-lifecycle-radix-cache.zh.md)——把这些数据结构动起来,看一个真实请求从进入 scheduler 到 attention kernel 跑完,RadixCache 的形态如何变化。
