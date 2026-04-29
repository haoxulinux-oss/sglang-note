# KV cache 存什么、HBM 是什么、`_prefetch_kvcache()` 在做什么

围绕 `_add_request_to_queue` 里那一行 `self._prefetch_kvcache(req)` 展开三个相关问题:

1. KV cache 存的具体是什么数据?是不是 K·X 和 V·X 这种向量?
2. HBM 是什么?
3. `_prefetch_kvcache()` 干了什么?这是 KV cache 存储分层(HiCache) 的关键入口。

---

## 一 KV cache 存的是什么

**结论先行**:KV cache 存的是 **每个 token 在每一层 attention 算出来的 K(Key) 向量和 V(Value) 向量**——是的,就是你猜的那种「Wₖ·X 和 Wᵥ·X」类型的张量(只是有多头 / GQA / RoPE / 量化等细节)。

### 1.1 attention 公式回顾

每层 attention 对每个 token x_i 算三个向量:

```
Q_i = W_Q · x_i      ← Query
K_i = W_K · x_i      ← Key
V_i = W_V · x_i      ← Value

attn_output_i = softmax( Q_i · K_{0..i}ᵀ / √d ) · V_{0..i}
```

注意这个公式的关键事实:**算第 i 个 token 的 attention 时,需要它**之前所有 token 的 K 和 V**。但 Q 只用当前这个 token 的(算完就丢)。

所以缓存的目标只有 **K 和 V**——Q 不缓存,因为下一步用不到。

### 1.2 缓存的形状

按一个 token、一层 attention 来看,缓存的张量是:

```
K_layer:  shape [num_kv_heads, head_dim]
V_layer:  shape [num_kv_heads, head_dim]
```

例如 Qwen2.5-7B:`num_kv_heads=4`(GQA), `head_dim=128`,每个 token 每层缓存 `(4, 128) × 2 = 1024 floats × 2`,bf16 下 = 2 KB / token / layer。

整个模型有 N 层(28 层),所以**一个 token 总共占** `28 × 2 KB ≈ 56 KB`。一段 8K prompt 就是 `8192 × 56 KB ≈ 448 MB`。

### 1.3 为什么叫 cache

如果不缓存,每生成一个 decode token,都要重新跑一遍前面整段历史的 forward——复杂度从 O(N) 变成 O(N²),长序列基本跑不动。

KV cache 的核心思想就是「**K、V 是 token 在每层的产物,跟未来无关,算一次存起来反复读**」。decode 阶段每次只算新 token 的 K、V 写回 cache,然后让它的 Q 去读所有历史 K、V 做 attention——单步是 O(已生成长度) 而不是 O(N²)。

### 1.4 SGLang 中的实际布局

SGLang 把 KV cache 实现成两层结构:

| 层 | 名字 | 单位 | 作用 |
|---|---|---|---|
| 索引 | `req_to_token_pool` | per request,N 个 token id | 记录某 req 用了 KV pool 哪些 slot |
| 数据 | `token_to_kv_pool` | per layer, per token, per head | 实际存 K/V 张量 |

每层每 head 是一段连续 GPU 显存,按 page(默认 16 个 token 一页) 分配——对应论文 PagedAttention,目的是避免外部碎片。

---

## 二 HBM 是什么

**HBM** = **High Bandwidth Memory**(高带宽显存)

它是 NVIDIA / AMD 高端 GPU 用的**显存类型**——3D 堆叠 DRAM 颗粒,通过硅穿孔(TSV)和封装内的硅中介(silicon interposer) 直接和 GPU die 相连。

### 2.1 GPU 内存层级

| 层级 | 容量 | 带宽 | 延迟 | 例子 |
|---|---|---|---|---|
| 寄存器 | KB 级(per thread) | 几十 TB/s | <1 ns | per-thread register file |
| SM 共享内存 / L1 cache | 64-128 KB / SM | 几 TB/s | 几 ns | shared memory |
| L2 cache | 几十 MB | TB/s 级 | 几十 ns | unified L2 |
| **HBM(显存)** | 几十到几百 GB | 几 TB/s | 几百 ns | H100 80GB HBM3,3.35 TB/s |
| 主机 RAM | 几百 GB - 几 TB | <100 GB/s(走 PCIe) | 几 µs | DDR5 |
| SSD / 磁盘 | TB 级 | GB/s 级 | ms 级 | NVMe |

GPU 上你看到的所有 `cudaMalloc` 分配的内存,本质上都在 HBM 里——**`KV cache` 默认就存在 HBM**,这是它最快的位置。

### 2.2 为什么叫 high bandwidth

HBM 的带宽数字(TB/s) 是同代 DDR5(几十 GB/s) 的几十倍——做法是:

- **位宽暴力堆**:HBM3 一个 stack 是 1024-bit 总线(普通 DDR5 是 64-bit),H100 上有 5 个 stack,总位宽 5120-bit。
- **3D 堆叠**:8-12 层 DRAM 颗粒垂直堆,通过 TSV 互连,缩短信号路径。
- **就近封装**:不走主板,直接封装在 GPU 同一个 substrate 上,信号距离短到毫米级。

代价:贵、容量受限于堆叠层数、发热集中。所以 HBM 主要用在数据中心 GPU(H100、MI300X、TPU);消费级 GPU(RTX 4090 等) 还在用 GDDR6/6X(便宜、容量大、带宽稍逊)。

### 2.3 HBM 对 LLM 的意义

LLM decode 阶段是 **memory-bound**——每一步要读全部模型权重 + 全部 KV cache(几十 GB),计算量却很小。瓶颈是「GPU 能多快从 HBM 把这些字节读进 SM」。

| GPU | HBM 带宽 | 影响 |
|---|---|---|
| H100 80GB | 3.35 TB/s | decode 每秒能跑大约 100 个 7B-bf16 步 |
| H200 141GB | 4.8 TB/s | 同模型 decode 提升约 40% |
| MI300X 192GB | 5.3 TB/s | 显存大 + 带宽高,适合大模型 |

所以 LLM 推理的硬件评估,**HBM 带宽往往比算力(TFLOPS) 更决定 decode 性能**。

---

## 三 `_prefetch_kvcache()`:把 KV 从下层存储「预取」到 GPU

源码(`scheduler.py:2036-2056`):

```python
def _prefetch_kvcache(self, req: Req):
    if self.enable_hicache_storage:
        req.init_next_round_input(self.tree_cache, cow_mamba=False)
        last_host_node = req.last_host_node
        if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
            last_hash = last_host_node.get_last_hash_value()
            matched_len = len(req.prefix_indices) + req.host_hit_length
            new_input_tokens = req.fill_ids[matched_len:]

            prefix_keys = (
                last_host_node.get_prefix_hash_values(last_host_node.parent)
                if self.tree_cache.hicache_storage_pass_prefix_keys
                else None
            )
            self.tree_cache.prefetch_from_storage(
                req.rid,
                last_host_node,
                new_input_tokens,
                last_hash,
                prefix_keys,
            )
```

入口在 `_add_request_to_queue(req)`(`scheduler.py:2058`) 里——请求一进队就调,目的是让请求真正被调度跑 prefill 时,KV 已经在 HBM 里。

### 3.1 背景:HiCache 三层存储

`enable_hicache_storage` 控制是否启用 SGLang 的 **HiCache**(Hierarchical KV Cache)——把 KV cache 分成三层:

```
┌──────────────────────────────────────────┐
│ L1: HBM(GPU 显存)                        │
│     - 容量:几十 GB                         │
│     - 带宽:几 TB/s                         │
│     - 当前 forward 用的 KV 必须在这        │
└──────────────────────────────────────────┘
                 │ 异步拷贝
                 ▼
┌──────────────────────────────────────────┐
│ L2: Host RAM(主机内存)                   │
│     - 容量:几百 GB                         │
│     - 带宽:几十 GB/s(PCIe)               │
│     - 暂时被 evict 出 GPU 的热数据        │
└──────────────────────────────────────────┘
                 │ 异步拷贝
                 ▼
┌──────────────────────────────────────────┐
│ L3: 持久化存储(SSD / 远程 KV store)      │
│     - 容量:TB - PB                        │
│     - 带宽:GB/s                            │
│     - 跨请求 / 跨进程 / 跨节点共享         │
└──────────────────────────────────────────┘
```

KV cache 不再是「只在 GPU 上,evict 就丢」,而是可以下沉到更便宜的层,**用容量换更高的 prefix 命中率**。

`hicache_storage_backend`(server arg) 决定 L3 用什么:本地 SSD、3FS、Mooncake、远程 KV store 等。

### 3.2 `_prefetch_kvcache` 做的事

四步:

1. **`req.init_next_round_input(self.tree_cache, cow_mamba=False)`**
   - 在 RadixCache 树上查这个请求的 prompt 能匹配多长 prefix。
   - 设置 `req.prefix_indices`(GPU 上已有的 KV 对应 token 段) 和 `req.last_host_node`(host 层 RadixCache 的最末节点)。

2. **决定要不要预取**:
   ```python
   if last_host_node.backuped or last_host_node is self.tree_cache.root_node:
   ```
   - `backuped`:这个 host 节点已经被备份到 L3 storage 里(说明 L3 可能有更长的 prefix 等着我们 fetch 回来)。
   - 是 root_node:啥也没匹配,但仍允许查 storage(可能 L3 里有别的请求留下的 KV)。
   - 否则不预取(L3 没什么可拿的)。

3. **算「新 token 范围」**:
   ```python
   matched_len = len(req.prefix_indices) + req.host_hit_length
   new_input_tokens = req.fill_ids[matched_len:]
   ```
   - `prefix_indices`:GPU L1 已命中部分。
   - `host_hit_length`:host RAM L2 已命中部分。
   - `new_input_tokens`:除掉 L1/L2 已有的之后,L3 storage 要查的 token 段。

4. **异步触发 storage 预取**:
   ```python
   self.tree_cache.prefetch_from_storage(
       req.rid, last_host_node, new_input_tokens, last_hash, prefix_keys,
   )
   ```
   - `last_hash`:用最后命中节点的哈希做 lookup key(content-addressable)。
   - `prefix_keys`:可选的额外哈希链,某些 storage backend 需要它定位。
   - **异步发出**:函数立即返回,真正的拉取由 storage backend 后台线程做。

### 3.3 异步预取与调度的协作

预取是**非阻塞**的——`_add_request_to_queue` 调一次就返回,但加载不一定完成。完成检测在 `get_new_batch_prefill` 里:

```python
# scheduler.py:2549
if self.enable_hicache_storage:
    prefetch_done = self.tree_cache.check_prefetch_progress(req.rid)
    if not prefetch_done:
        # skip staging requests that are ongoing prefetch
        continue
    req.storage_hit_length = self.tree_cache.pop_prefetch_loaded_tokens(req.rid)
```

执行顺序:

```
t=0   _add_request_to_queue(req)
        └─ _prefetch_kvcache(req) → 异步发起 L3 → L2 → L1 加载
        └─ waiting_queue.append(req)

t≈0   prefetch backend 后台线程开始从 L3 拉数据
       (可能要几十 ms,因 SSD/网络速度而定)

t=K   主循环走到 get_new_batch_prefill
        └─ check_prefetch_progress(req.rid)
            ├─ 完成 → req.storage_hit_length = N(L3 命中 N 个 token)
            │         → 加入本批 prefill,跳过 N 个 token
            └─ 没完成 → 跳过这个请求,先组别的,下一轮再试
```

**早发起、迟检查**——把 IO 时间和调度其他工作重叠起来。

### 3.4 命中后的收益

设 prompt 是 8000 token,L1 命中 1000、L2 命中 3000、L3 命中再加 2000、新 token 2000:

- 不启用 HiCache:8000 - 1000 = **7000 token 要重新 prefill**(其余 1000 来自 GPU L1)。
- 启用 HiCache + storage:8000 - 1000 - 3000 - 2000 = **2000 token 要重新 prefill**。
- prefill 是 O(N²) 的,这意味着算力降到原来 `(2000/7000)² ≈ 8%`。

这是 SGLang 在长 prompt / 多轮对话 / RAG 场景吞吐能比同类方案高很多的关键之一。

### 3.5 为什么不无脑预取所有请求

预取也有代价:

- **L3 → L2 是 SSD/网络 IO**:吃带宽,可能延迟 prefill 调度。
- **L2 → L1 是 PCIe DMA**:吃 PCIe 带宽,会和其他 host↔device 拷贝竞争。
- **L1 占用 HBM**:预取来的 KV 要占 GPU 槽位,可能挤掉其他请求的预算。

所以 SGLang 加了几道闸门:

| 闸门 | 用途 |
|---|---|
| `enable_hicache_storage = False`(默认) | 不开 L3,完全跳过预取 |
| `last_host_node.backuped` 检查 | 只对「确实有 L3 备份的请求」预取 |
| 异步 + check_prefetch_progress | 没拉好就让 batch 跳过这个请求,先跑别的 |
| `pop_prefetch_loaded_tokens` 而不是「等到 100%」 | 拿到多少用多少,余下的下一轮再补 |

---

## 四 一句话总结

> **KV cache** 存的就是每个 token 在每层 attention 算出来的 K 向量(`W_K · x`) 和 V 向量(`W_V · x`),Q 不存——目的是 decode 阶段不重算历史 attention 的 K/V。
>
> **HBM** 是 GPU 用的高带宽 3D 堆叠显存,带宽几 TB/s,KV cache 默认就放在这里。LLM decode 是 memory-bound,所以 HBM 带宽往往比算力更决定 decode 性能。
>
> **`_prefetch_kvcache()`** 是 SGLang **HiCache 三层存储(HBM ↔ Host RAM ↔ SSD/远程 KV store)** 的预取入口——请求一入队就异步发起从 L3 → L2 → L1 的拉取,等 `get_new_batch_prefill` 真正调度它时,KV 已经准备好,大幅减少需要 prefill 的 token 数,长 prompt / 多轮场景吞吐显著提升。
