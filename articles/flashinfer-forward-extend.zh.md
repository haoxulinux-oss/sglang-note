# `FlashInferAttnBackend.forward_extend()` 详解(初学者向)

> 📖 **关联阅读**:
> - 父篇:[Attention 计算与 KV cache 写入](llama-attention-and-kvcache.zh.md) §三 已经讲清「`forward_batch.attn_backend.forward` 在 RTX 4070 上就是 `FlashInferAttnBackend.forward`」,然后按 `forward_mode` 派发到本函数
> - 上游:[Self-Attention 总览](llama-self-attention-overview.zh.md) §四 RadixAttention
> - 横向:[KV cache 存什么、HBM 是什么](kvcache-prefetch-and-storage.zh.md)

代码位置:
- `FlashInferAttnBackend.forward_extend`:`python/sglang/srt/layers/attention/flashinfer_backend.py:775`
- `MHATokenToKVPool.set_kv_buffer`:`python/sglang/srt/mem_cache/memory_pool.py:1022`
- `init_forward_metadata`:`python/sglang/srt/layers/attention/flashinfer_backend.py:433`

---

## 一 这一步在做什么

`FlashInferAttnBackend.forward_extend()` 是 **RTX 4070 上 prefill / extend / chunked-prefill 阶段 attention 的真实入口**。它做三件事:

1. **挑出对应的 paged-wrapper**(`prefill_wrapper_paged`)
2. **把这一批新算的 K、V 写入 KV cache 池**(可能跳过)
3. **算 attention**:`softmax(Q · Kᵀ / √d_k) · V`,带 causal mask

最终输出形状 `[total_tokens, num_q_heads × head_dim]`(和 `LlamaAttention` 期望的 `attn_output` 一致)。

---

## 二 函数签名和调用者

```python
def forward_extend(
    self,
    q: torch.Tensor,            # [total_tokens, num_q_heads × head_dim]
    k: torch.Tensor,            # [total_tokens, num_kv_heads, head_dim] (本批新算的 K)
    v: torch.Tensor,            # [total_tokens, num_kv_heads, head_dim] (本批新算的 V)
    layer: RadixAttention,      # 当前 attention 层的元数据载体
    forward_batch: ForwardBatch,# 整批请求的全部数据
    save_kv_cache: bool = True, # 是否把 K/V 写进 cache(只有跑 encoder-only 等场景才 False)
):
```

调用者是 `AttentionBackend.forward`(`base_attn_backend.py:81`),按 `forward_mode` 分流到这里:

```
forward_batch.attn_backend.forward(...)              # base_attn_backend.py
   └─ if forward_mode.is_extend():
        self.forward_extend(q, k, v, layer, fb, save_kv_cache)   ← 本文
```

---

## 三 出场角色与关键成员预热

读懂这个函数,需要先认识 4 个对象:

| 对象 | 类型 / 类 | 重要字段 / 方法 | 作用 |
|---|---|---|---|
| `layer` | `RadixAttention` | `layer_id` / `tp_q_head_num` / `tp_k_head_num` / `head_dim` / `scaling` / `k_scale_float` 等 | 当前层的元数据。把 q/k/v 形状信息和 scale 携带过来 |
| `forward_batch` | `ForwardBatch` | `out_cache_loc` / `token_to_kv_pool` / `req_pool_indices` / `seq_lens` / `extend_prefix_lens` | 整批请求的运行时状态(详见父篇 §2.1) |
| `forward_batch.token_to_kv_pool` | **`MHATokenToKVPool`**(`memory_pool.py:764`) | `set_kv_buffer()` / `get_kv_buffer()` / `k_buffer` / `v_buffer` | **真正持有 KV 显存池的对象**——这是 `set_kv_buffer` 的归属类 |
| `self.forward_metadata` | **`PrefillMetadata`**(`flashinfer_backend.py:99`) | `prefill_wrappers` / `use_ragged` / `extend_no_prefix` / `multi_item_params` | 启动期+本轮预算好的「attention kernel 启动配置」 |

下面 §四、§五、§六 分别讲清这三个核心对象。

---

## 四 `set_kv_buffer` 是哪个类的成员

`forward_extend` 里这行:

```python
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
)
```

调到 `MHATokenToKVPool.set_kv_buffer`(`memory_pool.py:1022`)——**MHA Token-to-KV Pool**,SGLang 持有 KV 显存的核心类。继承关系:

```
KVCache (抽象基类, memory_pool.py:~700)
   └── set_kv_buffer  (抽象方法, memory_pool.py:742)
   └── get_kv_buffer
   └── get_key_buffer
   ↑ 继承
MHATokenToKVPool       (memory_pool.py:764)           ← Llama / Qwen 等 MHA/GQA 模型用这个
MHATokenToKVPoolFP4    (memory_pool.py:1111)          ← FP4 量化版本
MLATokenToKVPool       (memory_pool.py:~1400)         ← DeepSeek 系 MLA 用
HybridLinearKVPool     (memory_pool.py:~1750)         ← 混合架构用
...
```

### 4.1 `MHATokenToKVPool` 的物理布局

`__init__` 里 `_create_buffers()`(`memory_pool.py:872`)给每层都开两个张量:

```python
self.k_buffer = [
    torch.zeros((size + page_size, head_num, head_dim), dtype=store_dtype, device=device)
    for _ in range(layer_num)
]
self.v_buffer = [ 同样形状 ]
```

举例 Llama-3-8B(`head_num=8` GQA,`head_dim=128`,`bf16`,32 层,假设 `size=65536`):

```
k_buffer[layer_id]: [65536+16, 8, 128]  bf16  → 大约 128 MB / 层
v_buffer[layer_id]: 同上                       → 128 MB / 层
合计 32 层:K + V = 32 × (128+128) MB ≈ 8 GB    ← 这就是 KV cache 显存占用
```

**每张卡每层有 `(size + page_size)` 个 KV slot**,每个 slot 是 `[head_num, head_dim]` 的一片小张量。`size` 是 scheduler 分配给 KV pool 的物理 token 容量上限,启动时根据剩余显存决定。

### 4.2 `set_kv_buffer` 的核心代码

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, ...):
    layer_id = layer.layer_id
    # 处理量化的 scale(默认 bf16 时跳过)
    if cache_k.dtype != self.dtype:
        if k_scale is not None: cache_k.div_(k_scale)
        if v_scale is not None: cache_v.div_(v_scale)
        cache_k = cache_k.to(self.dtype)
        cache_v = cache_v.to(self.dtype)

    # ★ 真正的写入,调底层 fused kernel
    _set_kv_buffer_impl(
        cache_k, cache_v,
        self.k_buffer[layer_id - self.start_layer],    # 目标 K 池(本层)
        self.v_buffer[layer_id - self.start_layer],    # 目标 V 池(本层)
        loc,                                             # 写到哪些位置(物理 slot 索引)
        ...
    )
```

`_set_kv_buffer_impl`(`memory_pool.py:90`)做的事用伪代码表示:

```python
# 简化版,实际有 fused store_cache kernel(同时写 K、V)
k_buffer[indices] = k      # scatter 写
v_buffer[indices] = v
```

其中 `indices = forward_batch.out_cache_loc`,**这一批每个新 token 的物理 KV 槽位编号**。

> 关键事实:**`MHATokenToKVPool` 是真正持有 GPU HBM 上 KV 张量的对象**。它由 `ModelRunner.init_memory_pool()` 在启动期创建,挂在 `forward_batch.token_to_kv_pool` 上跨进程访问。

---

## 五 `self.forward_metadata.use_ragged` 是什么意思

`forward_metadata` 是 `FlashInferAttnBackend.init_forward_metadata`(`:433`) 在**每次 forward 开始前**重新生成的——根据当前 batch 的 `forward_mode` 选好对应的 wrapper 和参数。对 EXTEND 路径,它的类型是:

```python
@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]   # 实际要用的 paged wrapper
    use_ragged: bool                                               # ★ 本节焦点
    extend_no_prefix: bool                                         # 这一批是否完全没有历史前缀
    multi_item_params: Optional[MultiItemScoringParams]            # 多项打分专用
```

### 5.1 Ragged 和 Paged 的区别

FlashInfer 提供两种 prefill wrapper,**对应两种 K/V 的"输入布局"**:

| Wrapper | K/V 输入来源 | 适用场景 |
|---|---|---|
| **`BatchPrefillWithPagedKVCacheWrapper`** (paged) | 从 KV cache **页表**读 K、V(K/V 在 cache 池里离散分布) | 永远适用——既能读历史前缀,也能在写入后读到自己的新 K/V |
| **`BatchPrefillWithRaggedKVCacheWrapper`** (ragged) | 直接从**当前一批 concat 在一起的 K、V 张量**读,**不经过 cache** | 只跑「无前缀」prefill 时更快,因为省掉一次 page table 间接寻址 |

**Ragged = "不规整地拼接"**——这一批每个 req 的 K、V 序列长度不一样(128、200、64),直接 concat 成 `[total_tokens, num_kv_heads, head_dim]` 的扁平张量,kernel 通过 `seq_lens_cumsum` 知道每个 req 的边界。

> 名字来源:相对于 padded(规整 padding 到统一长度,浪费显存),"ragged" 表示**不 padding、不规整、变长**——和 NestedTensor 是类似概念。

### 5.2 `use_ragged` 何时为 True

在 `init_forward_metadata` 的 EXTEND 分支(`:491`):

```python
if self.is_multimodal or self.enable_mis:
    use_ragged = False        # 多模态 / 多项打分,需要 paged 的特殊掩码功能
else:
    use_ragged = (
        not self.enable_deterministic                  # 确定性推理时必须 paged(可复现)
        and not is_in_piecewise_cuda_graph()           # piecewise CUDA Graph 内不用 ragged
    )
    extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
```

**RTX 4070 上的常规 prefill 默认 `use_ragged=True`**(没开多模态、没开 deterministic、没在 piecewise graph 里时)。

### 5.3 `extend_no_prefix` 是什么

```python
extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
```

如果**这一批所有请求都没有"历史前缀"**(没命中 RadixCache 的共享前缀),`extend_no_prefix=True`。**全新 prompt 是这种情况**;如果命中过 RadixCache 前缀(前 1024 token 已在 cache 里、只需对剩余 token 跑 prefill),就是 False。

这个标志影响 `use_ragged=True` 路径下的 sub-branch(详见 §七)。

---

## 六 `prefill_wrapper_paged.forward` 的实现在哪

```python
prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
    self._get_wrapper_idx(layer)
]
```

`prefill_wrappers` 元素的类型是 **`flashinfer.BatchPrefillWithPagedKVCacheWrapper`**——**SGLang 不写这个类,它来自外部 `flashinfer` Python 库**(import 在 `flashinfer_backend.py:51`):

```python
if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
```

### 6.1 FlashInfer 项目在哪

- GitHub 仓库:[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- Python 安装时的位置:`$VIRTUAL_ENV/lib/python3.*/site-packages/flashinfer/`
- 源码主要文件:
  - `flashinfer/prefill.py` — `BatchPrefillWithPagedKVCacheWrapper` 的 Python 类
  - `flashinfer/decode.py` — `BatchDecodeWithPagedKVCacheWrapper`
  - `csrc/flashinfer_ops.cu` — C++/CUDA kernel 入口
  - `include/flashinfer/attention/` — CUDA kernel 模板

### 6.2 wrapper 的两阶段使用

FlashInfer wrapper 的 API 是**两阶段**:`plan` + `run`(`forward` 是 `run` 的便捷别名)。

```python
# ── 阶段 1:plan(每次 forward 开始前调一次,在 SGLang 里就是 init_forward_metadata 里间接调的)──
prefill_wrapper_paged.plan(
    qo_indptr=...,            # 每个 req 的 query 偏移
    paged_kv_indptr=...,      # 每个 req 的 KV 页偏移
    paged_kv_indices=...,     # KV 页的物理编号
    paged_kv_last_page_len=...,  # 最后一页有多少有效 token
    num_qo_heads=...,
    num_kv_heads=...,
    head_dim=...,
    page_size=...,
)

# ── 阶段 2:forward(每层 attention 都调一次)──
output = prefill_wrapper_paged.forward(
    q,                       # [total_tokens, num_q_heads, head_dim]
    kv_cache,                # token_to_kv_pool.get_kv_buffer(layer_id) → (k_buffer, v_buffer)
    causal=True,
    sm_scale=layer.scaling,
    window_left=...,         # sliding window 配置(Llama 通常 -1 关闭)
    logits_soft_cap=...,
    k_scale=..., v_scale=...,
)
```

**`plan` 把"形状/索引/分块策略"算好,存在 wrapper 内部**——这是 host 端的轻量准备(几百 μs)。**`forward` 才是真正调 CUDA kernel**——所有 32 层都共享同一个 plan,只在第一层 attention 之前 plan 一次。

### 6.3 forward 内部走到哪个 kernel

`BatchPrefillWithPagedKVCacheWrapper.forward` 内部根据 `backend` 参数(`auto` / `cutlass` / `fa3` / `cudnn` 等)分发:

- **Ada Lovelace(RTX 4070)的默认 backend** 是 FlashInfer 自家的 paged attention CUDA kernel(模板生成的),源码在 `flashinfer/include/flashinfer/attention/prefill.cuh`
- 调用栈:`Python forward → C++ binding → CUDA kernel launch → SM 上的 paged attention 算法`

paged attention 算法核心:
1. 按 page(16 token 一页是 SGLang 默认)读 K、V
2. tile 化:把 query / KV 分块装入 shared memory
3. 在 shared memory 内算 `softmax(Q · Kᵀ / √d)`,边算 softmax 边乘 V(FlashAttention 思想——不实例化中间 `[N, N]` 注意力矩阵)
4. 写回 output

> 想看 CUDA kernel 源码请去 FlashInfer 项目;**SGLang 这边只负责"调它"**。这种"Python 高层 + 外部库底层 CUDA"是 SGLang 推理引擎的标准结构。

---

## 七 `forward_extend` 三条执行分支

现在把整个函数串起来。根据 `use_ragged` 和 `extend_no_prefix`,有**三条分支**:

```python
if not self.forward_metadata.use_ragged:
    # ─ 分支 A:paged-only(use_ragged=False)─
    ...
else:
    # use_ragged=True
    if self.forward_metadata.extend_no_prefix:
        # ─ 分支 B:纯 ragged(无历史前缀)─
        ...
    else:
        # ─ 分支 C:ragged + paged + merge_state(有历史前缀)─
        ...
```

### 7.1 分支 A:`use_ragged=False`(paged-only)

适用场景:多模态、deterministic 推理、piecewise CUDA Graph 内。

```python
# ① 写 K/V 到 cache 池
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
    )

# ② 直接调 paged wrapper(K/V 已经在 cache 里了,kernel 从 cache 读)
o = prefill_wrapper_paged.forward(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),   # (k_buffer, v_buffer)
    causal=True,
    sm_scale=layer.scaling,
    window_left=layer.sliding_window_size,
    logits_soft_cap=layer.logit_cap,
    k_scale=layer.k_scale_float, v_scale=layer.v_scale_float,
)
```

特点:**先写 cache、再从 cache 读**。简单可靠。

### 7.2 分支 B:`use_ragged=True` + `extend_no_prefix=True`(纯 ragged)

适用场景:**全新 prompt,没命中任何 RadixCache 前缀**——最快路径。

```python
o = self.prefill_wrapper_ragged.forward(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k.view(-1, layer.tp_k_head_num, layer.head_dim),     # ★ 直接读输入的 k 张量,不经过 cache
    v.view(-1, layer.tp_v_head_num, layer.head_dim),
    causal=True,
    sm_scale=layer.scaling,
    logits_soft_cap=logits_soft_cap,
)

# 写 cache 在 attention 算完之后做(因为 attention 不依赖 cache,可以延后)
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
```

特点:
- attention 不读 KV cache 池,**直接消费 `forward_extend` 的输入 k、v 参数**
- 省掉一次"先写 cache 再读"的多余间接寻址
- KV cache 写入推迟到 attention 之后

### 7.3 分支 C:`use_ragged=True` + 有历史前缀(split-K merge)

适用场景:**chunked prefill 或 RadixCache 命中**——一部分 K、V 在 cache 池里(前缀),一部分是本次新算的(增量),需要分两次跑 attention 然后合并。

```python
# ── 第一段:对本次新算的 K、V 跑 ragged ──
o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k.view(-1, layer.tp_k_head_num, layer.head_dim),    # 新 K
    v.view(-1, layer.tp_v_head_num, layer.head_dim),    # 新 V
    causal=True,                                         # 新 token 之间有 causal
    sm_scale=layer.scaling,
    logits_soft_cap=logits_soft_cap,
)

# ── 第二段:对历史前缀(在 paged cache 里)跑 paged ──
o2, s2 = prefill_wrapper_paged.forward_return_lse(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
    causal=False,                                        # 前缀对当前 token 是「过去」,不用 causal
    sm_scale=layer.scaling,
    logits_soft_cap=logits_soft_cap,
)

# ── 第三步:用 log-sum-exp 合并两段 attention ──
o, _ = merge_state(o1, s1, o2, s2)

# 最后写 cache
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
```

**`forward_return_lse` 比 `forward` 多返回一个 `s`**——是 `log-sum-exp`(LSE),记录这一段 softmax 的归一化常数。**`merge_state` 用 LSE 把两段 attention 在数学上正确地合并**:

```
最终 attention = softmax([logits_新 | logits_旧]) · [V_新 | V_旧]
              = α · softmax(logits_新) · V_新 + (1-α) · softmax(logits_旧) · V_旧
                                       (α 由 LSE 决定)
```

这就是「split-K attention」——和 FlashAttention 论文里的 split-K 概念一致,只是 SGLang 把它用来分别处理"新 K"和"prefix K"。

---

## 八 一次完整 forward_extend 的形状追踪(分支 B 为例)

batch=2,prompt 长度 [128, 64],Llama-3-8B,TP=1:

```
total_tokens = 128 + 64 = 192

输入:
  q:            [192, 4096]       (32 Q 头 × 128 head_dim)
  k:            [192, 1024]       (8 KV 头 × 128 GQA)
  v:            [192, 1024]
  cache_loc:    [192]              本批每个 token 的物理 KV slot 编号
  layer.layer_id: 比如 5(第 6 层)

prefill_wrapper_paged 选定:self.forward_metadata.prefill_wrappers[idx]
use_ragged = True, extend_no_prefix = True  → 走分支 B

调 prefill_wrapper_ragged.forward:
  q.view(-1, 32, 128)              [192, 32, 128]
  k.view(-1, 8, 128)               [192, 8, 128]
  v.view(-1, 8, 128)               [192, 8, 128]
  →  FlashInfer kernel 计算
  →  output o ∈ [192, 32, 128]

写 KV cache:
  token_to_kv_pool.k_buffer[5][cache_loc] ← k         (scatter)
  token_to_kv_pool.v_buffer[5][cache_loc] ← v

返回:
  o.view(-1, 32 × 128) = [192, 4096]   ← 给 LlamaAttention.o_proj
```

---

## 九 一图串起所有对象关系

```
forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
   │
   ├─ self.forward_metadata                            ← PrefillMetadata
   │     ├─ .prefill_wrappers[idx]                     ← BatchPrefillWithPagedKVCacheWrapper (FlashInfer 库)
   │     ├─ .use_ragged             ← True/False       ← Ragged 还是 Paged 路径
   │     └─ .extend_no_prefix       ← True/False       ← 是否有历史前缀
   │
   ├─ forward_batch                                    ← ForwardBatch
   │     ├─ .out_cache_loc          ← [total_tokens]   ← 这批新 token 写到 KV pool 哪里
   │     ├─ .token_to_kv_pool       ← MHATokenToKVPool ← KV cache 池本体
   │     │     ├─ .k_buffer / .v_buffer  [layer_num × (size, head_num, head_dim)]
   │     │     ├─ .set_kv_buffer(layer, loc, k, v, ...)  ← 写 cache 的入口
   │     │     └─ .get_kv_buffer(layer_id)               ← 读 cache 的入口
   │     └─ .extend_prefix_lens     ← 每个 req 的前缀长度
   │
   ├─ layer                                            ← RadixAttention
   │     ├─ .layer_id                                  ← 当前层号
   │     ├─ .tp_q_head_num / .tp_k_head_num / .tp_v_head_num
   │     ├─ .head_dim / .scaling                       ← 1/√d_k
   │     ├─ .sliding_window_size                       ← Llama 通常 -1(关闭)
   │     └─ .k_scale_float / .v_scale_float            ← KV cache 量化时用
   │
   └─ 调用栈:
        └─ prefill_wrapper_paged.forward(...) (FlashInfer Python)
              └─ C++ binding (csrc/)
                   └─ CUDA kernel (paged attention,FA-style)
                        └─ shared memory tile + softmax + V 加权累加
```

---

## 十 一句话总结

> **`FlashInferAttnBackend.forward_extend` = 「拿 PrefillMetadata 里的 wrapper + 写 KV cache + 调 FlashInfer kernel 算 softmax(QKᵀ/√d)·V」**。`set_kv_buffer` 是 **`MHATokenToKVPool`**(继承自 `KVCache`)的方法,真正持有 GPU HBM 上的 K/V 张量;`use_ragged` 表示 **K/V 来源是「输入张量(ragged 布局)」还是「KV cache 页表(paged 布局)」**,有历史前缀时会同时跑两条然后用 LSE 合并;`prefill_wrapper_paged.forward` 的实现**不在 SGLang,在外部 `flashinfer` 项目**(github flashinfer-ai/flashinfer),底层是手写的 paged attention CUDA kernel。
