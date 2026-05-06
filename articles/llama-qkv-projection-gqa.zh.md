# QKV Projection 与 GQA(初学者向)

> 📖 **关联阅读**:
> - 父篇:[Self-Attention 总览](llama-self-attention-overview.zh.md) ← 看这里来
> - 系列:[RMSNorm + 残差](llama-rmsnorm-and-residual.zh.md) → 本篇 → [RoPE](llama-rope.zh.md) → [Attention + KV cache](llama-attention-and-kvcache.zh.md)

代码位置:
- `LlamaAttention.__init__`:`python/sglang/srt/models/llama.py:121-198`
- `forward_prepare_native`:`python/sglang/srt/models/llama.py:200`
- `QKVParallelLinear` 实现:`python/sglang/srt/layers/linear.py`

---

## 一 什么是 QKV Projection

每个 token 的 hidden 向量 `x ∈ ℝ^{hidden}` 要被三个不同的线性层"投影"成三个角色:

```
Q = x · W_Q       (作为「问题」)
K = x · W_K       (作为「索引」)
V = x · W_V       (作为「内容」)
```

直觉:
- **Q(query)**:当前 token 的"问题"——「我想找什么样的信息」
- **K(key)**:历史 token 的"标签"——「我能回答什么样的问题」
- **V(value)**:历史 token 的"实际内容"——被借来更新当前 hidden

attention 公式 `softmax(Q·Kᵀ/√d) · V` 就是:用我的 Q 和所有人的 K 算相似度,按相似度加权混合所有人的 V。

---

## 二 多头(Multi-Head)

把整个 hidden 维度切成 `num_heads` 份,**每份独立做一次 attention**,最后拼回来。Llama-7B 的设置:

| 字段 | 值 |
|---|---|
| hidden_size | 4096 |
| num_attention_heads (Q 头数) | 32 |
| head_dim | 4096 / 32 = 128 |
| num_key_value_heads (K/V 头数) | 32(Llama-1/2) 或 **8**(Llama-3 GQA) |

**为什么多头**?让模型能在不同子空间分别做 attention——一个头关注语法,一个头关注语义,一个头关注指代,等等。**经验上**:头多比头宽好。

---

## 三 GQA:为什么 K/V 头比 Q 头少

**纯多头(MHA)**:Q、K、V 各 32 头 → KV cache 体积 ∝ `num_heads × head_dim × num_layers × seq_len × 2`(K + V),爆炸。

**Multi-Query Attention(MQA)**:K、V 只 1 头,所有 Q 头共享。**问题**:质量掉得多。

**Grouped-Query Attention(GQA,Llama-3 / Qwen / DeepSeek 都用)**:K/V 头数小于 Q 头数,Q 头分组共享 K/V 头。Llama-3-8B 是 32 Q 头 / 8 KV 头,**4 个 Q 头共享一组 K/V**。

```
Q heads:   [Q0 Q1 Q2 Q3] [Q4 Q5 Q6 Q7] ... [Q28 Q29 Q30 Q31]    (32 个)
              ↓ 共享         ↓ 共享              ↓ 共享
KV heads:    K0 V0          K1 V1            ...   K7 V7        (8 个)
```

**收益**:KV cache 缩小 4 倍。**质量**:介于 MHA 和 MQA 之间,实测接近 MHA。

代码里就一对字段:

```python
self.total_num_heads    = num_heads          # 全局 Q 头数  (Llama-3-8B = 32)
self.total_num_kv_heads = num_kv_heads       # 全局 KV 头数 (Llama-3-8B = 8)
```

---

## 四 fused QKV Projection:为什么三合一

**朴素写法**(三个独立的线性层):

```python
q = self.q_proj(x)    # [N, hidden] @ [hidden, num_q × head_dim]
k = self.k_proj(x)    # [N, hidden] @ [hidden, num_kv × head_dim]
v = self.v_proj(x)    # [N, hidden] @ [hidden, num_kv × head_dim]
```

**SGLang 写法**(融合成一次 GEMM):

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,                  # in
    self.head_dim,
    self.total_num_heads,         # 32
    self.total_num_kv_heads,      # 8
    bias=bias, ...
)

# forward:
qkv, _ = self.qkv_proj(hidden_states)                          # [N, q_size + kv_size + kv_size]
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

**为什么要融合**?
- 一次大 GEMM 比三次小 GEMM 更能打满 GPU 算力(算术强度更高)
- 节省一次 hidden 张量的读取(`x` 只读一次,而不是三次)
- 反向几乎都是 fused op,推理也跟着 fused 更顺

**权重在内存里就是拼好的**:`W_QKV ∈ ℝ^{hidden × (q_size+kv_size+kv_size)}`,加载时 SGLang 用 `stacked_params_mapping`(`llama.py:490`)把 HuggingFace 原始的 `q_proj.weight / k_proj.weight / v_proj.weight` 在第一维 concat 起来:

```python
self.stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    ...
]
```

---

## 五 TP 切分(分布式)

TP(tensor parallel)把权重切到多张卡。`QKVParallelLinear` 是**列并行**——把输出维度切开,每张卡只算自己那一份头。

```python
tp_size = get_tensor_model_parallel_world_size()
assert self.total_num_heads % tp_size == 0
self.num_heads     = self.total_num_heads    // tp_size      # 本卡 Q 头数
self.num_kv_heads  = max(1, self.total_num_kv_heads // tp_size)  # 本卡 KV 头数
self.q_size        = self.num_heads    * self.head_dim
self.kv_size       = self.num_kv_heads * self.head_dim
```

**两种情况**:

| 情况 | 处理 |
|---|---|
| `total_num_kv_heads ≥ tp_size` | 平均切到每张卡(常见,如 32 KV head + TP=4 → 每张 8 个) |
| `total_num_kv_heads < tp_size` | **复制**到每张卡(GQA 模型 + 大 TP,如 Llama-3-8B 8 KV head + TP=8 → 每张都是 1 个,完全相同) |

代码里这两种情况:

```python
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

后一种 `max(1, ...)` 就保证了"至少 1 个 KV 头"。

---

## 六 完整初始化代码逐行讲解(`llama.py:139-173`)

```python
tp_size = get_tensor_model_parallel_world_size()      # 全局 TP 大小
self.total_num_heads = num_heads                      # 全局 Q 头数(从 hf_config 读)

assert self.total_num_heads % tp_size == 0
self.num_heads = self.total_num_heads // tp_size      # 本卡 Q 头数

self.total_num_kv_heads = num_kv_heads                # 全局 KV 头数
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    assert tp_size % self.total_num_kv_heads == 0
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

self.head_dim = getattr(config, "head_dim", self.hidden_size // self.total_num_heads)
self.q_size  = self.num_heads    * self.head_dim
self.kv_size = self.num_kv_heads * self.head_dim
self.scaling = self.head_dim ** -0.5                  # softmax 里的 1/√d_k

self.qkv_proj = QKVParallelLinear(
    hidden_size,                                       # 输入维度
    self.head_dim,
    self.total_num_heads,
    self.total_num_kv_heads,
    bias=bias,
    quant_config=quant_config,                         # 支持量化(int8/int4/...)
    prefix=add_prefix("qkv_proj", prefix),
)
```

---

## 七 split 的细节(`forward_prepare_native`)

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

- `qkv.shape == [N, q_size + kv_size + kv_size]`
- `q_size = num_heads × head_dim`(本卡)
- `kv_size = num_kv_heads × head_dim`(本卡)

`split` **不复制内存**——它返回 view,共享底层 buffer。后面 RoPE 会**原地修改 q、k**(数值变,形状不变),v 全程不动。

---

## 八 一句话总结

> **`qkv_proj` 用一个融合大 GEMM 把 hidden 投影成 `[Q | K | V]` 拼起来的张量,split 拆开拿到 q、k、v**;Q 头数和 KV 头数可以不一样(GQA),KV 头数少 → KV cache 体积按比例缩小,这是 Llama-3 / Qwen / DeepSeek 推理性能能起来的根因之一。TP 时 `QKVParallelLinear` 列并行切头,KV 头数不够分时直接复制。
