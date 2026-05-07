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

### 2.1 「切的是输出维度,不是输入 hidden」

很多教程说"把 hidden 切成 num_heads 份",这话**容易让人误以为是切输入 X**——其实不是。**真正切的是投影矩阵的输出维度**。

先把矩阵尺寸列清楚(以 Llama-7B 为例):

```
X     ∈ [N, hidden]                    形状 [N, 4096]    输入,N = token 数
W_Q   ∈ [hidden, num_heads × head_dim] 形状 [4096, 4096] 权重矩阵
Q     = X @ W_Q                                          → [N, 4096]
```

注意 `W_Q` 的**输出维度** = `num_heads × head_dim` = 32 × 128 = 4096(刚好等于 hidden_size,但**这只是巧合**,等会儿讲 GQA 就破了)。

**多头的真正含义**:把 `W_Q` 的输出列**按 `head_dim` 一段一段切**,看成 `num_heads` 个并列的小投影:

```
W_Q = [ W_Q^(0)  |  W_Q^(1)  |  ...  |  W_Q^(31) ]      ← 列方向并排放
        ↑           ↑                    ↑
       [4096,128]  [4096,128]           [4096,128]      每个 W_Q^(i) 都吃完整的 hidden=4096
```

数学上完全等价:

```
Q = X @ W_Q                                  一次大 GEMM
  = [X @ W_Q^(0) | X @ W_Q^(1) | ... | X @ W_Q^(31)]    把每个头看成独立小 GEMM 拼起来
```

每个 `W_Q^(i)` 都吃**整个 hidden=4096 输入**,产出 `head_dim=128` 维的 Q 子向量。**输入 X 不切,输出列在切**。这才符合矩阵乘法规则。

> 你之前的疑问"切 hidden 维不符合矩阵乘法"是对的——所以这里要切的不是 X,而是 W_Q 的输出列(产出的 Q 张量在最后一维上看上去被切了 num_heads 段)。

### 2.2 切完之后怎么用

`Q ∈ [N, 4096]` 通过一个 **reshape**(零拷贝,只改 stride/shape 元数据)变成 `[N, num_heads=32, head_dim=128]`,**这就是"多头"在代码里的形状**:

```python
q = q.view(N, num_heads, head_dim)        # [N, 4096] → [N, 32, 128]
```

后面 attention 就**对每个头独立**算 softmax(Q·Kᵀ/√d)·V,32 个头并行进行。

### 2.3 Llama-7B 数字一览

| 字段 | 值 |
|---|---|
| hidden_size | 4096 |
| num_attention_heads (Q 头数) | 32 |
| head_dim | 4096 / 32 = 128 |
| num_key_value_heads (K/V 头数) | 32(Llama-1/2) 或 **8**(Llama-3 GQA) |
| W_Q 形状 | `[4096, 4096]`(输出 = 32 × 128) |
| W_K 形状(GQA) | `[4096, 1024]`(输出 = 8 × 128,**比 hidden 小**) |
| W_V 形状(GQA) | `[4096, 1024]` |

**关键观察**:`W_K` 的输出维度 = `num_kv_heads × head_dim` = `8 × 128 = 1024`,**比 hidden=4096 小很多**。这就是 GQA 省 KV cache 的根源——K、V 在投影出来时就**变窄**了。

### 2.4 为什么多头

让模型能在不同子空间分别做 attention——一个头关注语法,一个头关注语义,一个头关注指代,等等。**经验上**:头多比头宽好(`32 × 128` 比 `1 × 4096` 效果好)。

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

## 四 GEMM 是什么

**GEMM = General Matrix Multiplication**(通用矩阵乘法),BLAS-3 标准算子,数学上就是:

```
C = α · A · B + β · C
```

A、B、C 是矩阵,α、β 是标量。在深度学习里大多数情况 α=1、β=0,所以化简为 `C = A · B`。

**为什么 GEMM 这么重要**:
- 现代 GPU 把 GEMM 优化到极致(NVIDIA cuBLAS / cuDNN / CUTLASS,Tensor Core 起飞就是因为它专做 GEMM)
- 一个**够大**的 GEMM 能跑到 GPU 算力上限的 ~90%(算力受限,也就是 compute-bound)
- 反过来,**一堆小 GEMM** 每次 launch 都有几 μs 开销,而且每次都要把权重从显存读到寄存器,**显存带宽受限**(memory-bound),浪费算力

LLM 推理里的所有线性层(Linear / Projection)本质上都是 GEMM:`output = input @ weight + bias`。**「一次大 GEMM 比三次小 GEMM 快」就是 fused QKV 的核心动机**。

---

## 五 Q、K、V 在代码层面到底怎么算出来

这一节回答"qkv 融合到底是怎么融合的""三个矩阵在代码里是怎么算出来的"。

### 5.1 朴素写法(三次独立 GEMM)

最直白的实现是三个独立的 `nn.Linear`:

```python
self.q_proj = nn.Linear(hidden, num_q × head_dim,  bias=False)   # W_Q ∈ [4096, 4096]
self.k_proj = nn.Linear(hidden, num_kv × head_dim, bias=False)   # W_K ∈ [4096, 1024]
self.v_proj = nn.Linear(hidden, num_kv × head_dim, bias=False)   # W_V ∈ [4096, 1024]

# forward:
q = self.q_proj(x)    # GEMM #1: [N, 4096] @ [4096, 4096]  → [N, 4096]
k = self.k_proj(x)    # GEMM #2: [N, 4096] @ [4096, 1024]  → [N, 1024]
v = self.v_proj(x)    # GEMM #3: [N, 4096] @ [4096, 1024]  → [N, 1024]
```

三次 GEMM,每次都要把同一个 `x` 从显存读一遍,共 3 次冗余读取。

### 5.2 SGLang 写法:融合成一个大 GEMM

把三个权重矩阵在**输出维度(列)上拼接**,得到一个大权重 `W_QKV`:

```
W_QKV = [ W_Q  |  W_K  |  W_V ]               按列拼
       ∈ [hidden, q_size + kv_size + kv_size]
       = [4096, 4096 + 1024 + 1024]
       = [4096, 6144]
```

一次 GEMM 把 Q/K/V 全算出来:

```python
self.qkv_proj = QKVParallelLinear(
    hidden_size,                   # 4096
    self.head_dim,                 # 128
    self.total_num_heads,          # 32  → q_size = 32 × 128 = 4096
    self.total_num_kv_heads,       # 8   → kv_size = 8 × 128 = 1024
    bias=bias, ...
)
# 内部权重张量:weight ∈ [hidden=4096, q_size+kv_size+kv_size=6144]

# forward:
qkv, _ = self.qkv_proj(hidden_states)
# 等价于:qkv = hidden_states @ W_QKV
#         [N, 4096] @ [4096, 6144]  →  [N, 6144]
```

**数学上**:

```
qkv = x @ W_QKV
    = x @ [W_Q | W_K | W_V]
    = [x @ W_Q  |  x @ W_K  |  x @ W_V]      ← 矩阵乘法对列拼接的天然分配律
    = [   Q    |     K     |     V   ]
```

所以一次 GEMM 的输出**自动就是 [Q | K | V] 拼起来的样子**。

### 5.3 split 拿到 q、k、v(零拷贝!)

```python
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
# q.shape = [N, 4096]    ← qkv 的列 [0     : 4096]
# k.shape = [N, 1024]    ← qkv 的列 [4096  : 5120]
# v.shape = [N, 1024]    ← qkv 的列 [5120  : 6144]
```

**`split` 不复制内存**,返回 3 个 view,共享 `qkv` 的底层 Storage,只是 stride/offset 不同。这是 PyTorch Tensor 的"元数据头 + 共享 Storage"设计的好处(参考 [`torch.Tensor` 是怎样的数据结构]——只改头不动数据)。

### 5.4 三次 vs 一次的性能差距

| | 朴素三次 GEMM | 融合一次 GEMM |
|---|---|---|
| GEMM 次数 | 3 | 1 |
| **`x` 显存读取次数** | **3 次** | **1 次** |
| 权重显存读取 | 3 块独立权重 | 1 块大权重(连续布局,缓存友好) |
| GEMM 算术强度 | 三次中等规模 | 一次大规模(更接近 GPU 算力上限) |
| Python kernel launch 开销 | 3 次 | 1 次 |

prefill 阶段 N 大(几百~几千),GEMM 是 compute-bound,差距体现在"是否能打满 Tensor Core";decode 阶段 N=batch_size 很小,GEMM 是 memory-bound,差距体现在"权重和 hidden 张量的读取次数"。**两阶段都受益**。

### 5.5 权重加载:HuggingFace → SGLang 的拼接

HuggingFace 原始 checkpoint 里 `q_proj.weight / k_proj.weight / v_proj.weight` 是**三个独立张量**,SGLang 启动时要把它们**沿输出维度 concat** 成一个 `qkv_proj.weight`。这一步通过 `stacked_params_mapping`(`llama.py:490`)告诉权重加载器"这三个名字应该被合并到一个目标参数":

```python
self.stacked_params_mapping = [
    # (param_name,    shard_name,  shard_id)
    (".qkv_proj",     ".q_proj",   "q"),
    (".qkv_proj",     ".k_proj",   "k"),
    (".qkv_proj",     ".v_proj",   "v"),
    ...
]
```

加载时,加载器看到 HF 的 `model.layers.0.self_attn.q_proj.weight`,就知道把它写到目标 `qkv_proj.weight` 的"q 段"(前 4096 列);看到 `k_proj.weight` 写到"k 段"(中间 1024 列);`v_proj.weight` 写到"v 段"(后 1024 列)。

**所以模型功能完全等价**,只是内存里的权重布局变了。

---

## 六 fused QKV 收益小结

- 一次大 GEMM 比三次小 GEMM 更能打满 GPU 算力(算术强度更高)
- `x` 只读一次而不是三次(显存带宽减少 1/3)
- 一次 kernel launch 而不是三次(减少 ~10 μs CPU 开销)
- 权重在内存里连续,加载器和缓存更友好

---

## 七 TP 切分(分布式)

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

## 八 完整初始化代码逐行讲解(`llama.py:139-173`)

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

## 九 split 在 `forward_prepare_native` 里的实际形态

代码 `llama.py:200`:

```python
qkv, _ = self.qkv_proj(hidden_states)
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
```

- `qkv.shape == [N, q_size + kv_size + kv_size]`
- `q_size = num_heads × head_dim`(**本卡**,TP 切过)
- `kv_size = num_kv_heads × head_dim`(**本卡**)

`split` **不复制内存**(详见 §5.3)。后面 RoPE 会**原地修改 q、k**(数值变,形状不变),v 全程不动。

---

## 十 一句话总结

> **`qkv_proj` 用一个融合大 GEMM 把 hidden 投影成 `[Q | K | V]` 拼起来的张量,split 拆开拿到 q、k、v**;Q 头数和 KV 头数可以不一样(GQA),KV 头数少 → KV cache 体积按比例缩小,这是 Llama-3 / Qwen / DeepSeek 推理性能能起来的根因之一。TP 时 `QKVParallelLinear` 列并行切头,KV 头数不够分时直接复制。
