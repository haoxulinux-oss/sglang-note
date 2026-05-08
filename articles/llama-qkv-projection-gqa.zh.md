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

### 3.1 为什么 K 和 V 一定是**同样的头数**

很自然的问题:K 用来算注意力权重,V 是被取的内容,作用不同,头数不必对称?

**答:必须对称。原因有四层。**

#### ① K 和 V 是「成对」的:每个"位置槽"都同时持有一个 key 和一个 value

attention 的语义:
- 对每个历史 token 的每个头,**同时**存一份 key(被 query 检索的"标签")和一份 value(检索命中后被取出的"内容")
- key 和 value 描述**同一个 token 在同一个头视角下**的两面

如果 K 头数 ≠ V 头数,**就没有"一个 key 对应哪个 value"的天然映射**——你存 8 个 key 头但 4 个 value 头,attention 算出 8 路注意力权重要怎么去配 4 个 value?要么平均、要么再加投影,**都引入额外参数和计算,得不偿失**。

#### ② attention 公式天然要求"K 和 V 用同一个头索引访问"

GQA 的 attention 写出来(对 Q 头 i):

```
g = i // group_size                          # 所属 group
attn_weights = softmax(Q_i @ K_g.T / √d_k)   # 用 K_g
output_i     = attn_weights @ V_g            # 还得用 V_g(同一个 g!)
```

**`K` 和 `V` 用了完全相同的索引 `g`**——这意味着它们必须有相同的头数和相同的分组规则,否则公式就写不出来。

#### ③ KV cache 的物理布局是「成对存储」

SGLang 的 `token_to_kv_pool` 给每层开两个 buffer:`k_buffer[layer_id]` 和 `v_buffer[layer_id]`,**形状完全一致**:

```
[max_total_tokens, num_kv_heads, head_dim]
```

`req_to_token_pool` 给每个 req 维护一张 token → 物理槽位映射,**同一个槽位编号同时索引 K 和 V**:

```python
k_buffer[layer][slot] = K_new
v_buffer[layer][slot] = V_new
```

K/V 头数不同就要分别维护两套槽位映射、两套 page table、两套 prefix tree——**复杂度翻倍,内存碎片更糟**。

#### ④ 不对称没有任何理论好处

- 减 K 头(K 比 V 少):注意力权重分辨率降低,但 V 仍然高分辨率——**多余的 V 信息没人能精细寻址**,白白浪费
- 减 V 头(V 比 K 少):精确算注意力权重,但取出的内容粗糙——**精确寻址了一个被压扁的目标**,也没意义

最优做法就是**一致地砍掉一对 (K, V) 头**:同步降低分辨率,KV cache 体积按比例缩小,**质量损失最小**。这就是 GQA 的设计:Llama-3-8B 从 32 砍到 8,K 和 V 一起砍,从不分别处理。

### 3.2 为什么多个 Q 头共享 1 个 KV 头能奏效

直觉上很反常:不是每个头代表不同特征吗?把 4 个 Q 头硬塞到同一个 KV 头上,不会破坏多头的初衷?

下面 6 层解释 GQA 为什么有效。

#### ① 关键反直觉:训练好的 MHA 里,K、V 头本身就高度冗余

GQA 不是凭空设计——它的实证根基是:

> 跑遍训练好的 MHA 模型(GPT-2 / Llama-1 / Llama-2),研究者发现 **K、V 头之间互相非常相似**;砍掉一半 KV 头,质量几乎不变;**Q 头之间反而很不一样**(剪 Q 头损失明显)。

来自一系列 head pruning 论文(Michel et al. 2019 *Are Sixteen Heads Really Better than One?*,GQA 原论文 Ainslie et al. 2023)。结论:

> MHA 里有 32 个 KV 头,但实际"独立工作"的可能只有 6~10 个,其他都是冗余「同款」——既然如此,何必占 32 份 KV cache?

GQA 把这个观察工程化:把冗余的 KV 头合并成共享的少数几个。

#### ② Q 和 KV 的角色根本不对称

| 角色 | 干什么 | 多样性需求 |
|---|---|---|
| **Q(查询)** | **主动**提问:「我现在需要什么样的信息」 | **高**——不同上下文要问不同问题 |
| **K(索引)** | **被动**提供标签:「我能被什么样的查询命中」 | **低**——同一个 token 的"标签"是固定的几个面 |
| **V(内容)** | **被动**提供内容:「命中后我交付什么」 | **低**——同一个 token 的"实际内容"也固定 |

**图书馆类比**:

```
Q(读者的问题):每个人来问的不一样
                "我想找罗马史"
                "我想找编程书"
                "我想找小说"
                "我想找食谱"
                ↓
K(每本书的标签):书架上的索引卡(固定的)
                每本书有一组主题词、作者、年份等
                ↓
V(书的内容):每本书的内容(固定的)
```

不同读者(Q 头)问完全不同的问题,但他们**用同一套索引卡(K)、读同一批书(V)**——你不会给每个读者准备一份独立的图书馆。**索引和内容只需要一份足够丰富,就能满足很多种查询**。

#### ③ 数学视角:128 维的 K 足够支持多种 Q 提问

GQA Llama-3 里 head_dim = 128。一个 KV 头的 k 向量是 **128 维的**——一个**很丰富的子空间**。

考虑 4 个共享同一个 K 头 `k_g ∈ ℝ^128` 的 Q 头:

```
score_0 = q_0 · k_g    (q_0 强调 k_g 的某些维度,如 dim 0-31)
score_1 = q_1 · k_g    (q_1 强调另一些维度,如 dim 32-63)
score_2 = q_2 · k_g    (q_2 强调 dim 64-95)
score_3 = q_3 · k_g    (q_3 强调 dim 96-127)
```

每个 q_i 是 128 维向量,它可以**学着只关注 k_g 的某些维度**。**关键洞察**:虽然 4 个 Q 头看的是同一个 k_g,但**它们各自只"看"k_g 的不同子区域**,内积出来的 score_0 / score_1 / score_2 / score_3 **可以差别很大**。V 同理。

> **128 维空间里能写下很多不同的"标签"**,每个 Q 头能从这 128 维里提取它感兴趣的部分。**信息容量足够,不需要每个 Q 头配一个专属 K**。

#### ④ 训练时模型自动学会利用共享性

GQA 不是"随便砍 KV 头然后凑合用"。**重点是模型在训练阶段就被告知"你只有这么多 KV 头,自己想办法"**——梯度会驱动:

1. **K_g 学成"通用索引"**:让 4 个 Q 头都能各取所需,k_g 不同维度对不同 Q 头编码不同信息
2. **每个 Q_i 学会"只读 K_g 的特定维度"**:形成自然的子空间分工
3. **V_g 学成"复合内容载体"**:每个 V_g 维度承载多种语义,被不同 Q 头按权重提取

这套适应是**端到端梯度学出来的**,不是手工设计——所以效果好。

#### ⑤ 极限实证:GQA 是甜点,MQA(group 拉满)就崩了

| 设计 | num_kv_heads | 质量 | KV cache 显存 |
|---|---|---|---|
| **MHA** | 32(=Q 头数) | 100% | 100% |
| **GQA-4**(每 4 Q 共享 1 KV) | 8 | **99%** | 25% |
| **GQA-8**(每 8 Q 共享 1 KV) | 4 | 97% | 12.5% |
| **MQA**(全部共享 1 个) | 1 | **~85-92%** | 3% |

**两端现象**:

- **MHA 那端**:32 个 KV 头里大部分冗余,全备份不划算
- **GQA 4~8 那段**:质量几乎不掉,KV cache 缩小 4-8 倍——**最佳 trade-off**
- **MQA 那端**:1 个 KV 头不够装下所有 Q 头需要的索引信息,**硬塞会丢质量**

这条曲线告诉我们:**KV 的"信息容量"有底线,但底线远低于"每个 Q 头一份"**。Llama-3、Qwen、DeepSeek 都选 group=4 或 8,踩在甜点上。

#### ⑥ 重新审视直觉

「不同的 QKV 代表不同的特征,一个 Q 去询问不同特征的 KV,为什么有效?」

**重新理解**:
- "不同的 Q 头代表不同的特征"——✅ 对,Q 头需要多样性
- "不同的 KV 头代表不同的特征"——⚠️ MHA 里**理论上**对,但**训练好的 MHA 里实际上 K、V 大量冗余**;GQA 利用了这个冗余
- "1 个 Q 询问不同 KV"——表述在 GQA 里不准确;**真实结构是 1 个共享 KV 头同时被 4 个不同 Q 头查询**,共享 KV 头自己带有足够丰富的"特征面",每个 Q 头各取所需

> **正确直觉**:KV 头不是"几个独立特征坐标轴",而是"高维的复合标签 + 复合内容";只需要少数几个 KV 头就能编码所有 Q 头需要的"特征面",模型在训练时自动找到最优分工。

#### ⑦ 一句话总结

> **多 Q 共享 1 KV 之所以奏效,是因为 K、V 是被动的"高维通用索引/内容",信息容量远超每个 Q 头需要提取的部分**;训练时不同 Q 头自然学会从同一个 K、V 的不同子空间里取自己需要的信息;实证上"少数 KV 头"就足够支撑"很多种 Q 查询",这是 GQA 的工程根基。

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

### 7.1 GQA 下 TP 数 / Q 头数 / KV 头数的硬约束

把代码里的 assert 总结成一句话:

> **Q 头数必须被 tp_size 整除**;**KV 头数和 tp_size 之间必须有整除关系**(谁是大数无所谓,但必须能整除)。

合法 / 非法举例(Llama-3-8B,32 Q 头 / 8 KV 头,GQA group=4):

| TP 大小 | 本卡 Q 头 | 本卡 KV 头 | KV 是否复制 | 每个 KV 头服务几个本卡 Q 头 |
|---|---|---|---|---|
| 1 | 32 | 8 | 否 | 4(group 完整保留) |
| 2 | 16 | 4 | 否 | 4 |
| 4 | 8 | 2 | 否 | 4 |
| 8 | 4 | **1** | 否 | 4 |
| 16 | 2 | 1 | **是,每 2 卡复制一份** | 2(原 group 被切成 2 半) |
| 32 | 1 | 1 | **是,每 4 卡复制一份** | 1 |
| **3 / 5 / 6 / 7** | — | — | — | **❌ 非法**(32 % 3 ≠ 0;8 与 5 互不整除) |

### 7.2 核心规律:GQA group 不能被 TP 切碎

GQA 的"分组" `group_size = num_heads / num_kv_heads`(8B 上是 4)。**TP 切分必须沿 group 边界切**,有两种合法形态:

- **TP ≤ num_kv_heads**:每张卡拿到完整的若干 group(group 不被打散)。共享 KV 头的 Q 头**全在同一张卡**,attention 完全本地。
- **TP > num_kv_heads**:**一个 group 被切到多张卡**,这几张卡各自复制同一个 KV 头,各拿这个 group 的不同 Q 头分片。

**为什么不能跨卡?** 因为 attention 是 `Q @ Kᵀ`——如果 Q 和它对应的 KV 不在同一张卡,每次都要跨卡通信取 KV,**带宽爆炸,性能崩盘**。复制 KV 头是"用少量显存冗余" 换 "完全本地化的 attention"。

### 7.3 头数为什么这样设计

- num_heads 选 2 的幂或带很多因子的数(32 / 64 / 96),**为的是给 TP 留出 ≥4 种合法尺度**(TP=1/2/4/8/...)
- num_kv_heads 通常选 num_heads 的因子,且 **≥ 常用 TP 尺度**(典型 8 或 16),让大多数部署落在"情况 A、KV 不复制" 的便宜路径
- 极端例:Llama-3-70B 是 64 Q / 8 KV(group=8),只有 TP=16/32/64 才会触发 KV 复制

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
