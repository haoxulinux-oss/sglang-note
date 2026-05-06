# RoPE 旋转位置编码(初学者向)

> 📖 **关联阅读**:
> - 父篇:[Self-Attention 总览](llama-self-attention-overview.zh.md)
> - 上一篇:[QKV + GQA](llama-qkv-projection-gqa.zh.md)
> - 下一篇:[Attention + KV cache](llama-attention-and-kvcache.zh.md)

代码位置:
- 工厂函数 `get_rope`:`python/sglang/srt/layers/rotary_embedding/factory.py`
- 基类 `RotaryEmbedding`:`python/sglang/srt/layers/rotary_embedding/base.py:50`
- `forward_native`:`base.py:205`
- 调用点:`python/sglang/srt/models/llama.py:203`(`q, k = self.rotary_emb(positions, q, k)`)

---

## 一 为什么 attention 需要"位置信息"

`softmax(QKᵀ)V` 对位置是**对称的**:把句子里的两个 token 调换顺序,attention 结果不变。但语言显然不是对称的——「猫追狗」和「狗追猫」意思相反。所以必须把"我是第几个 token"塞进 attention 计算里。

历史上有三种主流做法:

| 方法 | 怎么做 | 用在哪 |
|---|---|---|
| **绝对位置编码** | 给每个位置查一个 embedding 向量,加到 token embedding 上 | 原始 Transformer / GPT-2 |
| **相对位置 bias** | attention 算完后加一个 (i-j) 偏置 | T5 / ALiBi |
| **RoPE(旋转位置编码)** | **把 q、k 向量"旋转"一个角度,角度由位置决定** | Llama / Qwen / DeepSeek |

RoPE 的好处:
1. **天然相对**:`q_m · k_n` 的结果只依赖 `m - n`(相对距离)
2. **不增参数**:旋转矩阵是确定的,没有可学习参数
3. **可外推**:训练时见过 4k,推理时拉到 8k / 32k 也能工作(配合 NTK-aware / YaRN 等技巧)

---

## 二 RoPE 的数学骨架

### 2.1 核心思想

把 hidden 里每两个相邻维度看成一个"复数对" `(x_{2i}, x_{2i+1})` ↔ `x_{2i} + i·x_{2i+1}`,**用一个旋转角度 `θ_i × position` 旋转这个复数**:

```
[x_{2i}']     [cos(mθ_i)   -sin(mθ_i)] [x_{2i}]
[x_{2i+1}'] = [sin(mθ_i)    cos(mθ_i)] [x_{2i+1}]
```

其中:
- `m` = 这个 token 在序列里的位置(0, 1, 2, ...)
- `θ_i = base^(-2i/d)` ,`d` = `rotary_dim`,`base` 通常 10000
- 不同维度对用**不同角速度**:低维转得快(高频),高维转得慢(低频)——多频段共同表达位置

### 2.2 相对性证明(凭啥说"只依赖 m-n")

旋转矩阵 `R(θ)` 满足 `R(α) · R(β)ᵀ = R(α-β)`。所以:

```
q'_m · k'_n = (R(mθ) q_m) · (R(nθ) k_n)
             = q_mᵀ R(mθ)ᵀ R(nθ) k_n
             = q_mᵀ R((n-m)θ) k_n
```

最终结果**只取决于位置差** `n - m`。这就是"相对位置编码"四个字的由来。

---

## 三 SGLang 里的实现

### 3.1 启动期:预计算 cos/sin 缓存

`RotaryEmbedding.__init__` 调用 `_compute_cos_sin_cache()`(`base.py:140`):

```python
def _compute_inv_freq(self, base):
    # inv_freq[i] = 1 / (base ** (2i/rotary_dim))         ← θ_i
    return 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))

def _compute_cos_sin_cache(self):
    inv_freq = self._compute_inv_freq(self.base)            # [rotary_dim/2]
    t = torch.arange(self.max_position_embeddings).float()  # [max_pos]
    freqs = torch.einsum("i,j -> ij", t, inv_freq)          # [max_pos, rotary_dim/2]
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)                   # [max_pos, rotary_dim]
    return cache
```

最终 `self.cos_sin_cache` 形状 `[max_position_embeddings, rotary_dim]`,**前一半是 cos,后一半是 sin**。Llama-3 默认 `max_position=8192`、`rotary_dim=128`,缓存就是 `[8192, 128]` 的张量,**所有层共享**。

如果运行时序列超长,`_ensure_cos_sin_cache_length`(`base.py:151`)会增量扩展缓存。

### 3.2 运行期:`forward_native`(`base.py:205`)

```python
def forward_native(self, positions, query, key, offsets=None, ...):
    if offsets is not None:
        positions = positions + offsets

    positions = positions.flatten()
    num_tokens = positions.shape[0]

    cos_sin = self.cos_sin_cache.index_select(0, positions)  # [N, rotary_dim] ← 按位置查表
    cos, sin = cos_sin.chunk(2, dim=-1)                      # 各 [N, rotary_dim/2]

    # ★ 对 q 做旋转
    query = query.view(num_tokens, -1, self.head_size)        # [N, num_heads, head_dim]
    query_rot  = query[..., : self.rotary_dim]                # 前 rotary_dim 维要旋转
    query_pass = query[..., self.rotary_dim :]                # 后面剩余维度直接 pass
    query_rot  = self._apply_rotary_emb_wrapped(query_rot, cos, sin, self.is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    # ★ 对 k 做同样的旋转
    ... (代码结构和 q 完全一致)
    return query, key
```

**关键点 4 个**:

1. **`cos_sin_cache.index_select(0, positions)`**:按每个 token 的 `position` 查出对应的 cos / sin 向量。这就是"根据位置 `m` 算 `cos(mθ)`、`sin(mθ)`"的查表实现。
2. **`query[..., : self.rotary_dim]`**:有些模型只对前 `rotary_dim` 维做 RoPE(`partial_rotary_factor < 1`),后面维度保持原样。Llama 系是 100%。
3. **`_apply_rotary_emb_wrapped`**:执行 `[x_2i; x_{2i+1}] → [x_2i·cos - x_{2i+1}·sin; x_2i·sin + x_{2i+1}·cos]`,在前一半 / 后一半还是奇偶维度配对(`is_neox_style`)上有区别——Llama 用 NeoX 风格(前一半/后一半配对)。
4. **`v` 不参与 RoPE**——这是核心设计:位置信息只影响"相似度计算"(`Q·Kᵀ`),不影响"内容"(`V`)。

### 3.3 调用点(`llama.py:203`)

```python
def forward_prepare_native(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)        # ← RoPE
    return q, k, v
```

`q, k` 形状不变,数值变(被旋转了)。`v` 原封不动透传。

---

## 四 RoPE 和 KV cache 的相互关系(关键!)

**写进 KV cache 的 K 是已经旋转过的 K**(因为 `q, k = self.rotary_emb(...)` 在 `RadixAttention(q, k, v, ...)` 之前)。所以:

- 第 1 步:prefill 时,token i 的 K 被旋转了 `i × θ`,然后写进 cache
- 第 N 步:decode 出第 j 个 token,它的 q 被旋转 `j × θ`,**直接和 cache 里历史 K 相乘**——历史 K 已经"自带位置"了,不需要重做 RoPE

**好处**:KV cache 里的 K 是"位置已 baked-in"的版本,decode 时不需要重新旋转,**几十亿次 cache 读不浪费一个浮点运算**在 RoPE 上。

> **小陷阱**:如果你做"continuous batching",请求中途插入,**新请求的位置 0 必须从 cache 起始位置算**——SGLang 的 `req_to_token_pool` 会确保每个请求有自己的 position 空间。

---

## 五 长序列外推 / NTK-aware / YaRN(简介)

朴素 RoPE 在训练长度 4k 之外会"分布漂移",效果掉。常见 fix:

| 技巧 | 思路 | 在 SGLang |
|---|---|---|
| **Linear scaling** | 把 position 缩放成 `position / scale`,等于"把所有频率乘一个小系数" | rope_scaling type=linear |
| **NTK-aware** | 调整 `base`,让低频转得更慢,高频不变 | rope_scaling type=dynamic |
| **YaRN** | 分段重算,低/中/高频差异化处理 | `rotary_embedding/yarn.py` |
| **LongRoPE / Phi-3** | 训练阶段就用一组扩展频率 | rope_scaling 字段从 hf_config 来 |

`get_rope` 工厂函数会根据 hf_config 里的 `rope_scaling` 字段挑出对应实现。**初学者第一遍读到 RoPE 知道"它把位置塞进 q、k"** 就够了,长序列细节用到再回来查。

---

## 六 一句话总结

> **RoPE = 把 Q、K 按 token 位置旋转一个特定角度**,使 `Q·Kᵀ` 自动变成"只依赖相对位置"的形式;启动期把 cos/sin 预算成大小为 `[max_pos, rotary_dim]` 的查表,运行期 `index_select` + 一次旋转就完事;**V 不旋转**;旋转后的 K 直接写进 KV cache,decode 时直接用,不重做。
