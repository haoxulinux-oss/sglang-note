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

下面把这个函数**逐块**拆开,每一步说清楚"做了什么、为什么、形状变化、得到的张量长什么样"。先以具体例子打底:

> **运行例**:Llama-3-8B、prefill 一段 128 个 token、本卡 32 个 Q 头(没开 TP)、`head_dim=128`、`rotary_dim=128`(整个 head 都参与 RoPE)。

输入张量形状:
- `positions`: `[128]` —— 每个 token 的位置编号 [0, 1, 2, ..., 127]
- `query`: `[128, 32 × 128] = [128, 4096]` —— 32 个 Q 头拼起来
- `key`: `[128, 8 × 128] = [128, 1024]` —— GQA 8 个 KV 头(为简化先假设 32 同 Q,后面再讨论 GQA)

#### ① positions 预处理

```python
if offsets is not None:
    positions = positions + offsets        # 极少用到。多模态 / RAG 场景里某些 token 要"假装"自己在另一个位置,通过 offset 平移
positions = positions.flatten()            # 展平。如果 positions 是 [batch, seq] 二维,变成 [batch×seq] 一维
num_tokens = positions.shape[0]            # N = 总 token 数,这里 N=128
```

到此 `positions` 是 1D 张量,形状 `[N]`,值是每个 token 的绝对位置编号。

#### ② 按位置查 cos/sin 表

```python
cos_sin = self.cos_sin_cache.index_select(0, positions)
# self.cos_sin_cache 是启动期预算的 [max_pos=8192, rotary_dim=128] 张量
# index_select(dim=0, indices=positions) 按每个 token 的 position 在第 0 维抽对应的行
# 结果:cos_sin.shape = [128, 128]
# cos_sin[i] = cos_sin_cache[positions[i]] —— 第 i 个 token 拿到属于它位置的那一行
```

回想 §3.1 给出的预计算:`cos_sin_cache[m]` 的前 64 维是 `cos(m·θ_0..63)`,后 64 维是 `sin(m·θ_0..63)`。所以现在 `cos_sin[i]` 就是「第 i 个 token 对应位置 m 的 cos 和 sin 数值集合」。

**举例**:如果第 5 个 token 的 position=5,它就拿到 `cos_sin_cache[5]`,里面装着 `cos(5·θ_0), cos(5·θ_1), ..., cos(5·θ_63), sin(5·θ_0), ..., sin(5·θ_63)`。

#### ③ 拆成 cos 和 sin

```python
cos, sin = cos_sin.chunk(2, dim=-1)
# chunk(2, dim=-1) 沿最后一维平均切两半
# cos.shape = [128, 64]   ← 前一半 (rotary_dim/2 维)
# sin.shape = [128, 64]   ← 后一半 (rotary_dim/2 维)
```

`cos[i] = [cos(m·θ_0), cos(m·θ_1), ..., cos(m·θ_63)]`,`sin[i]` 同理。**每个 token 现在有 64 对 (cos, sin) 角度**,等会儿用来旋转 q 的 64 个维度对。

#### ④ 把 query 整成 [N, num_heads, head_dim] 形状

```python
query_shape = query.shape                           # 记下原始形状 [128, 4096],等会儿恢复用
query = query.view(num_tokens, -1, self.head_size)
# view 不复制,只改 stride/shape
# [128, 4096] → [128, 32, 128]
#               ↑    ↑    ↑
#              N 个  32   128
#             token  头   每个头的维度
```

为什么要 reshape?**因为 RoPE 是按"每个头"独立做旋转的**——每个头的 128 维向量内部进行旋转,头之间互相不影响。

#### ⑤ 切出 rotary 部分和 pass-through 部分

```python
query_rot  = query[..., : self.rotary_dim]          # 前 rotary_dim 维参与旋转
query_pass = query[..., self.rotary_dim :]          # 后面剩余维度不旋转,直接透传
```

形状:
- `self.rotary_dim` 一般等于 `head_dim`(Llama=128,**全部参与旋转**)
- 但有些模型 `partial_rotary_factor < 1`,比如 Phi-3 是 0.4,这时只前 `0.4 × head_dim = 51` 维做 RoPE,后 77 维保持原样
- Llama 默认 `partial_rotary_factor=1`,所以 `query_rot.shape = [128, 32, 128]`,`query_pass.shape = [128, 32, 0]`(空)

#### ⑥ 真正做旋转:`_apply_rotary_emb_wrapped`

```python
query_rot = self._apply_rotary_emb_wrapped(
    query_rot, cos, sin, self.is_neox_style
)
```

这个函数源码在 `rotary_embedding/utils.py:36`(`apply_rotary_emb`),核心 6 行:

```python
def apply_rotary_emb(x, cos, sin, is_neox_style):
    cos = cos.unsqueeze(-2).to(x.dtype)            # [N, 64] → [N, 1, 64]   广播到 head 维
    sin = sin.unsqueeze(-2).to(x.dtype)            # 同上

    if is_neox_style:                              # Llama / NeoX 风格(前一半 / 后一半配对)
        x1, x2 = torch.chunk(x, 2, dim=-1)         # x.shape=[N, 32, 128] → x1=x2=[N, 32, 64]
                                                    # x1 是 x 的前 64 维,x2 是后 64 维
    else:                                           # GPT-J 风格(奇偶维度配对)
        x1 = x[..., ::2]                           # 偶数维度
        x2 = x[..., 1::2]                          # 奇数维度

    o1 = x1 * cos - x2 * sin                       # ★ 旋转矩阵第 1 行
    o2 = x2 * cos + x1 * sin                       # ★ 旋转矩阵第 2 行

    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)         # 前一半放 o1,后一半放 o2
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)   # 偶数位放 o1,奇数位放 o2
```

**这两行就是 RoPE 的全部数学**:

```
o1 = x1 * cos - x2 * sin
o2 = x2 * cos + x1 * sin
```

它对应的旋转矩阵公式(每个维度对 (x1[i], x2[i]) 看成复数 x1[i] + i·x2[i],乘以 e^(iθ)):

```
[o1[i]]     [cos(mθ_i)   -sin(mθ_i)] [x1[i]]
[o2[i]]  =  [sin(mθ_i)    cos(mθ_i)] [x2[i]]
```

**Llama 用 NeoX style 配对**——前 64 维 `x1` 和后 64 维 `x2` 一一配对成 64 个复数对:`(x[0], x[64]), (x[1], x[65]), ..., (x[63], x[127])`。注意**不是 (x[0], x[1]) 这种相邻配对**——这是 NeoX vs GPT-J 的关键区别。

> NeoX vs GPT-J 在数学上等价(都是同样的旋转),只是**内存布局不同**——直接影响 GPU 上 attention kernel 的访存模式。Llama 选 NeoX 是历史原因(从 GPT-NeoX 衍生而来)。

旋转后 `query_rot.shape = [128, 32, 128]`,**形状不变,数值变了**。

#### ⑦ 拼回完整 query 并恢复原形状

```python
query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)
# (query_rot, query_pass) 在最后一维拼接 → [128, 32, 128]
# .reshape([128, 4096])  ← 恢复原始 [N, hidden] 形状
```

现在 `query` 形状回到 `[128, 4096]`,但**前一半内容是被 RoPE 旋转过的版本**(对 Llama 来说是全部内容)。

#### ⑧ 对 key 做完全相同的处理

```python
key_shape = key.shape                              # [128, 1024]
key       = key.view(num_tokens, -1, self.head_size)  # [128, 8, 128] (GQA: 8 个 KV 头)
key_rot   = key[..., : self.rotary_dim]            # [128, 8, 128]
key_pass  = key[..., self.rotary_dim :]            # [128, 8, 0]
key_rot   = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
key       = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
```

代码结构和 q 完全一致。**注意:key 用的是和 query 同一份 `cos`、`sin`**——cos/sin 跟"位置"绑,不跟"是 Q 还是 K"绑;同一个 token 位置的 q 和 k 旋转同样的角度。

#### ⑨ 返回

```python
return query, key
```

`v` 全程**没有被传入这个函数,也没有被改动**——这是 RoPE 的核心设计原则:**位置信息只影响"相似度计算"(Q·Kᵀ),不影响"内容"(V)**。

---

#### 一张表把所有形状串起来(N=128, num_q=32, num_kv=8, head_dim=128)

| 阶段 | 张量 | 形状 |
|---|---|---|
| 输入 | `query` | `[128, 4096]` |
| 输入 | `key` | `[128, 1024]` |
| 输入 | `positions` | `[128]` |
| 查表后 | `cos_sin` | `[128, 128]` |
| 拆分后 | `cos`, `sin` | `[128, 64]` 各一 |
| reshape 后 | `query` | `[128, 32, 128]` |
| reshape 后 | `key` | `[128, 8, 128]` |
| 切 rotary 后 | `query_rot` | `[128, 32, 128]`(Llama 全部参与) |
| 切 rotary 后 | `query_pass` | `[128, 32, 0]` |
| `_apply_rotary_emb_wrapped` 中 `x1, x2` | `[128, 32, 64]` 各一 | (NeoX 配对) |
| `_apply_rotary_emb_wrapped` 中 `o1, o2` | `[128, 32, 64]` 各一 | (旋转后) |
| 拼回后 | `query` | `[128, 4096]`(形状恢复) |
| 输出 | `query`, `key` | `[128, 4096]` 和 `[128, 1024]`(数值变,形状不变) |

#### 几个关键点

1. **`cos_sin_cache.index_select(0, positions)`**:按每个 token 的 `position` 查出对应的 cos / sin 行。**这就是"根据位置 m 算 cos(mθ)、sin(mθ)"的查表实现**——cos/sin 在启动期预算好,运行时只是查表。
2. **`query[..., : self.rotary_dim]`**:有些模型只对前 `rotary_dim` 维做 RoPE(`partial_rotary_factor < 1`),后面维度保持原样。Llama 系全部参与,所以 `query_pass` 是空。
3. **`_apply_rotary_emb_wrapped`** 内核就两行 `o1 = x1·cos - x2·sin; o2 = x2·cos + x1·sin`,等价于把每个维度对当作复数旋转 `mθ_i` 角度。
4. **NeoX vs GPT-J** 区别只在配对方式(前后半 vs 奇偶位),数学等价,内存布局不同。Llama 用 NeoX。
5. **q、k 用同一份 cos / sin**,因为 RoPE 跟"位置"绑,不跟"是 Q 还是 K"绑。
6. **v 不进这个函数**——位置信息只影响 Q·Kᵀ 相似度,不影响 V 内容。

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
