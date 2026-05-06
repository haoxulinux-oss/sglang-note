# Self-Attention 总览:`LlamaAttention.forward()`(初学者向)

> 📖 **关联阅读**:
> - 父篇:[`LlamaDecoderLayer.forward()` 总览](llama-decoder-layer-overview.zh.md)
> - 上一篇:[RMSNorm + 残差透传](llama-rmsnorm-and-residual.zh.md)
> - 后续:[QKV + GQA](llama-qkv-projection-gqa.zh.md) → [RoPE](llama-rope.zh.md) → [Attention + KV cache](llama-attention-and-kvcache.zh.md)

代码位置:`python/sglang/srt/models/llama.py:121`(类),`:220`(forward)

---

## 一 Self-Attention 在每层 decoder 里干啥

一句话:**让每个 token 看一眼"自己 + 自己之前的所有 token",根据相关性"借信息"过来**。

打个比方:你读到「她笑了」,模型要决定"她"指谁——它把这个位置的 query 拿去和前面所有 token 的 key 算相似度,算出最相关的(比如前 5 句出现的"小红"),把那个位置的 value 加权混合进当前位置的 hidden,**这就完成了一次 self-attention**。

数学骨架:

```
attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V
```

但在 decoder 里有两个关键限制:
1. **causal mask**:位置 i 只能看 0..i,不能看未来
2. **KV cache**:K、V 是历史信息,要存下来,**避免每次重算**

---

## 二 `LlamaAttention.forward()` 的 4 步骨架

源码全文(简化掉 NPU 分支):

```python
def forward(self, positions, hidden_states, forward_batch):
    q, k, v = self.forward_prepare_native(positions, hidden_states)   # ① + ② + ③
    attn_output = self.attn(q, k, v, forward_batch)                   # ④
    output, _ = self.o_proj(attn_output)                              # ⑤
    return output

def forward_prepare_native(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)                             # ① QKV 投影(融合)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)   # ② 拆开
    q, k = self.rotary_emb(positions, q, k)                            # ③ RoPE 旋转位置编码
    return q, k, v
```

**5 步流水**:

| 步骤 | 干啥 | 形状变化(以 Llama-7B 为例) | 详细解析 |
|---|---|---|---|
| ① **qkv_proj** | 把 hidden 投影成 Q、K、V(融合成一个大 GEMM) | `[N, 4096]` → `[N, 4096+1024+1024]` | [QKV + GQA](llama-qkv-projection-gqa.zh.md) |
| ② **split** | 把那个大张量按 q_size/kv_size/kv_size 切成 q、k、v 三块 | 见上表 | 同上 |
| ③ **RoPE** | 把位置信息以"旋转"形式注入到 q、k(v 不变) | 形状不变,数值变 | [RoPE 旋转位置编码](llama-rope.zh.md) |
| ④ **RadixAttention** | 写 KV cache + 算 softmax(QKᵀ/√d)V | `[N, 4096]` | [Attention + KV cache](llama-attention-and-kvcache.zh.md) |
| ⑤ **o_proj** | 多头合并后投回 hidden 维 | `[N, 4096]` → `[N, 4096]` | 本文末附简介 |

> **N** = `total_tokens`(prefill 时是 prompt 总长,decode 时是 batch_size)

---

## 三 `__init__` 里挂了哪些子模块

```python
self.qkv_proj   = QKVParallelLinear(...)         # 把 q/k/v 三个投影合并成一次 GEMM
self.o_proj     = RowParallelLinear(...)         # 输出投影
self.rotary_emb = get_rope(...)                  # RoPE 旋转矩阵的 cos/sin 缓存
self.attn       = RadixAttention(...)            # 真正算 softmax(QKᵀ)V 的统一壳,底下接 attn_backend
```

加上几个常量:

| 字段 | 含义 |
|---|---|
| `self.num_heads` | 本进程负责的 Q 头数(全局 Q 头数 / TP 大小) |
| `self.num_kv_heads` | 本进程负责的 K/V 头数(GQA 通常远少于 Q) |
| `self.head_dim` | 每个头的维度 |
| `self.q_size` | `num_heads * head_dim` |
| `self.kv_size` | `num_kv_heads * head_dim` |
| `self.scaling` | `head_dim ** -0.5`,attention 公式里的 `1/√d_k` |

**关键事实**(展开见 [QKV + GQA](llama-qkv-projection-gqa.zh.md)):
- Q 头数和 K/V 头数**可以不一样**(Grouped-Query Attention,GQA)
- TP 切分:`num_heads = total_num_heads / tp_size`,K/V 头数 < TP 时会被复制

---

## 四 ⑤ `o_proj` 在做啥

`o_proj` = output projection。多头 attention 的输出是把 `num_heads` 个头(每个 `head_dim` 维)拼回一起,得到 `[N, num_heads × head_dim]` —— 这其实就是 hidden_size。但**每个头是独立算出来的**,各头之间需要一次"信息混合 + 再投影",这就是 `o_proj`:

```python
self.o_proj = RowParallelLinear(
    self.total_num_heads * self.head_dim,    # in:  hidden
    hidden_size,                              # out: hidden
    bias=bias, ...
)
```

**为什么是 `RowParallelLinear`**?TP 下,attention 的 Q/K/V 是按"头"切到不同 GPU 上的(列并行),attention 算出来后,**每张卡只有自己那部分头的输出**——形状 `[N, hidden / tp_size]`。`RowParallelLinear` 在做 GEMM 后会触发一次 **all-reduce**,把所有 GPU 的输出加起来,得到完整的 `[N, hidden]`。

> 一般初学者只需要知道"`o_proj` 把多头输出投回 hidden_size,TP 时还顺便 all-reduce 一次"。

---

## 五 整层 self-attention 的形状追踪(Llama-7B,TP=1,prefill 128 tokens)

```
hidden_states                    [128, 4096]
        │
        ↓ qkv_proj
qkv                              [128, 4096+1024+1024]   ← 融合的 GEMM
        │
        ↓ split
q                                [128, 4096]             ← 32 个 Q 头 × 128
k                                [128, 1024]             ← 8 个 K 头 × 128(GQA)
v                                [128, 1024]
        │
        ↓ rotary_emb (RoPE 只对 q, k)
q', k'                           形状不变,数值变
        │
        ↓ RadixAttention(q', k', v)
        │   ├─ 写 K、V 到 token_to_kv_pool 对应位置
        │   └─ paged attention 算 softmax(qKᵀ/√d)V
attn_output                      [128, 4096]            ← 多头合并
        │
        ↓ o_proj
output                           [128, 4096]
```

---

## 六 prefill vs decode 时的两条路径

`LlamaAttention.forward` 自己**不区分 prefill / decode**——它对每个 token 都跑一遍同样的 5 步。区别在 `RadixAttention` 内部:

| 阶段 | 输入 token 数 | KV cache 行为 | attention 算法 |
|---|---|---|---|
| **prefill** | 整段 prompt(几百~几千) | 把整段 K、V 写进去 | 计算每个 query 对全段 K 的 attention |
| **decode** | 1 个 token | 写 1 行 K、V | 当前 query 对历史所有 K 的 attention |

详见 [Attention + KV cache](llama-attention-and-kvcache.zh.md)。

---

## 七 一句话总结

> **`LlamaAttention.forward` = qkv_proj(融合 GEMM) + split + RoPE + RadixAttention + o_proj**;5 步串起来就是「把 hidden 算成 Q/K/V,加上位置信息,算 softmax(QKᵀ)V,再投回 hidden 维」。Q/K 头数可以不一样(GQA),TP 时 o_proj 会 all-reduce 合并多卡结果。
