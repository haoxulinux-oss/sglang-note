# `LlamaDecoderLayer.forward()` 总览(初学者向)

> 📖 **关联阅读**:
> - 上游:[`ModelRunner.forward()` 方法详解](model-runner-forward.zh.md) §5.2 在 `LlamaModel.forward` 里循环调用 `LlamaDecoderLayer.forward`
> - 本系列后续 6 篇:RMSNorm+残差、Self-Attention 总览、QKV+GQA、RoPE、Attention+KV cache、MLP+SwiGLU

代码位置:`python/sglang/srt/models/llama.py:303`

---

## 一 这个函数为什么是核心

LLM 推理 99% 的 GPU 时间消耗在这里。一个 Llama-7B 有 32 层这样的 `LlamaDecoderLayer`,Llama-70B 有 80 层。**每过一层,hidden_states 就被"加工"一次,但形状不变**(`[total_tokens, hidden_size]`)。

源码全文(只有 23 行,但每行都有故事):

```python
def forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    forward_batch: ForwardBatch,
    residual: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Self Attention
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
    )

    # Fully Connected
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

---

## 二 一层 decoder 在做什么(用人话讲)

每层 decoder 干两件事:

1. **Self-Attention 子层**:让每个 token 看一眼"自己之前的所有 token",根据相关性更新自己。**作用**:让"她"知道前面提过"小明",所以"她"=「小明的女朋友」。
2. **MLP 子层**(也叫 Feed-Forward):每个 token 独立地、并行地、过一个两层的小 MLP。**作用**:在更高维空间里做"特征变换",把 attention 收集来的信息再加工。

每个子层前都套一个 **RMSNorm**(归一化幅度,稳数值),每个子层外都包一个 **残差连接**(`output = input + sublayer(norm(input))`,让深网络梯度好流)。

经典流程图:

```
输入 hidden_states                      ┌──→ residual ─────────────┐
       │                                │                          │
       ↓                                │                          ↓
  input_layernorm   ──→  self_attn  ────┘  ──→  + (residual add) ──┐
       │                                                           │
       ↓                                                           │
  post_attention_layernorm  ──→  mlp  ────────→  + (residual add) ─┘
                                                       │
                                                       ↓
                                               输出 hidden_states
```

这就是经典的 **Pre-Norm Transformer Block**(GPT-2/Llama/Qwen 都是这个架构)。

---

## 三 4 个子模块在 `__init__` 里的样子

```python
self.self_attn = LlamaAttention(...)                     # 自注意力(含 qkv_proj / RoPE / RadixAttention / o_proj)
self.mlp       = LlamaMLP(...)                           # FFN(含 gate_up_proj / SiLU·Mul / down_proj)
self.input_layernorm           = RMSNorm(hidden_size, eps=...)   # 自注意力前 RMSNorm
self.post_attention_layernorm  = RMSNorm(hidden_size, eps=...)   # MLP 前 RMSNorm
```

| 子模块 | 含什么 | 详细解析 |
|---|---|---|
| `self.input_layernorm` | RMSNorm 一层 | [RMSNorm + 残差透传](llama-rmsnorm-and-residual.zh.md) |
| `self.self_attn` | qkv_proj / RoPE / RadixAttention / o_proj | [Self-Attention 总览](llama-self-attention-overview.zh.md) → [QKV + GQA](llama-qkv-projection-gqa.zh.md) → [RoPE](llama-rope.zh.md) → [Attention + KV cache](llama-attention-and-kvcache.zh.md) |
| `self.post_attention_layernorm` | RMSNorm 一层 | [RMSNorm + 残差透传](llama-rmsnorm-and-residual.zh.md) |
| `self.mlp` | gate_up_proj / SiLU·Mul / down_proj | [MLP / SwiGLU](llama-mlp-swiglu.zh.md) |

---

## 四 `forward()` 的执行流(逐行翻译)

```python
# === 第 1 块:Self-Attention 入口 ===
if residual is None:
    # 第一层(没人传 residual 进来),手动建 residual
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
else:
    # 中间层:用 fused kernel 一次完成 (hidden + residual) + RMSNorm,并把"hidden+residual"作为新的 residual 传出去
    hidden_states, residual = self.input_layernorm(hidden_states, residual)

hidden_states = self.self_attn(positions, hidden_states, forward_batch)
# ↑ 这一步内部:
#   ① qkv = qkv_proj(hidden_states)
#   ② q, k, v = qkv.split(...)
#   ③ q, k = rotary_emb(positions, q, k)         ← RoPE
#   ④ attn_out = RadixAttention(q, k, v, forward_batch)   ← 写 KV cache + 算 attention
#   ⑤ out = o_proj(attn_out)
#   返回的是子层"输出",还没加 residual

# === 第 2 块:MLP 入口 ===
hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
# ↑ 同样:fused kernel 把"hidden + residual"当 norm 输入,并把那个 sum 作为新 residual

hidden_states = self.mlp(hidden_states)
# ↑ 内部:
#   ① gate_up = gate_up_proj(hidden_states)            ← fused gate + up 投影到 2 * intermediate_size
#   ② x = SiluAndMul(gate_up)                          ← SiLU(gate) ⊙ up,变成 intermediate_size
#   ③ x = down_proj(x)                                  ← 投回 hidden_size

return hidden_states, residual
# ↑ 注意:residual add 不在本层加!残差累加被推迟到"下一层的 layernorm" 或者"最后 self.norm"里完成
```

**关键洞察**:这个 forward 结尾返回的 `hidden_states` 是"MLP 的输出",`residual` 是"还没加上去的累计残差"。**真正的 `output = mlp_out + residual`** 是被推迟到下一层 `input_layernorm(hidden, residual)` 内部、或最末端 `LlamaModel.norm(hidden, residual)` 内部完成的。这是 SGLang 「fused add+rmsnorm」 的核心思想——**不要白白多读写一次 hidden 张量**。

---

## 五 形状对照(以 Llama-7B 为例,batch_size=1, prefill 长度=128)

| 张量 | 形状 | 说明 |
|---|---|---|
| `hidden_states` 入口 | `[128, 4096]` | 128 个 token、hidden=4096 |
| RMSNorm 后 | `[128, 4096]` | 形状不变 |
| `qkv_proj` 后 | `[128, 4096+1024+1024]` | Q 头 32 个 × 128,K/V 各 8 头 × 128(GQA) |
| q, k, v split 后 | q `[128, 4096]`, k/v `[128, 1024]` | 拆开 |
| RoPE 后 | 同上 | 形状不变,数值改变 |
| `RadixAttention` 输出 | `[128, 4096]` | 多头合并 |
| `o_proj` 后 | `[128, 4096]` | 投回 hidden |
| `gate_up_proj` 后 | `[128, 11008+11008]` | intermediate=11008 |
| `SiluAndMul` 后 | `[128, 11008]` | gate ⊙ up |
| `down_proj` 后 | `[128, 4096]` | 投回 hidden |
| 出口 `hidden_states` | `[128, 4096]` | 形状全程不变 |

---

## 六 阅读顺序建议

按下面顺序读,一篇接一篇平滑上升:

1. **本篇** — 总览
2. [RMSNorm + 残差透传](llama-rmsnorm-and-residual.zh.md) — 先把外层"皮"拆透
3. [Self-Attention 总览](llama-self-attention-overview.zh.md) — 进入 attention 子层骨架
4. [QKV Projection + GQA](llama-qkv-projection-gqa.zh.md) — q/k/v 怎么算、为什么 K/V 头少
5. [RoPE 旋转位置编码](llama-rope.zh.md) — 位置信息如何注入
6. [Attention + KV cache](llama-attention-and-kvcache.zh.md) — RadixAttention,prefill 和 decode 走法
7. [MLP / SwiGLU](llama-mlp-swiglu.zh.md) — FFN 子层

---

## 七 一句话总结

> **`LlamaDecoderLayer.forward` = pre-norm + self-attention + pre-norm + MLP**;每个子层前先 RMSNorm,残差以"参数透传"形式延迟相加,合并到下一次 fused add+norm 里跑,省掉一次显存读写。理解这一层等于理解整个 Llama/Qwen/DeepSeek 的推理骨架。
