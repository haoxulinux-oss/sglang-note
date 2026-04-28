# Prefill 和 Decode

LLM 自回归生成把推理拆成两个性质完全不同的阶段。

---

## 一 Prefill(预填充)

**定义**:把用户输入的 prompt 一次性喂进模型,**并行**算出每个 token 的 hidden state、KV(Key/Value) 向量,缓存到 KV cache 里。最终拿到最后一个 token 的 logits,采样出**第 1 个生成 token**。

```
Prompt: "你是谁"  ──tokenize──→  [token_0, token_1, token_2]   (假设 3 个 token)

           一次 forward(并行)
                  ▼
   ┌─────────────────────────────────┐
   │ Position 0  attend [0]           │ → KV[0]
   │ Position 1  attend [0,1]         │ → KV[1]
   │ Position 2  attend [0,1,2]       │ → KV[2]   logits → sample → "我"
   └─────────────────────────────────┘
                  │
                  ▼
              KV cache 写入 KV[0..2]
```

**特性**:

| 维度 | 表现 |
|---|---|
| 一次输入 | 整段 prompt(N 个 token) |
| 一次输出 | 1 个 token |
| 计算复杂度 | O(N²)(self-attention) |
| 矩阵形状 | `[bs, N, hidden]`,N 大 → matmul 大 |
| 硬件瓶颈 | **Compute-bound**(GPU 算力) |
| 时间成本 | 跟 prompt 长度成正比 |
| 决定的指标 | **TTFT**(Time To First Token) |

---

## 二 Decode(解码)

**定义**:在 prefill 已经填好 KV cache 的基础上,**一次 forward 只生成 1 个 token**。这一个 token attend 到所有历史 KV(prefill 写入 + 之前 decode 写入),算出新的 KV 写回 cache,最后采样出下一个 token。如此循环,直到 EOS / stop / max_new_tokens。

```
现状:KV cache 里已有 KV[0..2]("你是谁") + KV[3]("我")

       第 1 个 decode step
              ▼
   只把上一个 token "我" 喂进去:
   ┌────────────────────────────────────┐
   │ Position 3  attend [0..3]          │ → KV[3]  logits → sample → "是"
   └────────────────────────────────────┘
              │
              ▼
       第 2 个 decode step
              ▼
   只把 "是" 喂进去:
   ┌────────────────────────────────────┐
   │ Position 4  attend [0..4]          │ → KV[4]  logits → sample → "Q"
   └────────────────────────────────────┘
              │
              ▼
              ...直到 EOS / max_new_tokens
```

**特性**:

| 维度 | 表现 |
|---|---|
| 一次输入 | **1 个 token**(上次生成的) |
| 一次输出 | 1 个 token |
| 计算复杂度 | O(已生成长度)——attend 已存的 KV |
| 矩阵形状 | `[bs, 1, hidden]`,矩阵很「瘦」 |
| 硬件瓶颈 | **Memory-bound**(读 KV cache + 读权重的带宽) |
| 时间成本 | 跟当前总长度成线性,但**单步几乎恒定** |
| 决定的指标 | **ITL**(Inter-Token Latency)、TPS(Tokens Per Second) |

---

## 三 一图对比

```
        prompt = N tokens                    生成 M 个 token
   ─────────────────────────────────────────────────────────────
   Prefill    │  Decode  │  Decode  │  Decode  │  ...  │  Decode
   1 次 forward │ 1 step  │ 1 step   │ 1 step   │  ...  │ 1 step
   N tokens 输入│ 1 token │ 1 token  │ 1 token  │  ...  │ 1 token
   1 token 输出 │ 1 token │ 1 token  │ 1 token  │  ...  │ 1 token
   compute-bound│      memory-bound (主导吞吐)
   决定 TTFT     │      决定 ITL 和总吞吐
```

---

## 四 KV cache 的角色

把这两个阶段拼起来的就是 KV cache:

- **Prefill 写入**:把每个 prompt token 的 K、V 算好存到 GPU 显存的一段连续区域。
- **Decode 读取**:每个 step 只算新 token 的 K、V,然后让它和**整段历史 KV**做 attention。

如果没有 KV cache,每个 decode step 都要重新跑一遍整段历史的 forward——复杂度从 O(N) 变成 O(N²),根本跑不动长序列。

---

## 五 为什么 SGLang 要区分这两个阶段

| 设计 | 动机 |
|---|---|
| `forward_mode`: `EXTEND`(prefill) vs `DECODE` | 两种 forward 用不同的 attn metadata、不同的 CUDA graph、不同的 batch shape |
| **Continuous batching** | prefill 和 decode 可以混在一个批里跑(mixed chunked prefill) |
| **Chunked prefill** | 长 prompt 的 prefill 拆成多个 chunk,避免单次 prefill 把 GPU 占太久,影响其他请求 decode 的 ITL |
| **PD disaggregation** | prefill(compute-bound) 和 decode(memory-bound) 放在不同节点,用不同硬件配比,提高整体利用率 |
| `RadixCache` 命中 | 命中前缀 → prefill 跳过这部分,只算 prompt 中没缓存的尾巴 |

---

## 六 一句话总结

> **Prefill** 是「把整段 prompt 一次性塞进模型,填好 KV cache,顺便吐出第一个生成 token」——compute-bound,决定首 token 延迟。
>
> **Decode** 是「拿上一步生成的那一个 token,attend 历史 KV,吐下一个 token」——memory-bound,决定后续生成速度。一次只一个 token,所以叫**自回归**生成。
