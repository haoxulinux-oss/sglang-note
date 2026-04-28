# `forward` 在 LLM 领域的含义

---

## 一 来源:神经网络的「前向传播」

`forward` 这个词来自神经网络的两个互补阶段:

| 阶段 | 方向 | 干什么 |
|---|---|---|
| **forward**(前向传播) | 从输入层 → 输出层 | 给定输入,**算出输出** |
| **backward**(反向传播) | 从输出层 → 输入层 | 给定 loss,反向**算出梯度**,更新权重 |

PyTorch `nn.Module` 的 API 直接固化了这个命名:

```python
class MyModel(nn.Module):
    def forward(self, x):
        return self.layers(x)

# 调用 model(x) 实际上就是调用 model.forward(x)
output = model(x)            # ← 这一次过模型,叫做「一次 forward」
loss.backward()              # ← 反向传播
```

所以「**forward**」最初的含义就是:**输入过一遍模型,得到输出**。

---

## 二 在 LLM 推理中的含义

LLM 推理**只有 forward,没有 backward**(不更新权重)。所以 LLM 领域的 `forward` 就特指:

> **把一批 token 喂进模型,跑完所有 transformer 层的计算,在最后一层拿到 logits**(或 hidden states)。

```
input_ids        ──┐
positions          │
attention metadata │   ──→  Transformer Layer 1
KV cache 索引      │   ──→  Transformer Layer 2
sampling info      │   ──→  ...
                   │   ──→  Transformer Layer N
                  ─┘   ──→  LM Head
                            │
                            ▼
                          logits   ──sample──→  next_token
```

整个这一过程,就是「**一次 forward**」(或 forward pass)。

---

## 三 LLM 推理里 forward 的颗粒度

**一次 forward = 一次 GPU 上跑完整个模型 = 产生一批 token 输出**

具体能产出多少 token、消耗多少输入,取决于 forward 的「类型」:

| forward 类型 | 输入 | 输出 |
|---|---|---|
| **Prefill forward** | 一段 prompt(N 个 token) | 1 个 next_token + KV cache 写入 N 个位置 |
| **Decode forward** | 1 个 token(上一步生成的) | 1 个 next_token + KV cache 写入 1 个位置 |
| **Spec decode 校验 forward** | k 个 draft token | k 个 logits 用于校验 |
| **Embedding forward** | 一段 prompt | 1 个 pooled embedding 向量 |

不管哪种,都是「一次 forward = 把输入完整过一遍模型」。

---

## 四 SGLang 代码里的 `forward_*`

理解了上面这个核心含义,源码里这些命名就一目了然:

| 命名 | 含义 |
|---|---|
| `model.forward(...)` | 模型本身的 forward 方法(继承自 PyTorch) |
| `forward_batch_generation(model_worker_batch)` | model worker 跑一次 forward,产 next_token |
| `forward_batch_embedding(...)` | 跑一次 forward,产 embedding |
| `ForwardBatch` | 一次 forward 所需的所有输入打包成的 dataclass |
| `forward_mode` | 这次 forward 是 prefill / decode / idle / verify ... |
| `forward_stream` | 专门跑 forward 的 CUDA stream(overlap schedule) |
| `forward_ct` | 总共跑了多少次 forward(计数器) |
| `forward_entry_time` | 这个请求进入某次 forward 的时间戳 |
| `pre_forward / post_forward` | forward 前后的 hook |
| `record_forward_metrics(batch)` | 给这次 forward 打 metrics |

---

## 五 LLM 推理里「一次 forward」≈ 一个 GPU step

工程视角看:

- **一次 forward** = 一次完整的 GPU 计算调用(matmul + attention + activation + ...)。
- scheduler 主循环 `while True` 每轮干的事:**组一个 batch → 跑一次 forward → 处理结果**。
- 所以 SGLang 的吞吐 = `forward_ct / 时间`,延迟 = 单次 forward 耗时(对 decode) + prefill forward 耗时(对 TTFT)。

---

## 六 一句话总结

> **forward** 在神经网络里指「输入过一遍模型,产出输出」的那次前向传播。LLM 推理因为只有这个方向,所以「一次 forward」直接等于「一次模型计算 = GPU 上跑完整个 transformer = 产生一批新 token」——是推理引擎的最小调度单位。SGLang 的所有 `forward_*` 命名都是围绕这个最小单位展开的。
