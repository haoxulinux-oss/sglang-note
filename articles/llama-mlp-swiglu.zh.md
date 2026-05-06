# MLP / SwiGLU(初学者向)

> 📖 **关联阅读**:
> - 父篇:[`LlamaDecoderLayer.forward()` 总览](llama-decoder-layer-overview.zh.md)
> - 上一篇:[Attention + KV cache](llama-attention-and-kvcache.zh.md)
> - 系列首篇:[`LlamaDecoderLayer` 总览](llama-decoder-layer-overview.zh.md)

代码位置:
- `LlamaMLP`:`python/sglang/srt/models/llama.py:65`
- `SiluAndMul`:`python/sglang/srt/layers/activation.py:63`

---

## 一 MLP 在 transformer 里干啥

每层 transformer 由两个子层组成:**Self-Attention(token 之间通信)+ MLP(每个 token 独立深加工)**。

- Self-Attention:让 token "互相说话",混合上下文信息
- MLP:每个 token **独立地、并行地**过一个两层 FFN,在更高维空间做"特征变换",把 attention 拉来的信息与自己的状态融合

**经验数据**:transformer 模型大约 **2/3 的参数都在 MLP**(剩下 1/3 给 attention)。所以这一步既是计算大头,也是参数大头。

---

## 二 经典 FFN vs SwiGLU

### 2.1 经典 FFN(GPT-2 / BERT 时代)

```python
y = W_out · activation(W_in · x)
```

两层 + 中间一个非线性激活(早期是 GeLU)。`W_in: hidden → intermediate`(`intermediate` 通常是 4 × hidden);`W_out: intermediate → hidden`。

### 2.2 SwiGLU(Llama / Qwen / DeepSeek 用的)

把"一个 W_in"拆成两个并列的:**gate** 和 **up**。计算:

```
gate = W_gate · x         (∈ ℝ^intermediate)
up   = W_up   · x         (∈ ℝ^intermediate)
y    = W_down · (SiLU(gate) ⊙ up)     (回到 ℝ^hidden)
```

其中:
- `SiLU(x) = x · σ(x)` (也叫 Swish-1,σ 是 sigmoid)
- `⊙` 是逐元素相乘

直觉:**`up` 是"内容",`SiLU(gate)` 是"门控权重"**——每个维度自己决定让多少内容通过。这是 **GLU(Gated Linear Unit)** 思想。

**为什么效果更好**:实证上,SwiGLU 比 GeLU/ReLU 在同样参数下 perplexity 更低。代价是参数多了 50%(因为多一个 `W_up`),所以一般会把 `intermediate_size` 设成 `8/3 × hidden_size`(而不是 4 × hidden),保持总参数不变。Llama-7B:`hidden=4096`、`intermediate=11008 ≈ 8/3 × 4096`。

---

## 三 `LlamaMLP.__init__`(`llama.py:65`)

```python
class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, ...):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(           # ① fused: gate + up 一次 GEMM
            hidden_size,
            [intermediate_size] * 2,                              # 输出维度 = 2 × intermediate
            bias=False, ...,
        )
        self.down_proj = RowParallelLinear(                       # ② down: 投回 hidden,行并行 + all-reduce
            intermediate_size,
            hidden_size,
            bias=False, ...,
        )
        if hidden_act != "silu":
            raise ValueError("Only silu is supported")
        self.act_fn = SiluAndMul()                                # ③ 激活 + 逐元素乘
```

**三个组件**:

| 组件 | 作用 | 形状 |
|---|---|---|
| `gate_up_proj` | 把 `W_gate` 和 `W_up` 融合成一个 GEMM | hidden → 2 × intermediate |
| `act_fn` (`SiluAndMul`) | 把上面输出拆两半,做 `SiLU(gate) ⊙ up` | 2 × intermediate → intermediate |
| `down_proj` | 投回 hidden 维(TP 时还触发 all-reduce) | intermediate → hidden |

---

## 四 fused gate_up_proj:为什么三合一思想再用一次

朴素写法:

```python
gate = self.gate_proj(x)      # GEMM #1
up   = self.up_proj(x)        # GEMM #2
hidden_act = SiLU(gate) * up
```

SGLang 写法:

```python
gate_up = self.gate_up_proj(x)            # 一次 GEMM,输出 [..., 2 × intermediate]
hidden_act = self.act_fn(gate_up)         # 内部 split + SiLU + Mul
```

**收益**:
- 一次大 GEMM 比两次小 GEMM 算术强度更高
- `x` 只读一次(显存带宽减半)
- 加载权重时 SGLang 通过 `stacked_params_mapping`(`llama.py:495`)把 HuggingFace 的 `gate_proj.weight` 和 `up_proj.weight` 在第一维 concat 成一个大矩阵

```python
self.stacked_params_mapping = [
    ...,
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj",   1),
]
```

---

## 五 `SiluAndMul` 是什么(`activation.py:63`)

源码看一眼:

```python
class SiluAndMul(MultiPlatformOp):
    def forward_native(self, x):                       # CPU / fallback
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]         # 拆成 gate / up 两半,SiLU(gate) ⊙ up

    def forward_cuda(self, x):                          # GPU
        d = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
        silu_and_mul(x, out)                            # ← 调 sgl_kernel 的 fused kernel
        return out
```

**两件事在一个 kernel 里完成**:
1. 把输入按最后一维平均拆成两半:`gate = x[..., :d]`,`up = x[..., d:]`
2. 计算 `out = SiLU(gate) * up`(逐元素)

**为什么 fused**:这是个**显存带宽受限**的操作(每个元素只算几个浮点),fused 后只读一次输入、写一次输出,而不是先 SiLU 再乘。

---

## 六 `LlamaMLP.forward`(`llama.py:106`)

```python
def forward(self, x, forward_batch=None, use_reduce_scatter=False):
    gate_up, _ = self.gate_up_proj(x)                     # ① fused gate + up GEMM
    x = self.act_fn(gate_up)                              # ② SiLU(gate) ⊙ up
    x, _ = self.down_proj(                                # ③ down GEMM (+ all-reduce)
        x,
        skip_all_reduce=use_reduce_scatter,
    )
    return x
```

**三步**:GEMM → SiLU·Mul → GEMM,完事。

---

## 七 形状追踪(Llama-7B,prefill 128 tokens,TP=1)

```
hidden_states                           [128, 4096]
        │
        ↓ gate_up_proj      ← [hidden=4096] @ [4096, 2 × 11008]
gate_up                                 [128, 22016]
        │
        ↓ SiluAndMul        ← split [..., :11008] (gate), [..., 11008:] (up); SiLU(gate) * up
intermediate                            [128, 11008]
        │
        ↓ down_proj         ← [11008, 4096],RowParallelLinear,TP 时 all-reduce
output                                  [128, 4096]
```

---

## 八 TP 切分

`LlamaMLP` 的 TP 模式是**经典 Megatron 风格**:

| 模块 | 并行类型 | 切的维度 | 通信 |
|---|---|---|---|
| `gate_up_proj` | **列并行**(`MergedColumnParallelLinear`) | 输出维(intermediate)切到各卡 | 不通信(input 全复制) |
| `down_proj` | **行并行**(`RowParallelLinear`) | 输入维(intermediate)切到各卡 | **all-reduce**(每张卡算自己一份,加起来 = 完整 hidden) |

为什么这样配对:`gate_up_proj` 列并行后每张卡有"一部分 intermediate 维度",`SiluAndMul` 是逐元素的,**可以本地各算各的**——不需要通信;`down_proj` 行并行刚好对接,把每张卡的"部分 hidden" 加起来 = 完整 hidden。

> 这种"列并行 + 行并行 + 中间不通信" 是 Megatron-LM 论文里的经典模式,attention 的 `qkv_proj`(列并行)+ `o_proj`(行并行)也是同一套思路。

---

## 九 性能视角

`LlamaMLP` 的两个 GEMM 是**整层 transformer 里 FLOPS 占比最高的一对**:

| 操作 | FLOPS(per token,Llama-7B) |
|---|---|
| qkv_proj | `4096 × (4096+1024+1024) ≈ 25M` |
| o_proj | `4096 × 4096 ≈ 17M` |
| **gate_up_proj** | **`4096 × 2 × 11008 ≈ 90M`** |
| **down_proj** | **`11008 × 4096 ≈ 45M`** |
| attention(prefill,seq=128) | `≈ 2 × N × seq × heads × head_dim ≈ 4M` |

prefill 阶段 GEMM 主导,**MLP 占了 ~75% 的 FLOPS**。这就是为什么"提升 MLP 的吞吐"是推理优化重点之一(int8/int4 量化首先量这两个 GEMM)。

decode 阶段则反过来——**显存带宽主导**,MLP 的 GEMM 都算得很快(token 数小),attention 读 KV cache 才是大头。

---

## 十 一句话总结

> **`LlamaMLP` = SwiGLU 风格的 FFN**:`gate_up_proj` 用一个融合 GEMM 同时算出 gate 和 up,`SiluAndMul` 在一个 kernel 里完成 `SiLU(gate) ⊙ up`,`down_proj` 投回 hidden 并 all-reduce 合并 TP 多卡输出。**这一步占 transformer 推理 FLOPS 的大头**,也是量化和算子融合的重点对象。
