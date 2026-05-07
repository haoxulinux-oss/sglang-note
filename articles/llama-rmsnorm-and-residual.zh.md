# RMSNorm 与残差透传(初学者向)

> 📖 **关联阅读**:
> - 父篇:[`LlamaDecoderLayer.forward()` 总览](llama-decoder-layer-overview.zh.md)
> - 系列下一篇:[Self-Attention 总览](llama-self-attention-overview.zh.md)

代码位置:
- `RMSNorm` 类:`python/sglang/srt/layers/layernorm.py:151`
- `LlamaDecoderLayer.forward` 里的两次 RMSNorm 调用:`python/sglang/srt/models/llama.py:311-323`

---

## 一 LayerNorm 家族:为什么 LLM 要归一化

每跑一层 transformer,hidden 向量的"幅度"会被残差累加和 MLP 放大。如果什么都不做,**深网络一路下来数值会爆炸**:激活值溢出、梯度爆炸、训练直接 NaN。

归一化的作用就一个:**把每个 token 的 hidden 向量幅度拉回固定量级**。常见三种:

| 归一化 | 公式核心 | 出处 |
|---|---|---|
| **BatchNorm** | 沿 batch 维度归一化 | 早期 CV |
| **LayerNorm** | 减均值 + 除标准差 + 可学习 γ、β | 原始 Transformer (BERT/GPT-2) |
| **RMSNorm** | **不减均值**,只除均方根,乘 γ | Llama / Qwen / DeepSeek |

RMSNorm 比 LayerNorm 少了"减均值"和"加偏置 β" 两步,**计算更省、效果几乎无损**——这就是 LLM 圈集体迁过来的原因。

---

## 二 RMSNorm 公式

对一个 hidden 维度 `d` 的 token 向量 `x = [x_1, ..., x_d]`:

```
RMS(x) = sqrt(mean(x_i²) + ε)        # 均方根(标量)
x̂      = x / RMS(x)                   # 归一化:把"长度"拉回 1 量级
y      = γ * x̂                        # 再乘上每维独立的可学习权重 γ(d 维)
```

### 2.1 用 `forward_native` 讲数学(不是真正运行的版本)

下面这段是 PyTorch 纯实现,**只用来讲清楚 RMSNorm 的数学等价表达式,生产环境实际不走这里**(下一节 2.2 解释)。

```python
# layernorm.py:287,简化版
def forward_native(self, x, residual=None, ...):
    x = x.to(torch.float32)                       # FP32 算更稳
    if residual is not None:
        x = x + residual                          # 先把残差加上(下文展开)
        residual = x.to(orig_dtype)               # 把"hidden+residual"作为新 residual 传出去

    variance = x.pow(2).mean(dim=-1, keepdim=True)        # ← mean(x²)
    x = x * torch.rsqrt(variance + self.variance_epsilon) # ← / sqrt(...)
    x = x * self.weight                                    # ← * γ
    return x, residual
```

### 2.2 实际调用的是 `forward_cuda`(平台分发机制)

`RMSNorm` 继承自 `MultiPlatformOp`(`python/sglang/srt/layers/utils/multi_platform.py:26`),**构造时一次性根据平台选定具体实现**:

```python
# multi_platform.py
class MultiPlatformOp(nn.Module):
    def __init__(self):
        super().__init__()
        self._forward_method = self.dispatch_forward()      # ← 启动期一次性选定

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)        # ← 运行期直接走选好的版本

    def dispatch_forward(self):
        if _is_cuda:   return self.forward_cuda             # ← NVIDIA GPU 默认走这里
        elif _is_hip:  return self.forward_hip
        elif _is_npu:  return self.forward_npu
        elif _is_xpu:  return self.forward_xpu
        ...
        else:          return self.forward_native           # 兜底
```

所以 `LlamaDecoderLayer.forward` 里的 `self.input_layernorm(...)`,**在 NVIDIA GPU 上实际调用的是 `forward_cuda`**(`layernorm.py:180`):

```python
# layernorm.py:180,实际运行的版本
def forward_cuda(self, x, residual=None, ...):
    ...
    if residual is not None:
        # ★ 一个 CUDA kernel 内完成 (x + residual) + RMSNorm,原地写
        fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
        return x, residual
    out = rmsnorm(x, self.weight.data, self.variance_epsilon)
    return out
```

**实现差别**:

| 路径 | 谁来跑 | 几个 kernel | 何时被选中 |
|---|---|---|---|
| `forward_cuda` | sgl-kernel 编译的 C++/CUDA(`fused_add_rmsnorm` / `rmsnorm`) | **1 个 fused kernel** | NVIDIA GPU(默认) |
| `forward_hip` | 同 forward_cuda | 1 个 | AMD ROCm |
| `forward_native` | PyTorch op 一步步算(几个 reduce + 几个 elementwise) | **多个 kernel,中间张量来回读写** | CPU 兜底;torch.compile 模式临时切到这条以便 compile 穿透优化 |
| `forward_npu` / `forward_xpu` / ... | 各硬件对应实现 | 视实现 | 对应硬件 |

> **数学等价,性能差距巨大**:`forward_cuda` 把 add+RMSNorm+乘 γ 全融合,只对 hidden 张量读一次、写一次;`forward_native` 走 PyTorch 时要分多次 launch、产生中间张量。生产推理一律走 `forward_cuda`。

---

## 三 Pre-Norm vs Post-Norm:为什么放在子层"前"

原始 Transformer (Vaswani 2017) 是 **Post-Norm**:

```
y = LayerNorm(x + Sublayer(x))
```

后来发现 Post-Norm 训深(>30 层)会不稳。GPT-2 起改成 **Pre-Norm**:

```
y = x + Sublayer(LayerNorm(x))
```

效果:**残差路径上"没有 norm 操作"**,梯度可以原封不动从最后一层流到第一层,深到 80 层(Llama-70B)都能稳训。

Llama 这条 `LlamaDecoderLayer.forward` 显然是 pre-norm:

```
                      ┌── + ──→ output
                      │
input ──→ norm ──→ sublayer
   └────────residual ─┘
```

---

## 四 残差透传:为什么 `residual` 是个"输入参数"

直觉写法(逐层独立):

```python
def forward(self, x):
    h = self.input_layernorm(x)
    a = self.self_attn(h)
    x = x + a                                       # 加残差①
    h = self.post_attention_layernorm(x)
    m = self.mlp(h)
    x = x + m                                       # 加残差②
    return x
```

SGLang 实际写法(残差透传):

```python
def forward(self, positions, hidden_states, forward_batch, residual):
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)   # ← fused: x = x+res; norm(x); res=x
    hidden_states = self.self_attn(...)

    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)  # ← 同上
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual                  # 残差还没加,留给下一层 / LlamaModel.norm 处理
```

**两者数学等价**,但后者把"加残差"和"做归一化"**融合到一个 CUDA kernel** 里跑(`fused_add_rmsnorm`,见 `layernorm.py:217`),能省掉一次对整个 hidden 张量的显存读写。**Llama-70B 一层 hidden 就是 [seq_len, 8192] FP16,一次少读写 = 一次少几十 MB 流量**——80 层乘起来很可观。

---

## 五 fused add+rmsnorm 的两种调用形态

`LlamaDecoderLayer.forward` 里 `RMSNorm(...)` 调用有**两种参数形态**:

### 形态 A:第一层(没残差可加)

```python
if residual is None:
    residual = hidden_states               # 把当前 hidden 当作"待加的残差"
    hidden_states = self.input_layernorm(hidden_states)   # 单参数版本:只 norm
```

走的是 `forward_cuda` 里 `residual is None` 分支,只调 `rmsnorm(x, weight, eps)` 一个 kernel。

### 形态 B:中间层(残差透传进来)

```python
hidden_states, residual = self.input_layernorm(hidden_states, residual)
```

走的是 `forward_cuda` 里 `residual is not None` 分支:

```python
fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
return x, residual                        # 注意 residual 已被原地修改为 x+residual
```

`fused_add_rmsnorm` 在一个 kernel 里完成:
1. `tmp = x + residual`
2. `residual ← tmp`(原地写回,作为新残差)
3. `x ← RMSNorm(tmp)`(原地写回,作为下一子层输入)

---

## 六 残差最终是什么时候加上去的

`LlamaDecoderLayer.forward` 返回时残差还没加。三种可能加上去的时机:

| 场景 | 加在哪 |
|---|---|
| **下一层 decoder** | 进 `LlamaDecoderLayer` 又调 `input_layernorm(hidden, residual)`,fused kernel 内部加上 |
| **最后一层之后** | `LlamaModel.forward` 末尾调 `self.norm(hidden_states, residual)`(`llama.py:407`),把累计残差并入最终 hidden |
| **PP 切分:中间 stage** | `LlamaModel.forward` 把 `(hidden_states, residual)` 一起塞进 `PPProxyTensors` 发给下一段进程,由对方接着走 |

---

## 七 一句话总结

> **`input_layernorm` / `post_attention_layernorm` 都是 RMSNorm**;借「残差透传 + fused add+rmsnorm」一个 kernel 把"加残差"和"做归一化"合并跑,**避免每层一次冗余的 hidden 读写**。这是 Llama/Qwen 系列 pre-norm 架构在推理引擎里的标准优化模式。
