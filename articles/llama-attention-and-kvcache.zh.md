# Attention 计算与 KV cache 写入(初学者向)

> 📖 **关联阅读**:
> - 父篇:[Self-Attention 总览](llama-self-attention-overview.zh.md)
> - 上一篇:[RoPE 旋转位置编码](llama-rope.zh.md)
> - KV cache 存储细节:[KV cache 存什么、HBM 是什么](kvcache-prefetch-and-storage.zh.md)
> - 下一篇:[MLP / SwiGLU](llama-mlp-swiglu.zh.md)

代码位置:
- `RadixAttention.forward`:`python/sglang/srt/layers/radix_attention.py:99`
- 后端基类 `AttentionBackend`:`python/sglang/srt/layers/attention/base_attn_backend.py:18`

---

## 一 这一步在做什么

回到 `LlamaAttention.forward`:

```python
q, k, v = self.forward_prepare_native(positions, hidden_states)   # qkv_proj + split + RoPE
attn_output = self.attn(q, k, v, forward_batch)                    # ← 本文焦点
output, _ = self.o_proj(attn_output)
```

`self.attn` 是 `RadixAttention` 的实例(`llama.py:190`)。它在做**两件事**:

1. **写 KV cache**:把当前这批 token 的 K、V 存进 `token_to_kv_pool` 对应位置(为后续 decode 服务)
2. **算 attention**:`output = softmax(Q · Kᵀ / √d) · V`,但 Kᵀ、V 是从 cache 里取的"历史 + 当前"

---

## 二 `RadixAttention` 是个壳:真正干活的是 `attn_backend`

`RadixAttention.forward`(`radix_attention.py:99`)简化版:

```python
def forward(self, q, k, v, forward_batch, save_kv_cache=True, **kwargs):
    # ① reshape 成 [N, num_heads, head_dim]
    k = k.view(-1, self.tp_k_head_num, self.qk_head_dim)
    v = v.view(-1, self.tp_v_head_num, self.v_head_dim)

    # ② 派发:走 unified_attention_with_output 还是 attn_backend.forward
    if forward_batch.forward_mode.is_extend() and get_forward_context() is not None:
        ...
        unified_attention_with_output(q, k, v, output, save_kv_cache, self.layer_id, **kwargs)
        return output
    else:
        return forward_batch.attn_backend.forward(q, k, v, self, forward_batch, save_kv_cache, **kwargs)
```

**关键事实**:`RadixAttention` 自己不算 attention,它把活儿派发给 `forward_batch.attn_backend`。这个 backend 是启动期 `ModelRunner.init_attention_backend` 选定的,常见有:

| 后端 | 用在哪 | 特点 |
|---|---|---|
| **FlashInfer** | 默认 | 高度优化的 paged attention,支持 RadixCache 树结构 |
| **FA3 / FlashAttention-3** | Hopper(H100/H200) | Hopper 专属优化 |
| **Triton** | fallback | Python 友好、容易 debug |
| **NPU / CPU** | 对应硬件 | |

> "**Radix**Attention" 这个名字源于 SGLang 的 RadixCache(prefix tree KV 复用),不是另一种算法,是同一种 attention 但配合 prefix tree 做 KV 复用。

---

## 三 attention backend 的派发逻辑

`AttentionBackend.forward`(`base_attn_backend.py:81`)按 `forward_mode` 分流:

```python
def forward(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
    if forward_batch.forward_mode.is_idle():
        return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)   # 空跑
    elif forward_batch.forward_mode.is_decode():
        return self.forward_decode(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    elif forward_batch.forward_mode.is_mixed() and is_npu():
        return self.forward_mixed(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
    else:
        return self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache, **kwargs)
```

**三条主路径**:

| 模式 | 入口 | 特点 |
|---|---|---|
| **DECODE** | `forward_decode` | 每个 req 只输入 1 个 token,query 维度小,attention 体积大(读全部历史 K/V) |
| **EXTEND**(prefill / chunked prefill / target verify 等) | `forward_extend` | 每个 req 输入多个 token,query 维度大,可以分块算 |
| **IDLE** | 空跑 | DP attention 下凑 batch 用 |

---

## 四 prefill 路径详解(`forward_extend`)

prefill 时一次进来一段(几十~几千 token)。**FlashInfer 的 prefill 内部要做**:

1. **写 KV cache**:遍历这一批每个 token,按 `forward_batch.out_cache_loc` 给的索引,把对应的 K、V 写到 `token_to_kv_pool` 里
   ```
   token_to_kv_pool.k_buffer[layer_id][out_cache_loc[t]] = k[t]
   token_to_kv_pool.v_buffer[layer_id][out_cache_loc[t]] = v[t]
   ```
   `out_cache_loc` 是 scheduler 端 `_add_request_to_queue` 时分配好的物理槽位编号。
2. **prefix attention**:如果开启 RadixCache 共享(prompt 前缀已在 cache 里),只对**新增 token** 跑 attention,query 长度 ≠ key 长度
3. **causal mask**:每个 token 只看前面的 token
4. **softmax(Q·Kᵀ/√d) · V**:用专门的 prefill attention kernel(FlashInfer 的 `BatchPrefillWithPagedKVCacheWrapper`)

---

## 五 decode 路径详解(`forward_decode`)

decode 时每个 req 只输入 1 个 token。`q` 形状 `[batch_size, num_q_heads, head_dim]`,**特别小**;但 K、V 来自全部历史(可能几千 token),**特别大**——这就是为什么 decode 是**显存带宽受限**(memory-bound)而不是计算受限。

**FlashInfer 的 decode 内部**:

1. **写 1 行 KV cache**:这一步生成的 K、V 写到 cache 末尾
2. **paged attention**:KV cache 不是连续大数组,而是按 page(典型 16 token 一页)管理。每个 req 通过 `req_to_token_pool` 维护一张"我的 token 在哪些 page" 的表
3. **算 softmax(Q·Kᵀ/√d) · V**:专门的 decode kernel,batch 维高、seq 维高,一次 launch 处理整个 batch

```
req 0: page list = [12, 47, 23, ...]
req 1: page list = [88, 14, 5, ...]
req 2: page list = [3, 91, ...]
        ↓
decode kernel 一次 launch,跨 req 读 page,每 req 算自己的 attention
```

> KV cache 的物理布局、HBM 带宽细节,见 [KV cache 存什么、HBM 是什么](kvcache-prefetch-and-storage.zh.md)。

---

## 六 KV cache 是怎么"按位置写入"的

回到 `radix_attention.py` 看 `unified_attention_with_output`(`:140`):

```python
forward_batch.out_cache_loc = original_out_cache_loc[:real_num_tokens]
...
attention_layer.attn_backend.forward(...)        # ← 内部会调 token_to_kv_pool.set_kv_buffer
```

**`forward_batch.out_cache_loc`** 是这一批 token 的 KV 写入位置(展平后的物理 index 数组)。例如:

```
本批新 token = 5 个
out_cache_loc = [102, 103, 104, 105, 106]
   → 第 0 个新 token 的 K, V 写到 k_buffer[layer_id][102], v_buffer[layer_id][102]
   → 第 1 个新 token 的 K, V 写到 k_buffer[layer_id][103], v_buffer[layer_id][103]
   ...
```

scheduler 端在组 batch 时已经决定好这些位置——具体是 `BaseTokenToKVPoolAllocator` 从空闲池里分配的连续段(或非连续段)。

写入 + 读取一次性完成,因为 attention kernel 内部会:
1. 先写新 K、V 到 cache
2. 然后从 cache 读"全部 K、V"算 attention

这样**写和读共享同一份 KV pool**,不用复制。

---

## 七 一个完整 prefill batch 的数据流(以 FlashInfer 为例)

```
Q ∈ [128, 4096]     ← LlamaAttention 给出的 q (32 heads × 128 dim,reshape 后 [128, 32, 128])
K ∈ [128, 1024]     ← reshape [128, 8, 128](GQA,8 KV heads)
V ∈ [128, 1024]     ← reshape [128, 8, 128]
        │
        ↓ RadixAttention.forward
        │
        ↓ unified_attention_with_output
        │
        ↓ FlashInfer prefill wrapper:
        │   ① set_kv_buffer:K, V → token_to_kv_pool[layer_id][out_cache_loc]
        │   ② BatchPrefillWithPagedKVCache:
        │      - 取整批 Q
        │      - 从 page table 取 K, V(自己刚写的 + RadixCache 共享前缀)
        │      - 跑 softmax(Q·Kᵀ·scaling) · V,带 causal mask
        ↓
output ∈ [128, 32, 128]      ← 多头独立结果
        │
        ↓ reshape
output ∈ [128, 4096]          ← 回到 hidden 维度
```

---

## 八 一句话总结

> **`RadixAttention.forward` 是个分发壳**,把 q/k/v 派发到 `forward_batch.attn_backend`(FlashInfer / FA3 / Triton)。后端按 `forward_mode` 分 prefill / decode 两条路径,**先把当前 token 的 K、V 写到 `token_to_kv_pool`,再调 paged attention kernel 跑 softmax(Q·Kᵀ/√d)·V**。写和读用同一份 KV pool,这就是 SGLang 推理性能的核心所在。
