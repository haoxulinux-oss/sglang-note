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

## 七 GQA 下 attention kernel 怎么算(以「一个 TP rank 正好是一个 group」为例)

回顾 [QKV + GQA](llama-qkv-projection-gqa.zh.md):「4 个 Q 头共享 1 个 KV 头」。具体到 attention kernel,Q @ Kᵀ 是怎么算的?以 Llama-3-8B、TP=8 为例(本卡正好是 1 个 group)。

### 7.1 这张卡上的张量形状

```
全局:   32 Q 头, 8 KV 头, head_dim=128, GQA group_size=4
本卡:   4 Q 头, 1 KV 头, 1 V 头              ← 正好一个 group
```

经过 `qkv_proj` + split + RoPE + reshape 后:

```
Q ∈ [N=128, num_q_heads_local=4, head_dim=128]    形状 [128, 4, 128]
K ∈ [N=128, num_kv_heads_local=1, head_dim=128]   形状 [128, 1, 128]
V ∈ [N=128, num_kv_heads_local=1, head_dim=128]   形状 [128, 1, 128]
```

### 7.2 数学:4 次独立 attention,共享同一份 K、V

```
对本卡 Q 头 i ∈ {0, 1, 2, 3}:
    attn_logits[i]  = Q[:, i, :] @ K[:, 0, :].T  / √d_k     形状 [128, 128]
    attn_logits[i] += causal_mask                            (上三角设 -inf)
    weights[i]      = softmax(attn_logits[i], dim=-1)        形状 [128, 128]
    out[:, i, :]    = weights[i] @ V[:, 0, :]                形状 [128, 128]
```

注意 `K[:, 0, :]` 和 `V[:, 0, :]` 在 i=0,1,2,3 这 4 次循环里**完全一样**——这就是「共享」的字面意思。最后:

```
out ∈ [N, 4, head_dim] = [128, 4, 128]
       reshape →  [N, 4 × head_dim] = [128, 512]
```

### 7.3 实际实现:K、V **不会**真的被复制 4 份

朴素写法是把 K 重复 4 次让 Q、K 头数对齐再跑标准 MHA:

```python
K_rep = K.repeat_interleave(group_size=4, dim=1)   # [128, 1, 128] → [128, 4, 128]
V_rep = V.repeat_interleave(group_size=4, dim=1)
out   = standard_mha(Q, K_rep, V_rep)
```

**这种做法显存带宽爆炸**——同一份 K 从 HBM 被读 4 次。LLM 推理是 memory-bound,这等于把 GQA 的好处全废了。

**FlashInfer / FA2 / FA3 / Triton 的真做法**:在 attention kernel 内部"虚拟广播",**K、V 物理只读一次**,被 4 个 Q 头复用。伪代码:

```python
# 在 attention kernel 内部(CUDA,不是 Python)
for token_a in range(N):                            # query token
    for token_b in range(token_a + 1):              # 历史 key token(causal)
        # ★ K、V 从 HBM 读一次,缓存到 shared memory / 寄存器
        k_block = K[token_b, 0, :]                  # 一个 KV 头的 128 维向量
        v_block = V[token_b, 0, :]

        for q_head_local in range(group_size=4):    # ← 4 次复用同一个 k_block / v_block
            q = Q[token_a, q_head_local, :]
            logit = dot(q, k_block) / sqrt(d_k)
            ...
```

**关键事实**:
- K、V 从 HBM 读取 1 次,**每读 1 次被 4 个 Q 头复用**,显存带宽节省 4 倍
- KV cache 物理上只存 `[N, 1, 128]` 一份,**不复制**
- `group_size` 作为 kernel 启动参数传入,kernel 内部按 `q_head // group_size` 索引到 KV 头

### 7.4 单 group 是 GQA 的最简形态

`q_head // group_size = q_head_local // 4 = 0`(本卡 4 个 Q 头都属于同一个 group,共享同一个 KV 头)。所以 attention kernel 里:

```
所有本卡 Q 头都用 K[:, 0, :] / V[:, 0, :] —— 唯一的那个 KV 头
```

K、V 形状 `[N, 1, 128]` —— **token 维仍是 N**,head 维只有 1。Q 这边是 `[N, 4, 128]`。kernel 对 q_head 这个维度做"对所有 head 用同一个 K"的广播,本质上只是 4 倍 Q-head 维度的 batch 化。

### 7.5 算完之后 out 怎么用

```
本卡 out ∈ [N, 4, 128]
       → reshape → [N, 512]                   ← 4 × 128 = 512 = hidden_size / TP
       → o_proj (RowParallelLinear)
       → [N, 4096](本卡 partial sum)
       → all-reduce(跨 8 张 TP 卡求和)
       → [N, 4096](完整 hidden_states,所有卡一致)
```

`o_proj` 输入 512 = `(本卡 Q 头数) × head_dim` = `4 × 128`,**恰好是本卡 4 个 Q 头的拼接**。`RowParallelLinear` 把 512 当作"hidden 维度被切了 8 份的其中一份",GEMM 后做 all-reduce 把 8 张卡各自的 partial sum 加起来 = 全部 32 个 Q 头共同贡献的完整 attention 输出。

> 这一步呼应 [Self-Attention 总览](llama-self-attention-overview.zh.md) §四 `o_proj`:**TP 切的是 attention 头,不是 hidden_size 本身**;每张卡负责自己头的 attention,合并阶段在 `o_proj` all-reduce。

### 7.6 一句话总结(本节)

> 本卡 `Q ∈ [N, 4, 128]`,`K, V ∈ [N, 1, 128]`。**4 个 Q 头各自和这唯一的 K 头算 `Q_i @ K_0ᵀ / √d`,得 4 张 [N, N] logits;softmax 后再各自和这唯一的 V 头算加权和,得 [N, 4, 128] out**。物理上 K、V 只读 1 次,kernel 在 Q 头维做广播复用——这就是 GQA 在 prefill / decode kernel 里的真实运行方式。

---

## 八 一个完整 prefill batch 的数据流(以 FlashInfer 为例)

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

## 九 一句话总结

> **`RadixAttention.forward` 是个分发壳**,把 q/k/v 派发到 `forward_batch.attn_backend`(FlashInfer / FA3 / Triton)。后端按 `forward_mode` 分 prefill / decode 两条路径,**先把当前 token 的 K、V 写到 `token_to_kv_pool`,再调 paged attention kernel 跑 softmax(Q·Kᵀ/√d)·V**。写和读用同一份 KV pool;**GQA 下 K、V 在 kernel 内广播复用**,4 个 Q 头共享一份 K、V 而无需物理重复——这就是 SGLang 推理性能的核心所在。
