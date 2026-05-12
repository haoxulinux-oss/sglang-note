# `FlashInferAttnBackend.forward_extend()` 详解(初学者向)

> 📖 **关联阅读**:
> - 父篇:[Attention 计算与 KV cache 写入](llama-attention-and-kvcache.zh.md) §三 已经讲清「`forward_batch.attn_backend.forward` 在 RTX 4070 上就是 `FlashInferAttnBackend.forward`」,然后按 `forward_mode` 派发到本函数
> - 上游:[Self-Attention 总览](llama-self-attention-overview.zh.md) §四 RadixAttention
> - 横向:[KV cache 存什么、HBM 是什么](kvcache-prefetch-and-storage.zh.md)

代码位置:
- `FlashInferAttnBackend.forward_extend`:`python/sglang/srt/layers/attention/flashinfer_backend.py:775`
- `MHATokenToKVPool.set_kv_buffer`:`python/sglang/srt/mem_cache/memory_pool.py:1022`
- `init_forward_metadata`:`python/sglang/srt/layers/attention/flashinfer_backend.py:433`

---

## 一 这一步在做什么

`FlashInferAttnBackend.forward_extend()` 是 **RTX 4070 上 prefill / extend / chunked-prefill 阶段 attention 的真实入口**。它做三件事:

1. **挑出对应的 paged-wrapper**(`prefill_wrapper_paged`)
2. **把这一批新算的 K、V 写入 KV cache 池**(可能跳过)
3. **算 attention**:`softmax(Q · Kᵀ / √d_k) · V`,带 causal mask

最终输出形状 `[total_tokens, num_q_heads × head_dim]`(和 `LlamaAttention` 期望的 `attn_output` 一致)。

---

## 二 函数签名和调用者

```python
def forward_extend(
    self,
    q: torch.Tensor,            # [total_tokens, num_q_heads × head_dim]
    k: torch.Tensor,            # [total_tokens, num_kv_heads, head_dim] (本批新算的 K)
    v: torch.Tensor,            # [total_tokens, num_kv_heads, head_dim] (本批新算的 V)
    layer: RadixAttention,      # 当前 attention 层的元数据载体
    forward_batch: ForwardBatch,# 整批请求的全部数据
    save_kv_cache: bool = True, # 是否把 K/V 写进 cache(只有跑 encoder-only 等场景才 False)
):
```

调用者是 `AttentionBackend.forward`(`base_attn_backend.py:81`),按 `forward_mode` 分流到这里:

```
forward_batch.attn_backend.forward(...)              # base_attn_backend.py
   └─ if forward_mode.is_extend():
        self.forward_extend(q, k, v, layer, fb, save_kv_cache)   ← 本文
```

---

## 三 出场角色与关键成员预热

读懂这个函数,需要先认识 4 个对象:

| 对象 | 类型 / 类 | 重要字段 / 方法 | 作用 |
|---|---|---|---|
| `layer` | `RadixAttention` | `layer_id` / `tp_q_head_num` / `tp_k_head_num` / `head_dim` / `scaling` / `k_scale_float` 等 | 当前层的元数据。把 q/k/v 形状信息和 scale 携带过来 |
| `forward_batch` | `ForwardBatch` | `out_cache_loc` / `token_to_kv_pool` / `req_pool_indices` / `seq_lens` / `extend_prefix_lens` | 整批请求的运行时状态(详见父篇 §2.1) |
| `forward_batch.token_to_kv_pool` | **`MHATokenToKVPool`**(`memory_pool.py:764`) | `set_kv_buffer()` / `get_kv_buffer()` / `k_buffer` / `v_buffer` | **真正持有 KV 显存池的对象**——这是 `set_kv_buffer` 的归属类 |
| `self.forward_metadata` | **`PrefillMetadata`**(`flashinfer_backend.py:99`) | `prefill_wrappers` / `use_ragged` / `extend_no_prefix` / `multi_item_params` | 启动期+本轮预算好的「attention kernel 启动配置」 |

下面 §四、§五、§六 分别讲清这三个核心对象。

---

## 四 `set_kv_buffer` 是哪个类的成员

`forward_extend` 里这行:

```python
forward_batch.token_to_kv_pool.set_kv_buffer(
    layer, cache_loc, k, v, layer.k_scale, layer.v_scale
)
```

调到 `MHATokenToKVPool.set_kv_buffer`(`memory_pool.py:1022`)——**MHA Token-to-KV Pool**,SGLang 持有 KV 显存的核心类。继承关系:

```
KVCache (抽象基类, memory_pool.py:~700)
   └── set_kv_buffer  (抽象方法, memory_pool.py:742)
   └── get_kv_buffer
   └── get_key_buffer
   ↑ 继承
MHATokenToKVPool       (memory_pool.py:764)           ← Llama / Qwen 等 MHA/GQA 模型用这个
MHATokenToKVPoolFP4    (memory_pool.py:1111)          ← FP4 量化版本
MLATokenToKVPool       (memory_pool.py:~1400)         ← DeepSeek 系 MLA 用
HybridLinearKVPool     (memory_pool.py:~1750)         ← 混合架构用
...
```

### 4.1 `MHATokenToKVPool` 的物理布局

`__init__` 里 `_create_buffers()`(`memory_pool.py:872`)给每层都开两个张量:

```python
self.k_buffer = [
    torch.zeros((size + page_size, head_num, head_dim), dtype=store_dtype, device=device)
    for _ in range(layer_num)
]
self.v_buffer = [ 同样形状 ]
```

举例 Llama-3-8B(`head_num=8` GQA,`head_dim=128`,`bf16`,32 层,假设 `size=65536`):

```
k_buffer[layer_id]: [65536+16, 8, 128]  bf16  → 大约 128 MB / 层
v_buffer[layer_id]: 同上                       → 128 MB / 层
合计 32 层:K + V = 32 × (128+128) MB ≈ 8 GB    ← 这就是 KV cache 显存占用
```

**每张卡每层有 `(size + page_size)` 个 KV slot**,每个 slot 是 `[head_num, head_dim]` 的一片小张量。`size` 是 scheduler 分配给 KV pool 的物理 token 容量上限,启动时根据剩余显存决定。

#### 4.1.1 为什么要 `+ page_size`?这多出来的是什么

源码注释(`memory_pool.py:880`)给出答案:

> The padded slot 0 is used for writing dummy outputs from padded tokens.

也就是说,**`size` 是真实有效 slot 数,`+ page_size` 是为了多出一段"哑(dummy)区"专门接收 padding token 的写入**。这一段额外空间存在于每个 buffer 的尾部(实现上是把 buffer 拉长 `page_size` 个 slot,然后**约定 slot 0 / 哑区是 dummy slot**)。

**为什么需要哑区?** 跟 CUDA Graph 强相关。

回顾 [`ModelRunner.forward`](model-runner-forward.zh.md) 里讲的:
- decode 阶段大多走 **CUDA Graph 重放**(`graph_runner.replay`)
- CUDA Graph 是按"batch_size 桶"预录的——比如桶 = {1, 2, 4, 8, 16, 32, 64, ...}
- 如果实际 batch 只有 5 个 req,会向上对齐到桶 8,**多出来的 3 个位置就是 padding**

padding 出来的 3 个 token 也会走 attention kernel,kernel 计算后也会试图调 `set_kv_buffer` 写 K、V——**但这些 padding token 不属于任何真实请求**,数据是垃圾,**绝对不能污染真实 slot**。

解决办法:把这些 padding token 的 `out_cache_loc[i]` **统一写成 0**(`cuda_graph_runner.py:278` 启动期 `out_cache_loc.zero_()`,再用 `[:raw_num_token]` 覆盖真实 token),让它们都写到 **slot 0(哑区入口)** —— K、V 仍然被写出去,但被写到一个永远不会被任何 query 读取的"垃圾桶"位置。

`+ page_size` 的另外两层作用:

- **页对齐**:paged KV cache 按 `page_size`(SGLang 默认 16)分页管理,buffer 末尾留出一页避免最后一页越界访问
- **预取/写入的 over-read 安全垫**:CUDA kernel 可能一次读/写一整页,即便最后一个真实 token 落在某页的中间位置,后续位置被 over-read 也不会触发非法访存

直观示意:

```
k_buffer[layer_id]:
  [ slot 0 ][ slot 1 ][ slot 2 ] ... [ slot size-1 ][ slot size ][ ... ][ slot size+page_size-1 ]
    ↑                                  ↑              ↑                              ↑
    dummy 哑区入口                     有效 slot 末尾  哑区开始(padding token 写到这里)
    所有 padding token 都写到这里      
```

举例 Llama-3-8B + `size=65536`,`page_size=16`:`k_buffer = [65552, 8, 128]`,**前 65536 个 slot 真实可用,后 16 个 slot 是哑区**。哑区只占整个 buffer 的 0.024%,代价可忽略,换来 CUDA Graph 录制时不用区分 padding/真实 token,kernel 代码大幅简化。

### 4.2 `set_kv_buffer` 的核心代码

```python
def set_kv_buffer(self, layer, loc, cache_k, cache_v, k_scale=None, v_scale=None, ...):
    layer_id = layer.layer_id
    # 处理量化的 scale(默认 bf16 时跳过)
    if cache_k.dtype != self.dtype:
        if k_scale is not None: cache_k.div_(k_scale)
        if v_scale is not None: cache_v.div_(v_scale)
        cache_k = cache_k.to(self.dtype)
        cache_v = cache_v.to(self.dtype)

    # ★ 真正的写入,调底层 fused kernel
    _set_kv_buffer_impl(
        cache_k, cache_v,
        self.k_buffer[layer_id - self.start_layer],    # 目标 K 池(本层)
        self.v_buffer[layer_id - self.start_layer],    # 目标 V 池(本层)
        loc,                                             # 写到哪些位置(物理 slot 索引)
        ...
    )
```

`_set_kv_buffer_impl`(`memory_pool.py:90`)做的事用伪代码表示:

```python
# 简化版,实际有 fused store_cache kernel(同时写 K、V)
k_buffer[indices] = k      # scatter 写
v_buffer[indices] = v
```

其中 `indices = forward_batch.out_cache_loc`,**这一批每个新 token 的物理 KV 槽位编号**。

> 关键事实:**`MHATokenToKVPool` 是真正持有 GPU HBM 上 KV 张量的对象**。它由 `ModelRunner.init_memory_pool()` 在启动期创建,挂在 `forward_batch.token_to_kv_pool` 上跨进程访问。

---

## 五 `self.forward_metadata.use_ragged` 是什么意思

`forward_metadata` 是 `FlashInferAttnBackend.init_forward_metadata`(`:433`) 在**每次 forward 开始前**重新生成的——根据当前 batch 的 `forward_mode` 选好对应的 wrapper 和参数。对 EXTEND 路径,它的类型是:

```python
@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]   # 实际要用的 paged wrapper
    use_ragged: bool                                               # ★ 本节焦点
    extend_no_prefix: bool                                         # 这一批是否完全没有历史前缀
    multi_item_params: Optional[MultiItemScoringParams]            # 多项打分专用
```

### 5.1 Ragged 和 Paged 的区别

FlashInfer 提供两种 prefill wrapper,**对应两种 K/V 的"输入布局"**:

| Wrapper | K/V 输入来源 | 适用场景 |
|---|---|---|
| **`BatchPrefillWithPagedKVCacheWrapper`** (paged) | 从 KV cache **页表**读 K、V(K/V 在 cache 池里离散分布) | 永远适用——既能读历史前缀,也能在写入后读到自己的新 K/V |
| **`BatchPrefillWithRaggedKVCacheWrapper`** (ragged) | 直接从**当前一批 concat 在一起的 K、V 张量**读,**不经过 cache** | 只跑「无前缀」prefill 时更快,因为省掉一次 page table 间接寻址 |

**Ragged = "不规整地拼接"**——这一批每个 req 的 K、V 序列长度不一样(128、200、64),直接 concat 成 `[total_tokens, num_kv_heads, head_dim]` 的扁平张量,kernel 通过 `seq_lens_cumsum` 知道每个 req 的边界。

> 名字来源:相对于 padded(规整 padding 到统一长度,浪费显存),"ragged" 表示**不 padding、不规整、变长**——和 NestedTensor 是类似概念。

#### 5.1.1 Ragged 布局图示(本批 K、V 的输入张量)

batch=3,prompt 长度分别为 4、3、5(共 12 个 token):

```
                ┌─────── req A: 4 token ────────┐ ┌─── req B: 3 ───┐ ┌──── req C: 5 token ──────┐
k 张量(扁平):  K_A0  K_A1  K_A2  K_A3   K_B0  K_B1  K_B2   K_C0  K_C1  K_C2  K_C3  K_C4
索引(token):   0     1     2     3      4     5     6      7     8     9     10    11
                                                                                            ↑ total_tokens = 12

形状:  k.shape = [12, num_kv_heads, head_dim]
       v.shape = 同上

伴随的边界数组(指引 kernel 拆分):
    seq_lens          = [ 4, 3, 5 ]         每个 req 的长度
    seq_lens_cumsum   = [ 0, 4, 7, 12 ]     每个 req 的起始 / 终止 offset
                          ↑  ↑  ↑   ↑
                          A  B  C   end
```

**特点**:
- **物理上是一段连续内存**(单个张量)——这就是 ragged 的核心
- **没有 padding**(不补到统一长度,直接挨着拼)
- **kernel 通过 `seq_lens_cumsum` 知道边界**:第 i 个 req 的 K 是 `k[seq_lens_cumsum[i] : seq_lens_cumsum[i+1]]`
- 这一批算完 attention 后**才**写进 KV cache 池(分支 B 的执行顺序)

#### 5.1.2 Paged 布局图示(KV cache 池里的全局存储)

KV cache 池一层的全貌(`MHATokenToKVPool.k_buffer[layer_id]`):

```
        page 0          page 1          page 2          page 3          page 4         ...
     ┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┬────
slot │ 0 ...15  │ 16 ...    31 │ 32 ...    47 │ 48 ...    63 │ 64 ...    79 │ ...
     └──────────┴──────────────┴──────────────┴──────────────┴──────────────┴────
           ↑          ↑             ↑              ↑               ↑
       page_size=16   每页存连续 16 个 token 的 K(或 V),按 [page_size, head_num, head_dim] 排布

每个 req 的「页表」(`req_to_token_pool` 维护):
    req A (48 token, 占 3 页):    page_indices = [3, 0, 7]   ← 3 个页号,不一定连续!
    req B (32 token, 占 2 页):    page_indices = [5, 12]
    req C (60 token, 占 4 页):    page_indices = [1, 9, 14, 2]

如果 req 的最后一页没用满,记录 last_page_len。
```

**特点**:
- **物理上离散**——一个 req 的 KV 可能散落在多个非连续 page 上(显存动态分配 → 必然有碎片)
- **通过「页表」间接寻址**:`req_pool_indices` → 一组 page id → 在 `k_buffer` 里取 page
- **page_size=16 是 SGLang 默认值**(`--page-size` 可调,16 是性能 / 碎片的甜点)
- **支持 RadixCache 前缀共享**:多个 req 共享同一组前缀页,显存零拷贝复用

#### 5.1.3 两种布局对比图

```
══════════════════════ Ragged(用于本批新增 K、V)══════════════════════

       本批 12 个新 token 的 K(一段连续内存,不补 padding):
       ┌─────────────────────────────────────────┐
       │ K_A0 K_A1 K_A2 K_A3 K_B0 K_B1 K_B2 K_C0... │
       └─────────────────────────────────────────┘
              ↑
   kernel 直接读这段连续内存,通过 cumsum 知道 req 边界
   无寻址间接性,SM 加载 K 时连续访问 → 带宽利用率高


══════════════════════ Paged(用于 KV cache 历史)══════════════════════

       KV pool(分页存储,可能跨多个 req 共享):

       page 0     page 1     page 2     page 3     page 4     page 5  ...
       ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐
       │16 个 │  │16 个 │  │16 个 │  │16 个 │  │16 个 │  │16 个 │
       │ slot │  │ slot │  │ slot │  │ slot │  │ slot │  │ slot │
       └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘
          ↑         ↑                                      ↑
       req B    req A                                    req A
       (page 0) (page 1)                                 (page 5)

       req A 的页表 = [1, 5, ...]      ← 非连续!
       req B 的页表 = [0, ...]

   kernel 读 K_A 时:先查页表 [1, 5, ...] → 再去 page 1 / page 5 取数据
   多一次「页表→物理地址」间接寻址,但能灵活管理(碎片少 + 前缀共享)


══════════════════════ 一句话区分 ══════════════════════

   Ragged  =  「本批输入,一段连续张量,长度不一,无寻址」
   Paged   =  「全局历史,分页存储,跨 req 共享,有页表」
```

> **关键观察**:这两种布局不是"你用哪个"的选择,而是**「本批增量 K/V」用 ragged,「历史前缀 K/V」用 paged**——同一次 attention 可能两者都用到(就是 §七 分支 C 的情况)。`use_ragged` 这个 flag 控制的只是「**有没有可能走 ragged 路径**」,实际是否真的用 ragged 还要看是否有历史前缀。

### 5.2 `use_ragged` 何时为 True

在 `init_forward_metadata` 的 EXTEND 分支(`:491`):

```python
if self.is_multimodal or self.enable_mis:
    use_ragged = False        # 多模态 / 多项打分,需要 paged 的特殊掩码功能
else:
    use_ragged = (
        not self.enable_deterministic                  # 确定性推理时必须 paged(可复现)
        and not is_in_piecewise_cuda_graph()           # piecewise CUDA Graph 内不用 ragged
    )
    extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
```

**RTX 4070 上的常规 prefill 默认 `use_ragged=True`**(没开多模态、没开 deterministic、没在 piecewise graph 里时)。

### 5.3 `extend_no_prefix` 是什么

```python
extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)
```

如果**这一批所有请求都没有"历史前缀"**(没命中 RadixCache 的共享前缀),`extend_no_prefix=True`。**全新 prompt 是这种情况**;如果命中过 RadixCache 前缀(前 1024 token 已在 cache 里、只需对剩余 token 跑 prefill),就是 False。

这个标志影响 `use_ragged=True` 路径下的 sub-branch(详见 §七)。

---

## 六 `prefill_wrapper_paged.forward` 的实现在哪

```python
prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
    self._get_wrapper_idx(layer)
]
```

`prefill_wrappers` 元素的类型是 **`flashinfer.BatchPrefillWithPagedKVCacheWrapper`**——**SGLang 不写这个类,它来自外部 `flashinfer` Python 库**(import 在 `flashinfer_backend.py:51`):

```python
if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
```

### 6.1 FlashInfer 项目在哪

- GitHub 仓库:[flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- Python 安装时的位置:`$VIRTUAL_ENV/lib/python3.*/site-packages/flashinfer/`
- 源码主要文件:
  - `flashinfer/prefill.py` — `BatchPrefillWithPagedKVCacheWrapper` 的 Python 类
  - `flashinfer/decode.py` — `BatchDecodeWithPagedKVCacheWrapper`
  - `csrc/flashinfer_ops.cu` — C++/CUDA kernel 入口
  - `include/flashinfer/attention/` — CUDA kernel 模板

### 6.2 wrapper 的两阶段使用

FlashInfer wrapper 的 API 是**两阶段**:`plan` + `run`(`forward` 是 `run` 的便捷别名)。

```python
# ── 阶段 1:plan(每次 forward 开始前调一次,在 SGLang 里就是 init_forward_metadata 里间接调的)──
prefill_wrapper_paged.plan(
    qo_indptr=...,            # 每个 req 的 query 偏移
    paged_kv_indptr=...,      # 每个 req 的 KV 页偏移
    paged_kv_indices=...,     # KV 页的物理编号
    paged_kv_last_page_len=...,  # 最后一页有多少有效 token
    num_qo_heads=...,
    num_kv_heads=...,
    head_dim=...,
    page_size=...,
)

# ── 阶段 2:forward(每层 attention 都调一次)──
output = prefill_wrapper_paged.forward(
    q,                       # [total_tokens, num_q_heads, head_dim]
    kv_cache,                # token_to_kv_pool.get_kv_buffer(layer_id) → (k_buffer, v_buffer)
    causal=True,
    sm_scale=layer.scaling,
    window_left=...,         # sliding window 配置(Llama 通常 -1 关闭)
    logits_soft_cap=...,
    k_scale=..., v_scale=...,
)
```

**`plan` 把"形状/索引/分块策略"算好,存在 wrapper 内部**——这是 host 端的轻量准备(几百 μs)。**`forward` 才是真正调 CUDA kernel**——所有 32 层都共享同一个 plan,只在第一层 attention 之前 plan 一次。

#### 6.2.1 plan 和 forward 到底如何配合(展开)

两阶段的协作可以这样理解:**plan 是"对这一批请求做总体规划",forward 是"按规划逐层执行"**。

#### 第一步:plan 阶段(每次 forward_batch 跑一次)

plan 的输入是**和具体某一层无关、只和这一批请求结构有关**的信息——`qo_indptr`(每个 req 的 query 偏移)、`paged_kv_indices`(每个 req 的页号)、`num_kv_heads`、`head_dim`、`page_size` 等。这些信息**不管跑第几层 attention 都一样**(因为同一 batch 内,每层的 KV 都存在同一组页里、batch_size、seq_lens 都不变)。

plan 内部做的事:

1. **算 tile 切分策略**:这一批要分成多少个 thread block,每个 block 负责哪几个 (req, token, head)
2. **算 scheduler 元数据**:CUDA kernel 启动时要读的"任务派发表",决定哪个 SM 处理哪段
3. **(可能)autotune**:跑 micro-benchmark 选最快的 kernel 变体
4. **分配 scratch 显存**:每个 thread block 在中间结果(部分 softmax 的 LSE)需要的临时空间
5. **把这些元数据存到 wrapper 内部的 GPU buffer 里**(`wrapper._paged_kv_indptr_buf` 等)

plan 的特点:
- **CPU 端工作较重**(几百 μs)——要做调度、autotune
- **GPU 端工作较轻**(一些小张量拷贝)
- **每次 forward_batch 跑前调用一次**——`FlashInferAttnBackend.init_forward_metadata` 内部通过 `indices_updater_prefill.update(...)` → `wrapper.begin_forward()`(`plan` 的别名)间接触发

#### 第二步:forward 阶段(每层调一次,共 32 次)

forward 的输入是**和具体某一层强相关的**——这一层的 Q 张量、这一层的 KV 池(`token_to_kv_pool.get_kv_buffer(layer_id)`)、scale 等。**plan 里准备好的所有元数据 wrapper 都还记得**,不需要再传一遍。

forward 内部做的事:
1. 用 plan 里存好的 tile 切分启动 CUDA kernel
2. kernel 从 wrapper 内部的 GPU buffer 读 `paged_kv_indices` 等元数据
3. 按 plan 决定的分配,SM 们并行算 attention
4. 把结果写到 output 张量

forward 的特点:
- **CPU 端工作极轻**(只是一次 kernel launch,几 μs)
- **GPU 端工作就是真正算 attention 的全部计算**
- **每层 attention 调一次**——32 层模型就调 32 次

#### 为什么要拆成两步?——「一次规划、多次执行」的节省

朴素写法是把所有参数都塞给 forward 调用,每层都重新算一遍调度。但是:

```
                    plan      forward
                    ┌──┐      ┌──┐
Llama-3-8B 32 层:    │CPU│ +   │GPU│ × 32 层
                    └──┘      └──┘
                    几百μs    每层几十 μs

如果不拆开,每层都要重做 plan:
                    ┌─────────────┐
                    │CPU+GPU│ × 32 层
                    └─────────────┘
                    几百 μs × 32 ≈ 累计 10+ ms ← 浪费,因为 plan 信息没变
```

CUDA Graph 时这个收益更明显:**plan 在 graph 外做(host 端)、forward 在 graph 内录制(纯 device 端)**——graph 录制要求所有操作不依赖 CPU 决策,所以"决策"必须提前完成。

#### 具体时序示意(一个完整 forward_batch 的 attention 部分)

```
forward_batch 开始
   │
   ├─ FlashInferAttnBackend.init_forward_metadata()    ← 决定 use_ragged 等参数
   │      └─ indices_updater_prefill.update()
   │             └─ wrapper.begin_forward(...)         ★ 这就是 plan,一次
   │                  │
   │                  └─ 把 tile 调度 / page 索引 / scratch 配置存进 wrapper 内部
   │
   ├─ Llama 层 0 跑 forward
   │      └─ attention →  prefill_wrapper_paged.forward(q, kv_cache, ...)   ★ run #1
   │                            ↑ 从 wrapper 内部读 plan 好的配置,只传 q 和 kv_cache
   │
   ├─ Llama 层 1 跑 forward
   │      └─ attention →  prefill_wrapper_paged.forward(q, kv_cache, ...)   ★ run #2
   │
   ├─ ... 中间 28 层同样 ...
   │
   ├─ Llama 层 31 跑 forward
   │      └─ attention →  prefill_wrapper_paged.forward(q, kv_cache, ...)   ★ run #32
   │
   └─ forward_batch 结束
```

#### 一句话总结 plan + forward

> **`plan` 做「批级一次性 host 规划」**(算调度、autotune、固化元数据到 GPU buffer);**`forward` 做「层级 GPU 计算」**(只传 Q 和该层的 KV cache 视图,kernel 从 plan 好的元数据起跑)。**32 层共享同一个 plan,所以拆开来比每层重做规划省一个数量级的 CPU 开销**——这是高性能推理引擎(SGLang / vLLM / TensorRT-LLM)都遵循的"plan-then-run"模式。

### 6.3 forward 内部走到哪个 kernel

`BatchPrefillWithPagedKVCacheWrapper.forward` 内部根据 `backend` 参数(`auto` / `cutlass` / `fa3` / `cudnn` 等)分发:

- **Ada Lovelace(RTX 4070)的默认 backend** 是 FlashInfer 自家的 paged attention CUDA kernel(模板生成的),源码在 `flashinfer/include/flashinfer/attention/prefill.cuh`
- 调用栈:`Python forward → C++ binding → CUDA kernel launch → SM 上的 paged attention 算法`

paged attention 算法核心:
1. 按 page(16 token 一页是 SGLang 默认)读 K、V
2. tile 化:把 query / KV 分块装入 shared memory
3. 在 shared memory 内算 `softmax(Q · Kᵀ / √d)`,边算 softmax 边乘 V(FlashAttention 思想——不实例化中间 `[N, N]` 注意力矩阵)
4. 写回 output

> 想看 CUDA kernel 源码请去 FlashInfer 项目;**SGLang 这边只负责"调它"**。这种"Python 高层 + 外部库底层 CUDA"是 SGLang 推理引擎的标准结构。

---

## 七 `forward_extend` 三条执行分支

现在把整个函数串起来。根据 `use_ragged` 和 `extend_no_prefix`,有**三条分支**:

```python
if not self.forward_metadata.use_ragged:
    # ─ 分支 A:paged-only(use_ragged=False)─
    ...
else:
    # use_ragged=True
    if self.forward_metadata.extend_no_prefix:
        # ─ 分支 B:纯 ragged(无历史前缀)─
        ...
    else:
        # ─ 分支 C:ragged + paged + merge_state(有历史前缀)─
        ...
```

### 7.1 分支 A:`use_ragged=False`(paged-only)

适用场景:多模态、deterministic 推理、piecewise CUDA Graph 内。

```python
# ① 写 K/V 到 cache 池
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(
        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
    )

# ② 直接调 paged wrapper(K/V 已经在 cache 里了,kernel 从 cache 读)
o = prefill_wrapper_paged.forward(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),   # (k_buffer, v_buffer)
    causal=True,
    sm_scale=layer.scaling,
    window_left=layer.sliding_window_size,
    logits_soft_cap=layer.logit_cap,
    k_scale=layer.k_scale_float, v_scale=layer.v_scale_float,
)
```

特点:**先写 cache、再从 cache 读**。简单可靠。

### 7.2 分支 B:`use_ragged=True` + `extend_no_prefix=True`(纯 ragged)

适用场景:**全新 prompt,没命中任何 RadixCache 前缀**——最快路径。

```python
o = self.prefill_wrapper_ragged.forward(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k.view(-1, layer.tp_k_head_num, layer.head_dim),     # ★ 直接读输入的 k 张量,不经过 cache
    v.view(-1, layer.tp_v_head_num, layer.head_dim),
    causal=True,
    sm_scale=layer.scaling,
    logits_soft_cap=logits_soft_cap,
)

# 写 cache 在 attention 算完之后做(因为 attention 不依赖 cache,可以延后)
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
```

特点:
- attention 不读 KV cache 池,**直接消费 `forward_extend` 的输入 k、v 参数**
- 省掉一次"先写 cache 再读"的多余间接寻址
- KV cache 写入推迟到 attention 之后

### 7.3 分支 C:`use_ragged=True` + 有历史前缀(split-K merge)

适用场景:**chunked prefill 或 RadixCache 命中**——一部分 K、V 在 cache 池里(前缀),一部分是本次新算的(增量),需要分两次跑 attention 然后合并。

```python
# ── 第一段:对本次新算的 K、V 跑 ragged ──
o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    k.view(-1, layer.tp_k_head_num, layer.head_dim),    # 新 K
    v.view(-1, layer.tp_v_head_num, layer.head_dim),    # 新 V
    causal=True,                                         # 新 token 之间有 causal
    sm_scale=layer.scaling,
    logits_soft_cap=logits_soft_cap,
)

# ── 第二段:对历史前缀(在 paged cache 里)跑 paged ──
o2, s2 = prefill_wrapper_paged.forward_return_lse(
    q.view(-1, layer.tp_q_head_num, layer.head_dim),
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),
    causal=False,                                        # 前缀对当前 token 是「过去」,不用 causal
    sm_scale=layer.scaling,
    logits_soft_cap=logits_soft_cap,
)

# ── 第三步:用 log-sum-exp 合并两段 attention ──
o, _ = merge_state(o1, s1, o2, s2)

# 最后写 cache
if save_kv_cache:
    forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)
```

**`forward_return_lse` 比 `forward` 多返回一个 `s`**——是 `log-sum-exp`(LSE),记录这一段 softmax 的归一化常数。**`merge_state` 用 LSE 把两段 attention 在数学上正确地合并**:

```
最终 attention = softmax([logits_新 | logits_旧]) · [V_新 | V_旧]
              = α · softmax(logits_新) · V_新 + (1-α) · softmax(logits_旧) · V_旧
                                       (α 由 LSE 决定)
```

这就是「split-K attention」——和 FlashAttention 论文里的 split-K 概念一致,只是 SGLang 把它用来分别处理"新 K"和"prefix K"。

---

## 八 命中的前缀什么时候参与 attention(关键澄清)

> 📖 **想深入「RadixCache 是怎么管理这些 prefix 的」**,见两篇配套文章:
> - [RadixCache 数据结构详解](radix-cache-structure.zh.md) — 树形 / TreeNode / match_prefix / split / lock_ref
> - [一个请求在 RadixCache 视角下的完整生命周期](request-lifecycle-radix-cache.zh.md) — T0~T7 全程跟踪 `req.prefix_indices` / `req.last_node` / `extend_prefix_lens` 的变化

你的疑问很对:**`forward_extend` 的参数 `q, k, v` 只包含「本批新 token」,不含已命中的 prefix**。那 prefix 怎么参与 attention 的?**答案是 prefix 从来不会被复制进 k, v 张量,它一直待在 KV cache 池里**——attention kernel 通过两种方式之一让它"参与":

### 8.1 关键事实:prefix 一直在 cache 池里,不需要被"加进 q/k/v"

回顾 [`_prefetch_kvcache`](kvcache-prefetch-and-storage.zh.md) 文章——**当一个新请求命中 RadixCache 前缀时,scheduler 在请求入队阶段已经把 prefix 的物理 KV slot 关联到本请求**:

1. **RadixCache 命中**:scheduler 发现新请求的 prompt 前 500 个 token 已经在 cache 里(可能是另一个请求留下的,可能是 HiCache 预取来的)
2. **关联(不复制)**:scheduler 通过 `req_to_token_pool[req_idx]` 这张映射表,**把那 500 个物理 slot 的编号"挂到"本请求名下**——零拷贝,只是改了一张 int 表
3. **后续算 attention 时**:本请求的 KV "历史"就是这 500 个 slot,**它们的 K、V 数据物理上一直在 `k_buffer[layer]` / `v_buffer[layer]` 里**

所以 `forward_extend` 被调到时:
- **`q, k, v` 这三个参数**:只装"本批新 token"(prompt 剩余的、需要新算的部分)
- **prefix 的 K、V**:**不出现在参数里**,但在 `forward_batch.token_to_kv_pool` 那个共享对象里随时可读
- **`forward_batch.extend_prefix_lens`**:告诉 wrapper "这个 req 在 cache 里有这么多 token 的历史"

### 8.2 两种"让 prefix 参与"的方式

#### 方式 ①:Paged 全包(分支 A,`use_ragged=False`)

```python
# 先写新 K/V 到 cache —— 此时 cache 里同时有 prefix + 本批新 K/V,完整连续
forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v, ...)

# attention 一次性从 cache 读「prefix + new」整段
o = prefill_wrapper_paged.forward(
    q,
    forward_batch.token_to_kv_pool.get_kv_buffer(layer.layer_id),   # 整个 KV pool
    causal=True, ...
)
```

**关键**:wrapper 的 plan 阶段(见 §6.2.1)用 `seq_lens = prefix_len + extend_len`(总历史长度)和 `paged_kv_indices = [prefix 的 page id... | new 的 page id...]` 配置好,kernel 自动按完整序列长度跑 attention。**对 kernel 来说,prefix 和 new 没有差别,都是"cache 里的 KV"**。

#### 方式 ②:Ragged + Paged 分两路 + LSE 合并(分支 C,`use_ragged=True` 有 prefix)

```python
# ── 第一路:新 K/V(ragged 张量,直接消费 forward_extend 输入)──
o1, s1 = prefill_wrapper_ragged.forward_return_lse(q, k, v, causal=True, ...)

# ── 第二路:prefix K/V(在 paged cache 里,kernel 通过页表读)──
o2, s2 = prefill_wrapper_paged.forward_return_lse(q, cache, causal=False, ...)

# 用 log-sum-exp 数学合并两段结果
o, _ = merge_state(o1, s1, o2, s2)
```

**关键**:**q 是同一个 q**(本批新 token 的 query)在两路都用到,只是分别对"新 K"和"prefix K"算 attention,LSE 合并保证数学等价于「把 prefix 和 new K 拼起来一次性算」。

### 8.3 数学等价性证明(为什么 merge_state 合法)

把 prefix 的 key 记作 `K_p`,新 token 的 key 记作 `K_n`,对应 value 同理。**朴素做法**:

```
o = softmax([q·K_p^T | q·K_n^T] / √d) · [V_p; V_n]
```

记 `s_p = q·K_p^T / √d`,`s_n = q·K_n^T / √d`(都是行向量)。softmax 等价于:

```
o = [exp(s_p) | exp(s_n)] / Z · [V_p; V_n]
  = (exp(s_p)·V_p + exp(s_n)·V_n) / Z          其中 Z = sum(exp(s_p)) + sum(exp(s_n))
```

**分别算**:

```
o_p = softmax(s_p) · V_p = exp(s_p)·V_p / Z_p     LSE_p = log(Z_p) = log(sum exp(s_p))
o_n = softmax(s_n) · V_n = exp(s_n)·V_n / Z_n     LSE_n = log(Z_n)
```

**merge_state 用 LSE 重新归一化**:

```
α = Z_p / (Z_p + Z_n) = exp(LSE_p) / (exp(LSE_p) + exp(LSE_n))

o = α · o_p + (1-α) · o_n
  = (Z_p · o_p + Z_n · o_n) / (Z_p + Z_n)
  = (exp(s_p)·V_p + exp(s_n)·V_n) / Z          ★ 和朴素做法相等
```

**所以 LSE 合并是无损的**——不是近似,是严格数学等价。两路 attention 可以并行算(GPU 多 SM 上并行),最后只需要一次轻量的 elementwise 合并。

### 8.4 为什么搞这么麻烦?分两路有什么好处

为何不直接走分支 A "先写 cache 再 paged 全包"?**两个工程原因**:

1. **延后写 cache,让 attention 的写依赖更少**:分支 A 必须先 `set_kv_buffer` 把新 K/V 写完才能跑 attention,**两个 GPU 操作有 RAW 依赖**;分支 C 把"算 attention"和"写 cache"解耦,**两者可以并行 / 重排**,GPU stream 利用率更高
2. **新 K/V 的布局优势**:本批新 K/V 是 ragged 紧凑张量,**显存访问是连续的 + 不经过页表间接寻址**,kernel 跑得比"从 paged cache 读这部分"略快。把热数据(新 K/V)用最快的方式算掉,冷数据(prefix)走 paged 走个保险

代价是要多写一段 LSE 合并逻辑——但 merge_state 本身就是个小 elementwise kernel,几 μs,**得不偿失就是值**。

### 8.5 prefix 是什么时候被放进 cache 的(回到时间线)

把整个生命周期串起来:

```
T1:  请求 R1 第一次到来(prompt = "AAA + BBB",长度 2000)
       └─ forward_extend(q=2000个token, k, v) 走分支 B(extend_no_prefix=True)
            └─ prefill_wrapper_ragged 算 attention
            └─ set_kv_buffer 把全部 2000 个 K/V 写进 cache pool
       └─ R1 释放后,RadixCache 把这 2000 个 slot 标记为"可共享"

T2:  请求 R2 到来(prompt = "AAA + CCC",前 1500 个 token 和 R1 一致)
       └─ scheduler 在 RadixCache 里搜:发现前 1500 个 token 已存在
       └─ req_to_token_pool[R2.req_idx][0:1500] = R1 留下的那 1500 个物理 slot
       └─ R2 真正需要新算的只有 "CCC" 这部分,假设 500 个 token
       └─ forward_extend(q=500个token, k=500, v=500) 走分支 C 或 A
            ├─ 分支 A:set_kv_buffer 写新 500 个 → paged.forward(q=500, full_cache_len=2000)
            └─ 分支 C:ragged.forward(q=500, k_new=500, v_new=500)  ←─ 新 token 内部 attention
                      paged.forward(q=500, prefix_cache_len=1500)   ←─ 对 prefix 的 attention
                      merge_state                                    ←─ LSE 合并
```

**核心观察**:R2 阶段 `forward_extend` 收到的 `q, k, v` 只有 500 个 token,但实际 attention 计算覆盖了完整的 2000 个 token——**少出来的 1500 个 prefix token 由 cache 池提供**,根本不经过函数参数。

### 8.6 一句话总结

> **prefix 不会被"加进"q, k, v——它一直在 KV cache 池里**。`forward_extend` 参数只装本批新 K、V,**通过 `forward_batch.extend_prefix_lens` 告诉 wrapper「这个 req 在 cache 里还有这么多历史」**;wrapper 要么 ① 先写新 K/V 让 cache 完整,然后一次性 paged.forward 读 prefix+new,要么 ② 分两路对 new (ragged) 和 prefix (paged) 各跑一次,最后用 log-sum-exp 数学等价地合并——这就是命中前缀如何"参与"attention 的全部机制。

---

## 九 一次完整 forward_extend 的形状追踪(分支 B 为例)

batch=2,prompt 长度 [128, 64],Llama-3-8B,TP=1:

```
total_tokens = 128 + 64 = 192

输入:
  q:            [192, 4096]       (32 Q 头 × 128 head_dim)
  k:            [192, 1024]       (8 KV 头 × 128 GQA)
  v:            [192, 1024]
  cache_loc:    [192]              本批每个 token 的物理 KV slot 编号
  layer.layer_id: 比如 5(第 6 层)

prefill_wrapper_paged 选定:self.forward_metadata.prefill_wrappers[idx]
use_ragged = True, extend_no_prefix = True  → 走分支 B

调 prefill_wrapper_ragged.forward:
  q.view(-1, 32, 128)              [192, 32, 128]
  k.view(-1, 8, 128)               [192, 8, 128]
  v.view(-1, 8, 128)               [192, 8, 128]
  →  FlashInfer kernel 计算
  →  output o ∈ [192, 32, 128]

写 KV cache:
  token_to_kv_pool.k_buffer[5][cache_loc] ← k         (scatter)
  token_to_kv_pool.v_buffer[5][cache_loc] ← v

返回:
  o.view(-1, 32 × 128) = [192, 4096]   ← 给 LlamaAttention.o_proj
```

---

## 十 一图串起所有对象关系

```
forward_extend(q, k, v, layer, forward_batch, save_kv_cache)
   │
   ├─ self.forward_metadata                            ← PrefillMetadata
   │     ├─ .prefill_wrappers[idx]                     ← BatchPrefillWithPagedKVCacheWrapper (FlashInfer 库)
   │     ├─ .use_ragged             ← True/False       ← Ragged 还是 Paged 路径
   │     └─ .extend_no_prefix       ← True/False       ← 是否有历史前缀
   │
   ├─ forward_batch                                    ← ForwardBatch
   │     ├─ .out_cache_loc          ← [total_tokens]   ← 这批新 token 写到 KV pool 哪里
   │     ├─ .token_to_kv_pool       ← MHATokenToKVPool ← KV cache 池本体
   │     │     ├─ .k_buffer / .v_buffer  [layer_num × (size, head_num, head_dim)]
   │     │     ├─ .set_kv_buffer(layer, loc, k, v, ...)  ← 写 cache 的入口
   │     │     └─ .get_kv_buffer(layer_id)               ← 读 cache 的入口
   │     └─ .extend_prefix_lens     ← 每个 req 的前缀长度
   │
   ├─ layer                                            ← RadixAttention
   │     ├─ .layer_id                                  ← 当前层号
   │     ├─ .tp_q_head_num / .tp_k_head_num / .tp_v_head_num
   │     ├─ .head_dim / .scaling                       ← 1/√d_k
   │     ├─ .sliding_window_size                       ← Llama 通常 -1(关闭)
   │     └─ .k_scale_float / .v_scale_float            ← KV cache 量化时用
   │
   └─ 调用栈:
        └─ prefill_wrapper_paged.forward(...) (FlashInfer Python)
              └─ C++ binding (csrc/)
                   └─ CUDA kernel (paged attention,FA-style)
                        └─ shared memory tile + softmax + V 加权累加
```

---

## 十一 一句话总结

> **`FlashInferAttnBackend.forward_extend` = 「拿 PrefillMetadata 里的 wrapper + 写 KV cache + 调 FlashInfer kernel 算 softmax(QKᵀ/√d)·V」**。`set_kv_buffer` 是 **`MHATokenToKVPool`**(继承自 `KVCache`)的方法,真正持有 GPU HBM 上的 K/V 张量;`use_ragged` 表示 **K/V 来源是「输入张量(ragged 布局)」还是「KV cache 页表(paged 布局)」**,有历史前缀时会同时跑两条然后用 LSE 合并;`prefill_wrapper_paged.forward` 的实现**不在 SGLang,在外部 `flashinfer` 项目**(github flashinfer-ai/flashinfer),底层是手写的 paged attention CUDA kernel。
