# `forward_mode` 是什么

`forward_mode` 是 SGLang 中 **标识本批 forward 属于哪种类型** 的枚举。它决定 `model_runner` / attention backend / sampler 怎么解释这一批的输入,以及走哪条 CUDA graph。

定义在 `python/sglang/srt/model_executor/forward_batch_info.py` 的 `ForwardMode`(IntEnum)。

---

## 一 主要取值

| 取值 | 含义 | 何时出现 |
|---|---|---|
| `EXTEND` | **prefill**:本批每个请求要喂入多个新 token,attend 所有历史 KV,产生第一个(或下一个 chunk 的)输出 | 新请求 prefill / chunked prefill 中间轮次 / session 续 prefill |
| `DECODE` | **decode**:每个请求只喂 1 个 token(上一步生成的),自回归生成下一个 | 普通解码 |
| `IDLE` | **空跑**:本 rank 没活,但要陪跑通信原语 | DP attention 某些 rank 没请求时 |
| `TARGET_VERIFY` | spec decode 里 target model 一次校验多个 draft token | speculative decoding |
| `DRAFT_EXTEND` | spec decode 的 draft model 跑一次扩展 | EAGLE / 多层 draft |
| `SPLIT_PREFILL` | 长 prefill 切片(配合 pdmux,让出 GPU 给 decode) | `enable_pdmux` 模式 |
| `PREBUILT` | KV 已经从 prefill 节点传过来,本批走占位流程不真跑 forward | PD 解耦 decode 节点 |
| `DUMMY_FIRST` | 第一次 forward 的 warmup placeholder | 启动期 / CUDA graph 预热 |

实际代码里常用判定方法:

```python
batch.forward_mode.is_extend()     # prefill 路径(EXTEND / SPLIT_PREFILL / DRAFT_EXTEND)
batch.forward_mode.is_decode()     # decode 路径(DECODE / TARGET_VERIFY)
batch.forward_mode.is_idle()
batch.forward_mode.is_prebuilt()
```

`is_extend()` 是个**伞概念**——它把所有「这一批要 attend 多个新 token」的模式都囊括进来,因为它们的 attention metadata 形状一致(`extend_input_lens` per req)。`is_decode()` 同理。

---

## 二 为什么要这个字段

prefill 和 decode 在**形状**和**算法**上差别巨大:

| 维度 | EXTEND(prefill) | DECODE |
|---|---|---|
| `input_ids` 形状 | `[total_extend_tokens]`(每 req 多 token) | `[batch_size]`(每 req 1 token) |
| `positions` | 每 req 一段连续 position | 每 req 1 个 position |
| attention 类型 | causal mask + 新 token attend 历史 KV | 1 query attend 所有历史 KV |
| KV 写入 | 写入这一批所有新 token 的 KV | 只写入这一个新 token 的 KV |
| CUDA graph | 不同 batch shape 各 capture 一份 | 按 batch_size capture |
| sampler | 通常每 req 取最后一个位置的 logits | 每 req 取唯一那个 logits |

`forward_mode` 就是这条 if/else 的分派 key。

---

## 三 在 SGLang 代码里的作用

### 3.1 调度路径

```python
# scheduler.py
if batch.forward_mode == ForwardMode.EXTEND:
    set_time_batch(batch.reqs, "set_prefill_run_batch_start_time")
if batch.forward_mode.is_prebuilt():
    return self._run_batch_prebuilt(batch)
```

### 3.2 `process_batch_result` 分派

```python
if batch.forward_mode.is_decode():
    self.process_batch_result_decode(batch, result)
elif batch.forward_mode.is_extend():
    if batch.is_dllm():
        self.process_batch_result_dllm(batch, result)
    elif self.disaggregation_mode == DisaggregationMode.PREFILL:
        self.process_batch_result_disagg_prefill(batch, result)
    else:
        self.process_batch_result_prefill(batch, result)
elif batch.forward_mode.is_prebuilt():
    self.process_batch_result_prebuilt(batch)
elif batch.forward_mode.is_idle():
    self.process_batch_result_idle(batch, result)
```

### 3.3 attention backend 选 metadata

```python
# 伪代码
if forward_mode.is_extend():
    metadata = build_extend_attn_metadata(seq_lens, extend_input_lens, ...)
elif forward_mode.is_decode():
    metadata = build_decode_attn_metadata(seq_lens, ...)
```

FlashInfer / FA3 / Triton 各自的 prefill kernel 和 decode kernel 是不同的入口。

### 3.4 batch 准备

```python
batch.prepare_for_extend()    # prefill 路径用,组 input_ids/positions
batch.prepare_for_decode()    # decode 路径用,seq_lens += 1,只取上一步 token
```

### 3.5 DP attention 同步决策

```python
is_extend = lambda b: b and b.forward_mode.is_extend()

# 各 DP rank 必须一致地 prefill 或一致地 decode,不然 all-gather 死锁
need_mlp_sync = ...
```

### 3.6 overlap schedule 决定要不要禁用

```python
# 连续两个 prefill 不 overlap(为了首个 prefill 的 TTFT)
disable_overlap_for_batch = (
    SGLANG_DISABLE_CONSECUTIVE_PREFILL_OVERLAP
    and batch_is_extend
    and last_batch_is_extend
)
```

---

## 四 一个请求一生中 forward_mode 的变迁

```
新请求进队
    │
    ▼
EXTEND (chunk 1)  ─┐
EXTEND (chunk 2)   │  chunked prefill
EXTEND (chunk 3)  ─┘
    │
    ▼
DECODE  step 1 ─┐
DECODE  step 2  │
DECODE  step 3  │  自回归生成
  ...           │
DECODE  step N ─┘
    │
    ▼
finish(EOS / max_new_tokens / stop_str)
```

如果带 spec decode:中间会插 `DRAFT_EXTEND` 和 `TARGET_VERIFY`。
如果是 PD 解耦的 decode 节点:开头先有一个 `PREBUILT`(等 KV 接收完)。

---

## 五 一句话总结

> `forward_mode` 是 batch 的「类型标签」,告诉下游(attention backend、sampler、process_batch_result、CUDA graph 选择器)**这一批 forward 是 prefill、decode、idle、还是 spec/disagg 的特殊变体**——因为它们的输入形状和处理逻辑完全不同,必须显式分派。
