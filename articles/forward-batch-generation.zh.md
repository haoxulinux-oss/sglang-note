# `forward_batch_generation()` 解析(初学者向)

**位置**:`python/sglang/srt/managers/tp_worker.py:443`(标准 GPU 路径)

**角色**:这是 scheduler 把一个组好的 batch 真正喂给模型跑 forward 的入口函数。它做四件事:**搭好 ForwardBatch → 跑模型 forward → 采样下一个 token → 打包结果返回**。

> 配套阅读:看父函数怎么调它,见 [`run_batch()` 解析](scheduler-run-batch.zh.md) §六。

---

## 一 它在整个调用链里的位置

```
event_loop_normal()  (scheduler 主循环)
   │
   └─ run_batch(batch)                                (scheduler.py)
        │
        └─ self.model_worker.forward_batch_generation(model_worker_batch)   ← 本文
             │
             ├─ ① ForwardBatch.init_new(...)         构造一次 forward 的「输入参数包」
             │
             ├─ ② model_runner.forward(forward_batch) 真正跑模型(matmul/attention/...)
             │       │
             │       └─ model.forward(...)             具体模型(Qwen/Llama 等)
             │
             ├─ ③ model_runner.sample(logits, ...)    从 logits 采样 next_token
             │
             └─ ④ 包装成 GenerationBatchResult 返回
```

「**worker**」是 SGLang 里负责「把一个 batch 喂给模型」的角色;在 SGLang 中它和 scheduler 是**同一个进程**,只是逻辑上分工不同。

---

## 二 输入和输出

### 2.1 输入:`model_worker_batch`(ModelWorkerBatch 对象)

类型 `ModelWorkerBatch`——一个**纯数据 dataclass**,装着 forward 所需的全部信息:

| 字段 | 含义 | 形状 |
|---|---|---|
| `input_ids` | 这一批要跑的 token id | `[total_tokens]`(prefill) 或 `[bs]`(decode) |
| `positions` | 每个 token 的位置编号(给 RoPE 用) | 同上 |
| `seq_lens` | 每个请求当前的总长度 | `[bs]` |
| `req_pool_indices` | 每个请求在 KV 池里的 slot 编号 | `[bs]` |
| `out_cache_loc` | 这一批新 token 的 KV 要写到 KV 池的哪几个位置 | 一段 index |
| `forward_mode` | EXTEND(prefill) / DECODE / IDLE / ... | enum |
| `sampling_info` | 每个请求的采样参数(temperature/top_k/top_p/penalty/...) | per-req |
| `return_logprob` | 是否返回 logprob | bool |
| `lora_id` | 每个请求用的 LoRA adapter id(没 LoRA 时是 None) | per-req |
| `is_prefill_only` | embedding/reward 等只 prefill 不 decode 的标志 | bool |
| `hicache_consumer_index` | 这一批读的 KV 来自 HiCache 哪个版本 | int |

为什么不直接传 `ScheduleBatch` 而要拍平成这个?
- `ScheduleBatch` 含调度专用字段(time_stats、tree_cache_node 等),worker 用不上。
- 拍平成 dataclass 更适合**跨 stream / 跨进程 / 跨 rank** 高效传递。

### 2.2 输出:`GenerationBatchResult`

也是一个 dataclass:

| 字段 | 含义 |
|---|---|
| `next_token_ids` | 这一批每个请求生成的下一个 token id,形状 `[bs]` |
| `logits_output` | 最后一层的 logits(算 logprob 用) |
| `can_run_cuda_graph` | 这一批是不是用 CUDA Graph 跑的 |
| `expert_distribution_metrics` | MoE 专家路由统计(MoE 模型才有) |
| `routed_experts_output` | MoE 路由细节(MoE 才有) |
| `delay_sample_func` | 延迟采样 closure(spec/grammar 路径用) |
| `pp_hidden_states_proxy_tensors` | PP 中间 stage 用,传给下一 stage 的 hidden states |

scheduler 拿到这个对象后,`process_batch_result` 把里面的 `next_token_ids` 拼到对应请求,触发 stream_output。

---

## 三 函数主体:四步走

源码骨架(去掉边界处理):

```python
def forward_batch_generation(self, model_worker_batch, ...):
    # ① 构造 ForwardBatch
    self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)
    forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)

    if self.is_dllm():
        return self._forward_batch_generation_dllm(forward_batch)

    if self.pp_group.is_last_rank:
        # ② 跑模型 forward
        out = self.model_runner.forward(forward_batch, ...)
        logits_output = out.logits_output
        batch_result = GenerationBatchResult(logits_output=logits_output, ...)

        # ③ 采样 next_token
        if not model_worker_batch.is_prefill_only:
            batch_result.next_token_ids = self.model_runner.sample(
                logits_output, forward_batch
            )
        else:
            batch_result.next_token_ids = torch.zeros(...)   # prefill-only 不采样
            ...

        # ④ 返回
        return batch_result
    else:
        # PP 非末端 stage:只跑 forward,不 sample,把 hidden states 传给下一 stage
        out = self.model_runner.forward(forward_batch, ...)
        return GenerationBatchResult(
            pp_hidden_states_proxy_tensors=out.logits_output,  # 传 hidden,不是 logits
            ...
        )
```

下面逐步展开。

### 3.1 ① 构造 ForwardBatch

```python
self.set_hicache_consumer(model_worker_batch.hicache_consumer_index)
forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
```

#### `ModelWorkerBatch` vs `ForwardBatch`——别混淆

它们看似都是「forward 的输入包」,但有明确分工:

| | `ModelWorkerBatch` | `ForwardBatch` |
|---|---|---|
| 角色 | scheduler → worker 之间的传输格式(纯数据) | 跑 forward 时模型实际看到的对象(含 GPU tensor + attn metadata) |
| 字段类型 | 大多是 list / int / 简单 tensor | 经过 build 的 GPU tensor、attn backend 元信息 |
| 何时构造 | scheduler 端 `batch.get_model_worker_batch()` | worker 端 `ForwardBatch.init_new(...)` |
| 跨 rank | 适合 broadcast(已 pickle 友好) | 不跨 rank,只在本进程用 |

`ForwardBatch.init_new` 干的事:

1. 把 list 转成 GPU tensor(input_ids、positions、seq_lens 等)。
2. 算出 attention backend 需要的 metadata(KV page table、causal mask 边界、qo/kv 索引)。
3. 根据 model 是 MoE / 非 MoE 构造对应的 expert 路由信息。
4. 准备 sampler 工作区(温度向量、惩罚累计向量等)。

简单理解:**ModelWorkerBatch 是「材料清单」,ForwardBatch 是「装好的工具箱」,装工具箱发生在这一步**。

#### `set_hicache_consumer`

HiCache 是 SGLang 的多层 KV cache(GPU/RAM/SSD)。它内部的元信息会随着加载/写入更新版本号。`set_hicache_consumer` 告诉 worker「**这一批 forward 读的是 HiCache 的第 N 版**」,避免跑到一半被并发写入搅乱。

### 3.2 ② 跑 model.forward

```python
out = self.model_runner.forward(forward_batch, ...)
logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
```

这是函数里**唯一真正在 GPU 上算东西的一步**——上面 ①、下面 ③④ 加起来 CPU 时间通常 < 1 ms,这一步占绝对大头。

> 📖 **想深入看 `model_runner` 这个对象本身**:[`ModelRunner` 是什么(成员介绍)](model-runner-overview.zh.md);**想详细读 `forward()` 方法的决策树**:[`ModelRunner.forward()` 方法详解](model-runner-forward.zh.md)。

`model_runner.forward` 的内部:

1. **决定走 CUDA Graph 还是 eager 模式**:
   - 如果 batch_size 命中已 capture 的 CUDA Graph(典型是 decode 阶段的 1/2/4/8/...)→ `graph.replay()` 一条 launch 跑完。
   - 否则走 eager 模式:逐个 layer 调用 `model.forward`(可能几千个 kernel)。
   - 详见 [CUDA Graph 是什么](basics-cuda-graph.zh.md)。

2. **逐层跑 transformer**:
   ```
   x = embed(input_ids)
   for layer in 28 层:
       residual = x
       x = layernorm(x)
       q, k, v = attn_proj(x)              ← 这里产生 K、V (要存到 KV cache)
       attn_out = attention(q, k, v, KV_cache)  ← 用 attn backend (FlashInfer/FA3/Triton)
       x = residual + attn_out
       residual = x
       x = layernorm(x)
       x = mlp(x)                           ← MoE 模型这里会路由到 expert
       x = residual + x
   logits = lm_head(x)                      ← 词表大小(几十万维)
   ```

3. **写入 KV cache**:每层 attention 把新算出的 K、V 写到 `out_cache_loc` 指定的 KV pool 位置(为下次 decode 准备)。

返回 `logits_output`(里面有 `next_token_logits`,形状 `[bs, vocab_size]`)。

### 3.3 ③ 采样 next_token

```python
if not model_worker_batch.is_prefill_only:
    batch_result.next_token_ids = self.model_runner.sample(
        logits_output, forward_batch
    )
```

`sample` 函数把 logits 变成具体 token id:

1. **应用 sampling_info**:
   - 加 frequency / presence / repetition penalty。
   - 应用 temperature(`logits / T`)。
   - top_k / top_p / min_p 过滤。
   - structured output 的 grammar mask(把不合法的 token 概率压成 -inf)。
2. **softmax → 抽样**:
   - greedy(temperature=0):取 argmax。
   - 否则:multinomial 抽样。
3. 返回 `next_token_ids`(`[bs]` 形状)。

#### prefill-only 的特殊路径

```python
else:
    batch_result.next_token_ids = torch.zeros(
        len(model_worker_batch.seq_lens), dtype=torch.long, ...
    )
    if model_worker_batch.return_logprob and ...:
        self.model_runner.compute_logprobs_only(logits_output, model_worker_batch)
```

embedding / classifier / `max_new_tokens=0` 这种「prefill-only」请求(见 [为什么会有 prefill-only 请求](#) ←参见 update_running_batch 那篇)**不需要采样**——填一组 0 占位就行。需要 logprob 的话单独算。

### 3.4 ④ delay_sample_func 路径(可跳过)

```python
if (
    self.enable_overlap
    and not self.enable_spec
    and model_worker_batch.sampling_info.grammars is not None
):
    def sample_batch_func():
        batch_result.next_token_ids = self.model_runner.sample(...)
        return batch_result

    batch_result.delay_sample_func = sample_batch_func
    return batch_result
```

启用 overlap + grammar 约束(JSON schema 等结构化输出) 时,采样要**延迟到下一轮 forward 之后**——因为 grammar 状态机要等到下一轮才能确定。这里不真采样,把 closure 存起来,scheduler 后面 `launch_batch_sample_if_needed` 时再调用。详见 [run_batch §3.3.6](scheduler-run-batch.zh.md)。

初学阶段忽略这条路径即可。

---

## 四 PP > 1 时的非末端 stage

```python
if self.pp_group.is_last_rank:
    ...                  # 上面 ①②③④
else:
    out = self.model_runner.forward(...)
    return GenerationBatchResult(
        pp_hidden_states_proxy_tensors=out.logits_output,
        ...
    )
```

PP(pipeline parallel)流水线把模型按层段切到不同 rank:

| stage | 跑的层 | 这一批 forward 输出 | 给谁 |
|---|---|---|---|
| stage 0(`pp_rank=0`) | layer 0..N/3 | hidden_states | send 给 stage 1 |
| stage 1(中间) | layer N/3..2N/3 | hidden_states | send 给 stage 2 |
| stage 2(`pp_rank=last`) | layer 2N/3..N + lm_head | logits → next_token_ids | 走完整 ②③④ |

**只有最后一个 stage 跑 sample**——因为只有它有 lm_head,能产出 logits。其他 stage 只跑 forward,把 hidden_states 通过 PP 通信传给下一 stage。

返回的 `pp_hidden_states_proxy_tensors` 不是真 logits,是 hidden states——上层 scheduler 再通过 PP 通信送给下一 rank 接力。

初学者只关心单 rank / 单机部署的话,这一支可以暂时跳过。

---

## 五 小例子:Qwen2.5-7B 跑一次 decode

假设 batch 里 4 个请求都在 decode 阶段:

```
输入 model_worker_batch:
  input_ids       = [上一步 4 个 token,形状 [4]]
  positions       = [当前 4 个位置]
  seq_lens        = [256, 312, 128, 480]   各请求当前总长
  forward_mode    = DECODE
  is_prefill_only = False
  sampling_info   = {temperature=0.7, top_k=40, ...} per-req

执行:
  ① ForwardBatch.init_new
       - 把 list 转 GPU tensor,准备 attn backend metadata
       - 时间:< 1 ms

  ② model_runner.forward
       - batch_size=4 命中 CUDA Graph,replay
       - 28 层 transformer 一次跑完
       - 输出:logits_output.next_token_logits [4, 152064]
       - 每层 K、V 写入 KV pool
       - 时间:几个 ms

  ③ model_runner.sample
       - 应用 temperature/top_k/top_p
       - multinomial 采样
       - 输出:next_token_ids = [token_a, token_b, token_c, token_d]
       - 时间:< 1 ms

  ④ 返回 GenerationBatchResult(next_token_ids=..., logits_output=...)
```

整个调用大约几 ms,大头是 ②。

---

## 六 一句话总结

> `forward_batch_generation` 是 scheduler 把一个组好的 batch 喂给模型的**真正干活的入口**。流程是「**ForwardBatch.init_new(打包)→ model_runner.forward(跑模型)→ model_runner.sample(采样 next_token)→ 包装返回**」四步。
>
> 第二步是耗时大头(几乎所有 GPU 计算都发生在这里),其他步骤都是几百微秒级的薄壳代码。
>
> 注意区分:**ModelWorkerBatch 是「材料清单」**(scheduler 端打包,跨 stream 友好),**ForwardBatch 是「装好的工具箱」**(GPU tensor + attn metadata,在 worker 端构造、forward 实际看到的对象)——本函数第一步就是把前者转成后者。
