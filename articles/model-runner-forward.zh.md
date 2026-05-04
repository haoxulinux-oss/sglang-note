# `ModelRunner.forward()` 方法详解(初学者向)

> 📖 **关联阅读**:
> - 先读 [`ModelRunner` 是什么(成员介绍)](model-runner-overview.zh.md) 知道每个字段是啥再回来
> - 上游:[`forward_batch_generation()` 解析](forward-batch-generation.zh.md) 里把 `forward_batch` 交给本方法
> - 兄弟:`run_batch()` §3.3 [overlap schedule](scheduler-run-batch.zh.md) 解释 forward 是怎么和调度并行的

代码位置:`python/sglang/srt/model_executor/model_runner.py:2896` 是入口,真正干活的是 `_forward_raw`(`:2964`)。

---

## 一 调用入口和外层包装(`forward()`)

被调用的样子:

```python
# forward_batch_generation() 内
out = self.model_runner.forward(
    forward_batch,
    pp_proxy_tensors=pp_proxy_tensors,
    skip_attn_backend_init=skip_attn_backend_init,
)
```

源码骨架(精简后):

```python
def forward(self, forward_batch, skip_attn_backend_init=False,
            pp_proxy_tensors=None, reinit_attn_backend=False,
            split_forward_count=1) -> ModelRunnerOutput:

    self.forward_pass_id += 1                   # ① 计数 +1

    with step_span_ctx, get_global_expert_distribution_recorder() \
            .with_forward_pass(self.forward_pass_id, forward_batch) as recorder:

        output = self._forward_raw(...)          # ② ★真正干活★

        # ③ 弹性 EP:rank 故障后重新平衡专家,然后再跑一次
        if elastic_ep_state and not elastic_ep_state.is_active_equal_last():
            ... rebalance ...
            output = self._forward_raw(...)

    output.expert_distribution_metrics = recorder.get("metrics")
    output.routed_experts_output = ...           # ④ MoE 专家路由记录(若开启)
    if self.eplb_manager: self.eplb_manager.on_forward_pass_end()
    if dumper.may_enable: dumper.step()
    return output
```

**初学者只需记住**:`forward()` 这层的核心就是调一次 `_forward_raw()`。其他一堆代码都是"周边设施"——profiler 计数、MoE 专家分布记录、弹性 EP 容错。**第一遍读直接跳到 `_forward_raw`**。

---

## 二 `_forward_raw()` 的整体决策树

这是真正的"主体逻辑"。**核心就是一棵决策树**,根据 `forward_batch.forward_mode`(EXTEND / DECODE / IDLE / SPLIT_PREFILL)+ "能否走 CUDA Graph",分到不同分支:

```
_forward_raw(forward_batch)
│
├─【分支 A】can_run_graph?  ── 是 ──→ self.graph_runner.replay()  ★decode 默认路径★
│                                       └→ return ModelRunnerOutput(...)
│  否
│  ↓
├─【准备】MLP / TP scatter / num_token_non_padded normalize / SWA loc
│
└─【分支 B】按 forward_mode 派发:
       ├ DECODE       → self.forward_decode(...)        ★graph 不能用时的 decode★
       ├ EXTEND       → self.forward_extend(...)        ★prefill,可能走分段 graph★
       ├ SPLIT_PREFILL→ self.forward_split_prefill(...)  极长 prompt 分多次跑
       └ IDLE         → self.forward_idle(...)          DP 凑 batch 用的空跑
```

最终统一包成 `ModelRunnerOutput` 返回:

```python
@dataclass
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: ...
    routed_experts_output: ...
```

> `LogitsProcessorOutput` 里最关键的就是 **`next_token_logits`** 这个张量,后面 sample 阶段用它。

---

## 三 分支 A:CUDA Graph 重放(decode 主路径)

```python
mode_check = forward_batch.forward_mode.is_cuda_graph
can_run_graph = bool(
    mode_check()                              # forward_mode 允许 graph
    and self.graph_runner                     # 有录好的 graph
    and self.graph_runner.can_run(forward_batch)   # 当前 batch_size、seq_len 命中桶
)
if can_run_graph:
    ret = self.graph_runner.replay(
        forward_batch,
        skip_attn_backend_init=skip_attn_backend_init,
        pp_proxy_tensors=pp_proxy_tensors,
    )
    return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)
```

**为什么 decode 默认走这里?**

启动期 `init_device_graphs()` 已经预录了一系列 (batch_size, seq_len) 桶——比如 batch_size ∈ {1,2,4,8,16,32,...}。每个桶把"过 28 层 transformer + attention 计算"的 GPU kernel 序列固化成一个 CUDA Graph。运行期只要 `replay()`,不用每次从 Python 重新 launch kernel,**省掉每个 kernel ~5–10 μs 的 launch 开销**——decode 时 batch 里每个 token 单独算一次,这点开销累加起来很可观。

**触发条件三件套**(必须全满足):
1. `forward_mode.is_cuda_graph()`:不是所有模式都行,比如 SPLIT_PREFILL 就不行
2. `self.graph_runner` 不为 None(可能被 `--disable-cuda-graph` 关掉)
3. 当前 batch 的 size、序列长度等参数能命中预录桶

不命中就掉到下面分支 B(eager 模式)。

---

## 四 分支 B:按 `forward_mode` 派发

派发前先做几件准备工作:

```python
# DP attention 时同步 batch
if forward_batch.global_num_tokens_cpu is not None:
    forward_batch.prepare_mlp_sync_batch(self)
else:
    forward_batch.prepare_attn_tp_scatter_input(self)

# attention TP 下校准 num_token_non_padded
... adjust_num_token_non_padded_for_attn_tp(...) ...

# SWA(滑动窗口)缓存位置
if forward_batch.out_cache_loc_swa is not None:
    self.token_to_kv_pool.set_swa_loc(forward_batch.out_cache_loc_swa)
```

> 这些是"分布式同步 / 内存布局"层面的细节,**初学者第一遍可以直接跳过**。重点看下面 4 个 `forward_xxx` 分支里跑模型的部分。

下面 4 个分支函数,**结构出奇地一致**——都是"建 metadata → 调 self.model.forward()"。

### 4.1 `forward_decode()`(`:2782`)

```python
def forward_decode(self, forward_batch, skip_attn_backend_init=False, pp_proxy_tensors=None):
    if not skip_attn_backend_init:
        if self.server_args.enable_pdmux:
            self.decode_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.attn_backend = self.decode_attn_backend
        else:
            self.attn_backend.init_forward_metadata(forward_batch)   # ① 准备 attention 元数据

    kwargs = {}
    if self.support_pp:
        kwargs["pp_proxy_tensors"] = pp_proxy_tensors

    return self.model.forward(                                       # ② 真正过 28 层模型
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
        **kwargs,
    )
```

**两步**:
1. `attn_backend.init_forward_metadata(forward_batch)` —— 把这一 batch 里的 KV cache 索引、序列长度、page table 等信息塞给 attention 后端,attention kernel 后续从这里读
2. `self.model.forward(...)` —— 进 `LlamaForCausalLM.forward()`(或对应模型类),里面跑 28 层 transformer

### 4.2 `forward_extend()`(prefill,`:2805`)

跟 decode 几乎一样,**两点不同**:

1. **可能走分段 CUDA Graph**:
   ```python
   can_run_graph = (
       self.piecewise_cuda_graph_runner is not None
       and self.piecewise_cuda_graph_runner.can_run(forward_batch)
   )
   if can_run_graph:
       return self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs), can_run_graph
   ```
   prefill 序列长度变化大,整段录 graph 不现实,所以是"分段录"——把 transformer 切成几段,每段录一个 graph。这是较新的优化,关掉它也能跑。

2. **支持 input_embeds**(多模态用):如果传入了视觉 embedding,从 `forward_batch.input_embeds` 读,而不是从 `input_ids` 查 embedding 表。

### 4.3 `forward_idle()`(`:2856`)

DP attention 下,某些 rank 这一轮可能没活干,但因为 DP allreduce 要求所有 rank 同步,只能"空跑"凑数:

```python
if forward_batch.batch_size > 0:
    self.attn_backend.init_forward_metadata(forward_batch)   # 有 padding 的话还是要 init
return self.model.forward(forward_batch.input_ids, forward_batch.positions, forward_batch, **kwargs)
```

### 4.4 `forward_split_prefill()`(`:2875`,极长序列才走)

把一个 prompt 的 prefill 拆成多次 forward(每次跑模型的几层),用 `split_index` 记录跑到第几层。这是**和 chunked prefill 完全不同的概念**——chunked prefill 是按 token 分,split prefill 是按层分。**初学者基本接触不到**,跳过。

---

## 五 `self.model.forward()` 进去之后干啥

到这一步,控制权交给 `python/sglang/srt/models/llama.py`(或对应模型文件)里的 `LlamaForCausalLM.forward()`。它的本质就是经典 Transformer:

```python
def forward(self, input_ids, positions, forward_batch, **kwargs):
    hidden = self.embed_tokens(input_ids)            # 词嵌入

    for layer in self.layers[start_layer : end_layer]:   # ★ PP 切分体现在这里 ★
        # 每层内部:RMSNorm → self_attn → residual → RMSNorm → MLP → residual
        # self_attn 里调 forward_batch.attn_backend.forward(q, k, v, ...)
        #     attn_backend 负责:
        #       (a) 把 K/V 写入 token_to_kv_pool 对应位置
        #       (b) 用 paged attention 算 softmax(QK^T)V
        hidden = layer(hidden, positions, forward_batch)

    hidden = self.norm(hidden)
    logits = self.lm_head(hidden)                    # 最后一层 logits
    return LogitsProcessorOutput(next_token_logits=logits, ...)
```

**关键交互**:
- `attn_backend` ←─ attention 层的 Q/K/V → 计算 softmax(QK^T)V
- `token_to_kv_pool` ←─ 每层 attention 把 K、V 写进去,decode 时再读出历史 KV
- `req_to_token_pool` ←─ 用来知道每个 req 的历史 token 在 KV pool 的哪些位置

> 这一步细节庞大,涉及不同模型架构、不同注意力后端、不同 KV layout(MHA / GQA / MLA),**初学者只需知道"`self.model.forward` 就是过完整个 transformer 算出 logits"** 就够了。

---

## 六 一句话总结

> **`ModelRunner.forward()` = 一棵小决策树**:能走 CUDA Graph 就 `graph_runner.replay()` 直接重放;不能就按 `forward_mode` 走 `forward_decode / forward_extend / forward_idle / forward_split_prefill` 之一,这几个分支结构都一样:**先 `attn_backend.init_forward_metadata` 准备元数据,再调 `self.model.forward` 过完整个 transformer,返回 logits**。

回到上游:`forward_batch_generation()` 拿到 `ModelRunnerOutput.logits_output` 之后,会再调 `model_runner.sample()`(下一篇可看),把 logits 变成 next_token_ids。
