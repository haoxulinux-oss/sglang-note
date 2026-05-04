# `ModelRunner` 是什么(初学者向)

> 📖 **关联阅读**:
> - [`forward_batch_generation()` 解析](forward-batch-generation.zh.md) — 上游调用方,worker 把 `ForwardBatch` 交给 `model_runner.forward()`
> - [`ModelRunner.forward()` 方法详解](model-runner-forward.zh.md) — 本文是「类介绍」,那篇是「forward 方法详解」

代码位置:`python/sglang/srt/model_executor/model_runner.py:295`

---

## 一 一句话定位

**`ModelRunner` 是 worker 内部「真正抱着模型权重、负责把一个 batch 跑完一次 forward」的对象。**

回想 [`forward_batch_generation()` 解析](forward-batch-generation.zh.md) 里那段代码:

```python
if self.pp_group.is_last_rank:
    out = self.model_runner.forward(
        forward_batch,
        pp_proxy_tensors=pp_proxy_tensors,
        skip_attn_backend_init=skip_attn_backend_init,
    )
```

这里的 `self` 是 `TpModelWorker`,而 `self.model_runner` 就是这个 `ModelRunner` 实例。**worker 是"门面",model_runner 是"内核"**。

层级总览:

```
Scheduler 进程
└── TpModelWorker          (一层薄壳,处理 worker 间协议、ZMQ 等)
    └── ModelRunner        ← 本文主角
        ├── self.model              (真正的 nn.Module,例如 LlamaForCausalLM)
        ├── self.attn_backend       (FlashInfer / FA3 / Triton 注意力后端)
        ├── self.token_to_kv_pool   (KV cache 显存池)
        ├── self.req_to_token_pool  (req → token 索引)
        ├── self.graph_runner       (CUDA Graph 缓存,用于 decode 加速)
        ├── self.sampler            (top-k / top-p / 温度 等)
        └── ...
```

---

## 二 关键成员一览(初学者只需先认住前 6 个)

下面这张表把 `ModelRunner` 的字段按"作用层"分组。**初学者只要看懂前面打 ⭐ 的 6 个就够用了**,后面的可以用到再回来查。

| 字段 | 类型 / 来源 | 作用(用大白话讲) | ⭐ |
|---|---|---|---|
| `self.model` | `nn.Module`,例如 `LlamaForCausalLM` | **模型权重本体**。`self.model.forward()` 就是真正过 28 层 Transformer 的入口 | ⭐⭐⭐ |
| `self.attn_backend` | FlashInfer / FA3 / Triton 后端对象 | 注意力计算的实际执行者。`self.model.forward` 里的 attention 层最后会调到它 | ⭐⭐ |
| `self.token_to_kv_pool` | `MHATokenToKVPool` / MLA 等 | **KV cache 的物理显存池**(每层、每 token 的 K/V 都在这里) | ⭐⭐ |
| `self.req_to_token_pool` | `ReqToTokenPool` | 「第 i 号请求的第 j 个 token 在 KV pool 的哪一格」的索引表 | ⭐⭐ |
| `self.graph_runner` | `CudaGraphRunner` 或 `None` | **CUDA Graph 缓存**。decode 阶段的多数 batch_size 都有预录好的 graph,调用 `replay()` 直接重放 | ⭐⭐ |
| `self.sampler` | `Sampler` | 把 logits 变 token id(temperature/top-k/top-p/grammar mask) | ⭐ |
| `self.model_config` | `ModelConfig` | hf_config 包装。num_layers / num_heads / hidden_size / dtype 都从这里读 |   |
| `self.server_args` | `ServerArgs` | 启动参数。后面所有"是否启用 X 功能"都看这个 |   |
| `self.tp_rank / tp_size` | int | 当前进程在 TP 通信组里的位置 |   |
| `self.pp_rank / pp_size` | int | 当前进程在 PP 通信组里的位置 |   |
| `self.tp_group / pp_group` | 通信组对象 | TP allreduce、PP send/recv 走这里 |   |
| `self.start_layer / end_layer` | int | PP 切分:本进程负责模型的第几到第几层 |   |
| `self.forward_stream` | `torch.cuda.Stream` | overlap schedule 用的副 stream(详见 `run_batch` 文章 §3.3) |   |
| `self.spec_algorithm` | 枚举 | 投机解码算法(EAGLE / DFlash / NGRAM / NONE) |   |
| `self.piecewise_cuda_graph_runner` | 可选 | extend(prefill)阶段的"分段 CUDA Graph",较新功能 |   |
| `self.eplb_manager` | 可选 | MoE 模型的专家负载均衡 |   |
| `self.lora_manager` | 可选 | 启用 LoRA 时存放各 LoRA adapter |   |
| `self.expert_location_updater` | 可选 | MoE 专家位置更新 |   |
| `self.memory_saver_adapter` | 工具 | 显存节省辅助 |   |
| `self.hisparse_coordinator` | 可选 | 稀疏 KV cache 的协调器 |   |
| `self.eagle_aux_hidden_state_layer_ids` | list | EAGLE3 投机解码:从哪些层抽辅助隐状态 |   |
| `self.forward_pass_id` | int | 自增计数器,每跑一次 forward 加 1,profiler / dumper 用 |   |
| `self.use_mla_backend` | bool | 是否走 MLA(DeepSeek 系) |   |

---

## 三 ModelRunner 是什么时候被建出来的

构造在 `ModelRunner.__init__`(`model_runner.py:298`)里完成,主要四步,**一次性、启动时跑**:

```python
def __init__(self, ...):
    # ① 记参数:tp_rank / pp_rank / model_config / server_args / page_size 等
    ...
    # ② 初始化分布式 + forward stream
    pre_model_load_memory = self.init_torch_distributed()
    self.forward_stream = torch.get_device_module(self.device).Stream()

    # ③ 走总装流程
    self.initialize(pre_model_load_memory)         # 见下面展开
    ...
```

`self.initialize()`(`model_runner.py:529`)做的事(**重点**):

| 步骤 | 干啥 | 把谁挂到 self 上 |
|---|---|---|
| `create_sampler()` | 创建 token 采样器 | `self.sampler` |
| `self.load_model()` | **从磁盘加载模型权重到 GPU** | `self.model` |
| 推断 PP layer 范围 | 决定本进程跑模型的第几到第几层 | `self.start_layer / end_layer` |
| `init_lora_manager()` | LoRA 启用时建 adapter 表 | `self.lora_manager` |
| `init_memory_pool()` | **分配 KV cache 显存池** | `self.token_to_kv_pool / req_to_token_pool` |
| `init_attention_backend()` | 选 + 初始化注意力后端 | `self.attn_backend` |
| `init_cublas()` | cuBLAS 预热 | (副作用) |
| `kernel_warmup()` | 触发各 kernel JIT 编译 | (副作用) |
| `init_device_graphs()` | **录制 CUDA Graph(每个 batch_size 桶一个)** | `self.graph_runner / piecewise_cuda_graph_runner` |

**初学者要记住**:`self.model`、`self.token_to_kv_pool`、`self.attn_backend`、`self.graph_runner` 都是这一步生出来的。运行期不会再变(权重热更新除外)。

---

## 四 关键成员之间的关系图

```
                        ModelRunner
                   ┌─────────────────────┐
forward_batch  →   │  forward()          │
(请求批次数据)      │   └─→ _forward_raw  │
                   │       ├─→ graph_runner.replay()    (decode 多数走这条)
                   │       └─→ self.model.forward(...)   (extend / 不能跑 graph)
                   │              │
                   │              ↓
                   │    ┌─────────────────────────┐
                   │    │  Transformer 28 层 loop │
                   │    │  for layer in layers:   │
                   │    │    layer(hidden, ...)   │← 调到 attn_backend
                   │    │                         │← 读 / 写 token_to_kv_pool
                   │    └─────────────────────────┘
                   │              │
                   │              ↓ logits
                   │       sampler(logits) → next_token_ids
                   └─────────────────────┘
```

---

## 五 一句话总结

> **`ModelRunner = 模型权重 + KV cache 池 + 注意力后端 + CUDA Graph 缓存 + 采样器**;它是 worker 的内核,`forward_batch_generation()` 把 `ForwardBatch` 递进来,它负责跑出 logits 并采样出下一个 token。

具体 `forward()` 方法是怎么把这些零件串起来的,下一篇详细讲:**[`ModelRunner.forward()` 方法详解](model-runner-forward.zh.md)**。
