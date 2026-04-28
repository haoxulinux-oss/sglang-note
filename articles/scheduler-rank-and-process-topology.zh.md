# Scheduler / Worker 进程拓扑与各种 rank

本文回答两个相关问题:

1. SGLang 的 scheduler 进程和 worker 进程是什么关系?
2. SGLang 用了哪些并行维度(TP / PP / DP / CP / EP),每个 rank 是什么意思?

---

## 一 关键事实:scheduler 和 worker **是同一个进程**

很多推理引擎(早期 vLLM 等) 把 scheduler 和 worker 分到不同进程,通过 RPC 通信。**SGLang 不是这样**——

> **每个 scheduler 进程内部直接持有一个 `TpModelWorker` 实例作为 `self.tp_worker`**,worker 不是独立进程。

代码佐证(`scheduler.py:634-636`):

```python
from sglang.srt.managers.tp_worker import TpModelWorker
self.tp_worker = TpModelWorker(**worker_kwargs)
```

scheduler 主循环里调 worker 就是普通的 Python 方法调用:

```python
batch_result = self.model_worker.forward_batch_generation(model_worker_batch)
```

`TpModelWorker` 内部对 GPU 的访问通过当前进程的 CUDA context——也就是说,**一个 scheduler 进程绑定一张 GPU**(由 `gpu_id` 参数指定)。每个进程在它那张 GPU 上加载它负责的那一片模型权重(TP shard + PP stage)。

更准确的说法:

| 概念 | 实体 |
|---|---|
| **scheduler 进程** = **worker 进程** = **GPU rank 进程** | 一个 OS 进程,绑一张 GPU |
| 进程内的 scheduler 部分 | 跑 `event_loop_*`、组 batch、ZMQ 收发 |
| 进程内的 worker 部分 | 跑 `forward_batch_generation`、`model.forward`、CUDA kernel |
| 进程间通信 | **不是 RPC**,是 NCCL(GPU 集合通信) + ZMQ(控制消息) |

---

## 二 SGLang 的并行维度

SGLang 支持 5 种并行(可叠加),每种引入一个独立的 rank 编号:

| 维度 | 全称 | 切什么 | 通信原语 | 核心特征 |
|---|---|---|---|---|
| **TP** | Tensor Parallel | 层内权重(矩阵按列/行切) | NCCL all-reduce/all-gather(每层) | 同 TP 组完全同步,处理同一层 |
| **PP** | Pipeline Parallel | 模型按层段切 | point-to-point send/recv(stage 边界) | 流水线接力 |
| **DP** | Data Parallel | 整模型复制 N 份 | 仅 router 层面分流 | 互不通信,各跑各 |
| **CP** | Context Parallel | 单条序列的 token 长度 | ring all-reduce / DistAttention(attention 内) | 长上下文场景,把 N=128k 切到 8 卡每卡 16k |
| **EP** | Expert Parallel | MoE 模型的专家(experts)分散到不同卡 | all-to-all(每层 MoE) | 每卡装一部分专家完整副本,token 路由过去 |

每个进程同时持有多个 rank:`tp_rank` / `pp_rank` / `dp_rank` / `attn_cp_rank` / `moe_ep_rank` / `moe_dp_rank`。

### 2.1 TP(Tensor Parallel)

把每一层的矩阵乘按列或行切到 N 张卡。每层 attn / MLP 后做一次 all-reduce 把切片结果合起来。**通信开销最大**(每层都要),所以同 TP 组的 GPU 一般在同一台机的 NVLink 上。

### 2.2 PP(Pipeline Parallel)

把模型层按段切——比如 60 层模型 PP=3 就是 stage 0 跑 layer 0-19、stage 1 跑 20-39、stage 2 跑 40-59 + LM head。stage 之间通过 send/recv 传 hidden_states。

PP 适合**模型太大单卡装不下**且不能用 TP 装下时使用——但有 pipeline bubble(气泡) 问题,需要 micro-batch 流水才能填满。

### 2.3 DP(Data Parallel)

默认模式下 DP 就是「整模型复制 N 份,各跑各的」,DataParallelController 进程负责把请求分流到不同副本。**DP 副本之间不通信**(没有 NCCL group),只是 router 决定请求去哪个副本。

`enable_dp_attention` 是另一种 DP:DP 不复制整模型,而是和 TP 组共享——只对 attention 部分做 DP(每个 DP 子组独立处理一批请求的 attention),MLP 仍走 TP。这种模式下 attn rank 体系变成「DP × CP × attn_TP」分层。

### 2.4 CP(Context Parallel,上下文并行)

为长上下文设计。attention 是 O(N²),N=128k 时单卡 KV 装不下。CP 把同一条 prompt 的 token 长度切到 N 张卡,每卡装 N/N_cp 个 token 的 KV,通过 ring all-reduce / DistAttention 算法把 attention 拼起来。

SGLang 里有 `attn_cp_size` / `attn_cp_rank`——**只在 attention 部分做 CP**,MLP 走 TP/DP。

### 2.5 EP(Expert Parallel,专家并行)

MoE 模型(DeepSeek-V3、Mixtral、Qwen3-MoE) 每层有几十到几百个专家(experts),每个 token 只激活其中 k 个。

| MoE 切法 | 含义 |
|---|---|
| TP 跑 MoE | 每卡装所有专家的 1/TP 切片(每层每个 expert 都 all-reduce) |
| **EP** 跑 MoE | **每卡装一部分专家的完整副本**(token 通过 all-to-all 路由到对应 expert 所在卡) |

EP 优势:每个 expert 在自己的卡上完整,激活率不均时单卡内部高效;all-to-all 一次完成路由,代替每层 all-reduce。SGLang `ep_size` 控制 EP 维度,有专门的 `eplb` 模块按运行时统计动态调整 expert 分布。

`moe_dp_size` / `moe_ep_size` 是 MoE 体系下另一套 rank 划分(同一进程的 MoE 部分有自己的 rank,和 attn 部分独立)。

---

## 三 进程数公式

标准模式(不启用 `--enable-dp-attention`):

$$
\text{scheduler 进程数} = TP \times DP \times PP
$$

CP 在 SGLang 中是**复用 TP 组**的(`attn_cp_size` 是从 `tp_size` 里再切一刀),不增加进程数:

$$
attn\_tp\_size = \frac{tp\_size}{attn\_dp\_size \times attn\_cp\_size}
$$

EP 同理,是 MoE 部分 rank 的**重新划分**,不增加进程数。

---

## 四 实例:**TP=2, DP=2, PP=3** 的进程拓扑

### 4.1 总进程数

- scheduler 进程 = 2 × 2 × 3 = **12 个**
- HTTP server 主进程(含 TokenizerManager) = 1
- DataParallelController = 1(DP > 1 才有)
- DetokenizerManager = 1

合计 **15 个进程**,占 **12 张 GPU**。

### 4.2 拓扑图

```
┌──────────────────────────────────────────────────────────────────────┐
│  ① HTTP server 进程(主进程)                                         │
│     - FastAPI / uvicorn                                              │
│     - TokenizerManager(就在这个进程里,asyncio loop)                │
│     - 把请求 PUSH 到 DataParallelController 的 ZMQ                    │
└──────────────────────────────────────────────────────────────────────┘
                       │ ZMQ
                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ② DataParallelController 进程(DP > 1 时才有)                       │
│     - 接收请求,按 DP 路由策略选 dp_rank(round-robin / shortest queue) │
│     - 转发给对应 DP 副本的 scheduler                                  │
└──────────────────────────────────────────────────────────────────────┘
                       │ ZMQ(每 DP 副本独立一根)
       ┌───────────────┴───────────────────┐
       ▼                                   ▼
┌─────────────────────┐            ┌─────────────────────┐
│ DP 0 副本           │            │ DP 1 副本           │
│ (一份完整模型 6 卡) │            │ (一份完整模型 6 卡) │
│                     │            │                     │
│ ③ TP 0 + PP 0 GPU 0 │            │ ⑨ TP 0 + PP 0 GPU 6 │
│ ④ TP 1 + PP 0 GPU 1 │            │ ⑩ TP 1 + PP 0 GPU 7 │
│ ⑤ TP 0 + PP 1 GPU 2 │            │ ⑪ TP 0 + PP 1 GPU 8 │
│ ⑥ TP 1 + PP 1 GPU 3 │            │ ⑫ TP 1 + PP 1 GPU 9 │
│ ⑦ TP 0 + PP 2 GPU 4 │            │ ⑬ TP 0 + PP 2 GPU10 │
│ ⑧ TP 1 + PP 2 GPU 5 │            │ ⑭ TP 1 + PP 2 GPU11 │
└─────────────────────┘            └─────────────────────┘
        │ ZMQ(同 DP 副本内的输出 PUSH 出去)
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ⑮ DetokenizerManager 进程                                            │
│     - 收 BatchTokenIDOutput                                          │
│     - token id → text                                                │
│     - PUSH 给 TokenizerManager 的 PULL socket                         │
└──────────────────────────────────────────────────────────────────────┘
                       │ ZMQ
                       ▼
                 回到 ①(HTTP/SSE 流出)
```

### 4.3 GPU 分配规则

代码逻辑(`engine.py:560-565` 与 `data_parallel_controller.py:474-478`):

```python
gpu_id = base_gpu_id
       + (dp_rank * tp_size * pp_size)         # DP 占的偏移
       + ((pp_rank % pp_size_per_node) * tp_size_per_node)
       + (tp_rank % tp_size_per_node) * gpu_id_step
```

单机部署(`nnodes=1`)简化:

```
gpu_id = dp_rank * (tp_size * pp_size) + pp_rank * tp_size + tp_rank
```

| (dp, pp, tp) | gpu_id |
|---|---|
| (0, 0, 0) | 0 |
| (0, 0, 1) | 1 |
| (0, 1, 0) | 2 |
| (0, 1, 1) | 3 |
| (0, 2, 0) | 4 |
| (0, 2, 1) | 5 |
| (1, 0, 0) | 6 |
| (1, 0, 1) | 7 |
| (1, 1, 0) | 8 |
| (1, 1, 1) | 9 |
| (1, 2, 0) | 10 |
| (1, 2, 1) | 11 |

### 4.4 通信组拓扑

每个进程同时属于多个 NCCL 通信组。以 `(DP=0, PP=1, TP=0)` 这个进程为例:

| 组 | size | 成员 | 用途 |
|---|---|---|---|
| **TP group** | 2 | `(DP=0, PP=1, TP=0)` 与 `(DP=0, PP=1, TP=1)` | 每层 attn/MLP 后 all-reduce |
| **PP group** | 3 | `(DP=0, PP=0, TP=0)` `(DP=0, PP=1, TP=0)` `(DP=0, PP=2, TP=0)` | 跨 stage 传 hidden states(send/recv) |
| **DP group** | 不存在 NCCL group(默认 router 模式 DP 完全独立) | — | DP 副本不通信,只是 router 分流 |
| **CP/EP group** | 看是否启用 | — | 启用时才有 |

TP group 是性能关键(每层都通信),所以一般把同一 TP group 的 GPU 排在 PCIe / NVLink 上**邻近**——典型 8 卡机里,GPU 0/1 在同一 NVLink 子组,默认 `gpu_id_step=1` 就利用了这一点。

### 4.5 数据流示例:一个请求的完整路径

```
client POST /generate
   │
   ▼
HTTP server 进程(TokenizerManager)
   │   tokenize prompt
   │   PUSH(ZMQ) → DPController
   ▼
DPController 进程
   │   选 dp_rank(假设选了 DP 0)
   │   PUSH 到 DP 0 副本的 ZMQ
   ▼
进入 DP 0 副本的 6 个 scheduler 进程
   │   recv_requests:仅 PP=0, TP=0 那个 rank 真正从 ZMQ 拉
   │   broadcast_pyobj 把请求广播到同 PP 内的 (PP=0, TP=1)
   │   point_to_point_pyobj 把请求传给 PP=1 的 ranks,再传给 PP=2
   │
   │   组 batch → run_batch
   │   PP 0 rank 跑前 1/3 层
   │     ├─ TP 0 和 TP 1 各跑半边权重,每层后 all-reduce
   │     └─ 跑完 → hidden_states.send() 给 PP 1
   │   PP 1 rank 跑中间 1/3 层 → send 给 PP 2
   │   PP 2 rank 跑最后 1/3 层 + LM head
   │     └─ sample 出 next_token
   │
   │   仅 PP=2 的 ranks 做 process_batch_result
   │   → stream_output
   │   PUSH(ZMQ) → DetokenizerManager
   ▼
DetokenizerManager 进程
   │   detokenize → BatchStrOutput
   │   PUSH(ZMQ) → TokenizerManager
   ▼
HTTP server 进程(TokenizerManager.handle_loop)
   │   _handle_batch_output → state.event.set()
   │   生成器 yield → SSE → client
   ▼
client 收到 token
```

---

## 五 容易混淆的细节

### 5.1 Scheduler 进程内部不止主线程

scheduler 进程主线程跑同步 `while True` 调度循环,但同时还有几个辅助线程:

- **forward_stream**:overlap schedule 用的 CUDA stream(实际在主线程发起)。
- **HiCache 加载线程**:KV 在 GPU/RAM/SSD 间的异步搬运。
- **LoRA overlap loader 线程**:LoRA 权重加载与计算重叠。
- 模型加载阶段还有几个临时线程。

但**调度逻辑本身是单线程同步的**——event_loop 串行处理。

### 5.2 TP rank 0 是「主 rank」

只有 `pp_rank=0 + attn_tp_rank=0 + attn_cp_rank=0` 那一个 scheduler 进程**真的从 ZMQ 拉请求**(`recv_requests`)。其他 rank 通过 NCCL `broadcast_pyobj` 拿到同一份请求集——保证所有 rank 在同一物理时刻看到相同请求,batch 组装才能对齐。

### 5.3 PP=N 时只有最后一个 stage 做采样

PP 流水线下:
- PP=0 跑前段层,只产 hidden_states 传给 PP=1。
- 中间 stage 同上。
- PP=N-1 跑最后一段 + LM head + Sampler,产出 `next_token_ids`,再 ZMQ → DetokenizerManager。

所以 `stream_output` 只在 PP 最后 stage 上发生。

### 5.4 `enable_dp_attention` 是另一种 DP

加 `--enable-dp-attention` 后,DP 不再是「整模型复制」而是和 TP 组共享——只对 attention 部分做 DP,MLP 仍走 TP。进程数变成 `tp_size × pp_size`(DP 共享 TP 组),不是 `tp×dp×pp`。

此时 `compute_dp_attention_world_info` 把全局 TP world 切成「DP × CP × attn_TP」分层:

```
attn_tp_size = tp_size / attn_dp_size / attn_cp_size
```

### 5.5 多机部署

跨节点(`--nnodes=N --node-rank=K`)时,每节点只启动它负责的那部分 rank。比如 `nnodes=2`,12 个 rank 拆成每节点 6 个。HTTP server 和 DPController 只在 `node_rank=0` 启动;其他节点只有 scheduler 进程。

---

## 六 进程创建源码:根据 TP / PP / DP 派生 worker 的位置

### 6.1 顶层入口:`Engine._launch_subprocesses`

`python/sglang/srt/entrypoints/engine.py:633` 的 `_launch_subprocesses` 是整个 server 启动时拉起子进程的总入口。它做三件事:

1. 调 `_launch_scheduler_processes` 拉起 scheduler 进程组(下面详述)。
2. 起 `DetokenizerManager` 进程(`mp.Process(target=run_detokenizer_process_func)`,`engine.py:728`)。
3. 在主进程里构造 `TokenizerManager`(就在主 HTTP server 进程,不开新进程)。

### 6.2 关键分支:`_launch_scheduler_processes`(`engine.py:526-630`)

按 `dp_size` 一分为二:

```python
if server_args.dp_size == 1:
    # ── DP=1 直接 fork 出 PP × TP 个 scheduler 进程
    pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node = (
        _calculate_rank_ranges(
            server_args.nnodes, server_args.pp_size,
            server_args.tp_size, server_args.node_rank,
        )
    )

    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            reader, writer = mp.Pipe(duplex=False)
            gpu_id = (
                server_args.base_gpu_id
                + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            attn_cp_rank, moe_dp_rank, moe_ep_rank = _compute_parallelism_ranks(
                server_args, tp_rank
            )
            with maybe_reindex_device_id(gpu_id) as gpu_id:
                proc = mp.Process(
                    target=run_scheduler_process_func,
                    args=(server_args, port_args, gpu_id,
                          tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                          pp_rank, None, writer),
                )
                proc.start()
            scheduler_procs.append(proc)
else:
    # ── DP>1 只起一个 DataParallelController 进程,
    #    它内部再去 fork 各 DP 副本的 scheduler 组
    proc = mp.Process(
        target=run_data_parallel_controller_process,
        kwargs=dict(
            server_args=server_args, port_args=port_args,
            pipe_writer=writer,
            run_scheduler_process_func=run_scheduler_process_func,
        ),
    )
    proc.start()
```

要点:

- **DP=1 的快路径**:直接 `for pp_rank: for tp_rank:` 二重循环 fork 出 `pp_size × tp_size` 个 scheduler 进程,每个进程的 `target` 是 `run_scheduler_process`(即 scheduler 子进程入口,见 `scheduler.py:3751`)。
- **DP>1 的慢路径**:多一层间接——只起一个 `DataParallelController` 进程,后续 fork 由它负责。
- 每个 scheduler 进程通过 `mp.Pipe(duplex=False)` 拿到一根 writer,启动后会把 init 信息(`max_total_num_tokens`、`status="ready"` 等)从 writer 写回主进程,主进程通过 reader 等待所有 ready。

### 6.3 GPU 编号与 rank 范围:`_calculate_rank_ranges`(`engine.py:1245`)

```python
def _calculate_rank_ranges(nnodes, pp_size, tp_size, node_rank):
    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_pp_rank = max(nnodes // pp_size, 1)
    pp_rank_range = range(
        pp_size_per_node * (node_rank // nnodes_per_pp_rank),
        pp_size_per_node * (node_rank // nnodes_per_pp_rank + 1),
    )

    nnodes_per_tp_group = nnodes_per_pp_rank
    tp_size_per_node = tp_size // nnodes_per_tp_group
    tp_rank_range = range(
        tp_size_per_node * (node_rank % nnodes_per_tp_group),
        tp_size_per_node * (node_rank % nnodes_per_tp_group + 1),
    )

    return pp_rank_range, tp_rank_range, pp_size_per_node, tp_size_per_node
```

它告诉「**当前节点应该启动哪些 (pp_rank, tp_rank) 组合**」——单节点(`nnodes=1`) 时退化成完整的两个 range,多节点时按 `node_rank` 切分。`gpu_id` 公式:

```
gpu_id = base_gpu_id + (pp_rank % pp_size_per_node) * tp_size_per_node
                     + (tp_rank % tp_size_per_node) * gpu_id_step
```

`gpu_id_step` 默认 1,允许跳着用 GPU(比如同机有别的进程占了某些卡)。

### 6.4 派生 attn_cp / moe_dp / moe_ep rank:`_compute_parallelism_ranks`(`engine.py:1280`)

```python
def _compute_parallelism_ranks(server_args, tp_rank):
    attn_dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
    # Attention 层级:Global(TP) -> DP -> ATTN_CP -> ATTN_TP
    attn_tp_size = server_args.tp_size // attn_dp_size // server_args.attn_cp_size
    attn_cp_rank = (tp_rank // attn_tp_size) % server_args.attn_cp_size
    # MoE 层级:Global(TP) -> MOE_DP -> EP -> MOE_TP
    moe_dp_rank = tp_rank // (server_args.tp_size // server_args.moe_dp_size)
    moe_ep_rank = (
        tp_rank
        % (server_args.tp_size // server_args.moe_dp_size)
        // (server_args.tp_size // server_args.moe_dp_size // server_args.ep_size)
    )
    return attn_cp_rank, moe_dp_rank, moe_ep_rank
```

这就是「**CP/EP 不增加进程数,只是给已有 tp_rank 重排**」的具体实现——给定一个 tp_rank,推出它在 attn-CP / MoE-DP / MoE-EP 三个维度的局部 rank,作为参数传给 scheduler 进程。

### 6.5 DP > 1 的两层 fork:`DataParallelController`

`python/sglang/srt/managers/data_parallel_controller.py`:

```python
def launch_dp_schedulers(self, server_args, port_args):              # L238
    base_gpu_id = 0
    threads = []
    for dp_rank in range(server_args.dp_size):
        ...
        thread = threading.Thread(
            target=self.launch_tensor_parallel_group_thread,
            args=(server_args, tmp_port_args, base_gpu_id, dp_rank, ready_event),
        )
        threads.append(thread)
        base_gpu_id += (
            server_args.tp_size * server_args.pp_size * server_args.gpu_id_step
        )
        ...
    for thread in threads:
        thread.start()

def launch_tensor_parallel_group(self, server_args, port_args,        # L419
                                 base_gpu_id, dp_rank, worker_ports=None):
    pp_rank_range = ...
    tp_rank_range = ...
    for pp_rank in pp_rank_range:
        for tp_rank in tp_rank_range:
            ...
            gpu_id = (
                server_args.base_gpu_id + base_gpu_id
                + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
            )
            ...
            proc = mp.Process(
                target=self.run_scheduler_process_func,
                args=(server_args, rank_port_args, gpu_id,
                      tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                      pp_rank, dp_rank, writer),
            )
            proc.start()
```

DP>1 时是**两层循环**:

1. 外层 `for dp_rank in range(dp_size)`:每个 DP 副本起一个 Python 线程(注意是线程不是进程),每起一个就把 `base_gpu_id += tp_size * pp_size * gpu_id_step`,保证不同 DP 副本占不同 GPU。
2. 内层(`launch_tensor_parallel_group`)再走 `for pp_rank: for tp_rank:`,fork 出 `pp_size × tp_size` 个 scheduler 进程。

最终总进程数 = **dp_size × pp_size × tp_size**——和外层公式一致。

### 6.6 子进程入口:`run_scheduler_process`(`scheduler.py:3751`)

```python
def run_scheduler_process(
    server_args: ServerArgs,
    port_args: PortArgs,
    gpu_id: int,
    tp_rank: int,
    attn_cp_rank: int,
    moe_dp_rank: int,
    moe_ep_rank: int,
    pp_rank: int,
    dp_rank: Optional[int],
    pipe_writer,
):
    ...
    scheduler = Scheduler(server_args, port_args, gpu_id,
                          tp_rank, attn_cp_rank, moe_dp_rank, moe_ep_rank,
                          pp_rank, dp_rank)
    pipe_writer.send({"status": "ready", ...})
    scheduler.run_event_loop()
```

进入子进程后,所有 rank 信息通过参数传入。`Scheduler.__init__` 内部根据这些 rank 构造 NCCL 通信组(TP / PP / DP-attn / CP / EP)、加载模型对应分片、创建 `TpModelWorker`(`scheduler.py:634`)——这就是「scheduler = worker」融合的具体落点。

### 6.7 一图总结代码路径

```
Engine._launch_subprocesses               (engine.py:633)
  └─ _launch_scheduler_processes          (engine.py:526)
       │
       ├─ if dp_size == 1:                          ┌─ for tp_rank:
       │     _calculate_rank_ranges                 │     mp.Process(run_scheduler_process)
       │     _compute_parallelism_ranks  ──→  for pp_rank:
       │                                            └─    ...
       │
       └─ else:
            mp.Process(run_data_parallel_controller_process)
              └─ DataParallelController.launch_dp_schedulers
                   └─ for dp_rank:                  ┌─ for tp_rank:
                        launch_tensor_parallel_group│     mp.Process(run_scheduler_process)
                                                    └─ for pp_rank:
                                                          ...

run_scheduler_process(scheduler.py:3751)
  └─ Scheduler(...).run_event_loop()
       └─ self.tp_worker = TpModelWorker(...)        ← scheduler 内嵌 worker
```

---

## 七 一句话总结

> SGLang 的 **scheduler 进程 = worker 进程 = GPU rank 进程**(同一个 OS 进程,不是分离的)——每进程绑一张 GPU,内部 scheduler 做 batch 组装,worker 直接调 `model.forward` 跑 CUDA kernel。
>
> 标准并行模式下,scheduler 进程数 = **TP × PP × DP**;CP 和 EP 是在已有进程上**重新划分 rank**(CP 复用 TP 组切 attention 序列,EP 重排 MoE 的 expert 分布),不增加进程数。
>
> 例:**TP=2, DP=2, PP=3** → 12 个 scheduler 进程 + DPController + HTTP server + DetokenizerManager = 15 进程,占 12 张 GPU。
