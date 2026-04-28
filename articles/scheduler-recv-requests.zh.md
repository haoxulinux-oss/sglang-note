# `Scheduler.recv_requests()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:1504-1659`

**角色：** 主循环每轮的第一步——从 ZMQ socket 拉取 TokenizerManager 推过来的请求(以及 RPC 控制消息)，然后跨 TP/CP/PP rank 广播，让所有 rank 看到一致的请求集。

---

## 一 函数签名与整体结构

```python
def recv_requests(
    self,
) -> List[Union[TokenizedGenerateReqInput, TokenizedEmbeddingReqInput, Any]]:
    """Receive results at tp_rank = 0 and broadcast it to all other TP ranks."""
```

整体分四段：

1. **recv_skipper 早退**(可选)：根据上一个 forward_mode 决定是否本轮跳过 recv。
2. **主 rank 从 ZMQ 拉**：`pp_rank=0 + attn_tp_rank=0 + attn_cp_rank=0` 的那一个 rank 才真正调 ZMQ。
3. **跨 rank 广播**：通过 `broadcast_pyobj` / `point_to_point_pyobj` 让其他 rank 拿到同一份。
4. **shm 展开**：把 `ShmPointerMMData` 元信息换回真正的 tensor。

---

## 二 recv_skipper 早退

```python
if self.recv_skipper is not None:
    last_forward_mode = (
        self.last_batch.forward_mode if self.last_batch is not None else None
    )
    if not self.recv_skipper.handle(last_forward_mode):
        return []
```

`recv_skipper` 是个可选机制(`scheduler_recv_skipper.py`)，用于某些场景下让连续若干个 decode step 不去 recv 新请求(避免 prefill 频繁中断 decode 流水)。常规配置下 `recv_skipper is None`，这段直接跳过。

---

## 三 主 rank 从 ZMQ 拉(NOBLOCK 轮询)

```python
if self.pp_rank == 0:
    if self.attn_tp_rank == 0 and self.attn_cp_rank == 0:
        recv_reqs = []
        while True:
            try:
                if self.recv_limit_reached(len(recv_reqs)):
                    break
                recv_req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_req)

        while True:
            try:
                if self.recv_limit_reached(len(recv_reqs)):
                    break
                recv_rpc = self.recv_from_rpc.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            recv_reqs.append(recv_rpc)
    else:
        recv_reqs = None
```

**关键点：**

- **`zmq.NOBLOCK`**：一旦 socket 里没有消息，`recv_pyobj` 立即抛 `zmq.ZMQError`(`Resource temporarily unavailable`/EAGAIN)，被 except 接住后 `break` 跳出循环。这与 TokenizerManager 的 `await recv_pyobj()` 形成鲜明对比——那边是 IO 协程可以挂起，这边是同步主循环不能阻塞。
- **「拉满即止」语义**：内层 `while True` 把 socket 里现存的所有消息一次拉空(或拉到 `max_recv_per_poll` 上限)，然后再走主循环。这样能在吞吐高峰一次摊大很多请求，让 batch 组得更满；低峰也只是空转一圈。
- **两段：tokenizer + rpc**：
  - `recv_from_tokenizer`：日常请求(`TokenizedGenerateReqInput` 等数据面 + 部分管理面)。
  - `recv_from_rpc`：单独的 RPC socket，用于权重更新、profile 等更重的控制消息。它们走独立 socket 是为了避免大数据请求把控制消息卡在队尾。
- **`recv_limit_reached`**：单轮上限由 `--max-recv-per-poll` 控制(默认 `-1` 表示无限制)。设这个上限的目的：避免一次拉太多让本轮 forward 推迟太久，影响 ITL。
- 仅主 rank 调，其他 rank 直接 `recv_reqs = None`——它们不该和 ZMQ 直连，避免重复消费。

PP rank > 0 走另一支(`scheduler.py:1539-1550`)：通过 `point_to_point_pyobj` 从前一个 PP rank 拿请求(PP 里请求是「沿管线一节节传」)。

---

## 四 跨 rank 广播请求

之后所有 rank 必须看到同一份 `recv_reqs`，否则 batch 组装会分裂、立刻死锁。分两种模式：

### 4.1 普通 TP(`enable_dp_attention=False`)

```python
elif self.tp_size != 1:
    recv_reqs = broadcast_pyobj(
        recv_reqs,
        self.tp_group.rank,
        self.tp_cpu_group,
        src=self.tp_group.ranks[0],
    )
```

源 rank 把 list pickle 出去，其他 rank pickle 回来——经过 gloo CPU group(NCCL 不擅长不规则 Python 对象的传输)。

### 4.2 DP attention 模式

```python
work_reqs, control_reqs = self._split_work_and_control_reqs(recv_reqs)

if self.attn_tp_size != 1:
    work_reqs = broadcast_pyobj(work_reqs, self.attn_tp_group.rank,
                                self.attn_tp_cpu_group, src=...)
if self.attn_cp_size != 1:
    work_reqs = broadcast_pyobj(work_reqs, self.attn_cp_group.rank,
                                self.attn_cp_cpu_group, src=...)
```

DP attention 下不同 DP 子组接收的请求**不一样**(每个 DP 组独立分一片请求)，所以 `work_reqs`(实际推理请求) 只在 `attn_tp_group + attn_cp_group` 内广播；而 `control_reqs`(权重更新等) 必须全 rank 同步——通过额外一次 `tp_cpu_group` 广播或 `dp_attention_local_control_broadcast` 优化。

`_split_work_and_control_reqs`(`scheduler.py:1661-1688`) 按类型把请求拆成两组：
- work：`TokenizedGenerateReqInput` / `TokenizedEmbeddingReqInput` / 它们的 batch 版。
- control：其他所有(权重更新、profile、abort、open/close session …)。

### 4.3 EPD 解耦的多模态接收

```python
if (
    self.pp_rank == 0
    and self.server_args.language_only
    and self.server_args.encoder_transfer_backend == "zmq_to_scheduler"
):
    recv_reqs, abort_reqs = self.mm_receiver.process_waiting_requests(recv_reqs)
    for req, error_msg, error_code in abort_reqs:
        ...
        prepare_abort(req, error_msg, status_code=status_code)
        self.stream_output([req], req.return_logprob)
```

EPD(Encoder-Prefill-Decode 解耦) 模式下，多模态请求需要先等 vision encoder 节点把 embedding 通过独立 ZMQ 推过来——`mm_receiver` 把还没拿到 embedding 的请求暂存住，下一轮再放回。

---

## 五 共享内存特征展开

```python
if recv_reqs:
    if (
        not self.server_args.enable_dp_attention
        and self.tp_size > 1
        and self.model_config.is_multimodal
        and has_shm_features(recv_reqs)
    ):
        barrier(group=self.tp_cpu_group)
    for req in recv_reqs:
        unwrap_shm_features(req)
```

多模态张量(图片 / 视频帧解码后的 embedding) 体积大，`broadcast_pyobj` 走 pickle 太慢——所以源 rank 把它们写到 POSIX shm，只把 `ShmPointerMMData` 元信息(name + dtype + shape) 塞进 `recv_reqs`。`unwrap_shm_features(req)` 在每个 rank 上独立 `shm_open` 把 tensor 拿回来。

为什么需要 `barrier`：源 rank 在 `broadcast_pyobj` 返回时立刻继续往下走，可能在其他 rank 还没 `shm_open` 之前就 `shm_unlink`。barrier 强迫所有 rank 同步到这一行，再统一展开——保证 `shm_open` 都成功后才有 `shm_unlink`。

DP attention 模式不需要 barrier：因为它的 `control_reqs` 广播本身就是一次 collective，已经强制了同步点(代码注释里有详细说明)。

---

## 六 返回值与下一步

返回的 `recv_reqs` 是 `List[各种 *ReqInput]`。回到主循环后立即被 `process_input_requests(recv_reqs)` 处理。

---

## 七 设计要点小结

| 决策 | 原因 |
|---|---|
| `zmq.NOBLOCK` 轮询 | scheduler 主循环是同步 `while True`，不能阻塞 |
| 内层 while 拉满 | 一次摊大，batch 组得更满；空闲时也只是单轮空转 |
| tokenizer + rpc 双 socket | 避免大数据流量把控制消息卡住 |
| 仅主 rank 收 + broadcast | 保证所有 rank 在同一物理时刻看到一致请求集 |
| DP attention 拆 work/control | 工作请求按 DP 子组广播，控制消息必须全 rank 同步 |
| shm 元信息 + unwrap | 多模态张量太大，用共享内存避免 pickle 开销 |
| 多模态 + TP > 1 加 barrier | 防止源 rank 在其他 rank `shm_open` 完成前 `shm_unlink` |
