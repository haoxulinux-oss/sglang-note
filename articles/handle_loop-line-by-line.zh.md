# `handle_loop()` 每行代码分析

**位置：** `python/sglang/srt/managers/tokenizer_manager.py:1613-1626`

```python
async def handle_loop(self):
    """The event loop that handles requests"""
    while True:
        with self.soft_watchdog.disable():
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
        if isinstance(
            recv_obj,
            (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
        ):
            await self._handle_batch_output(recv_obj)
        else:
            self._result_dispatcher(recv_obj)
        self.last_receive_tstamp = real_time()
        self.soft_watchdog.feed()
```

整体角色：**TokenizerManager 进程里唯一的后台消费者协程**，从 ZMQ 入站 socket 拉取下游(DetokenizerManager / Scheduler)发回来的所有消息，分发到对应的请求状态或控制路径上。

---

## L1613：`async def handle_loop(self)`

- `async def` ⇒ 这是个 coroutine function。它的存在前提是后面有人 `await` 或 `loop.create_task(handle_loop())` 把它调度起来。
- 不接受任何参数(除 `self`)：所有状态都挂在 `self` 上(socket、watchdog、`rid_to_state`)。
- 返回类型隐式 `None` ——但这个函数**永不返回**(它是个 `while True`)，返回类型不重要。
- 所谓「event loop」是指**这个函数内部自己组织的处理循环**，不是 asyncio 的 event loop。注意命名歧义：asyncio 的 event loop 是调度者，`handle_loop` 是被调度的协程之一。

---

## L1614：docstring `"""The event loop that handles requests"""`

纯文档，无运行时行为。提示读者：这是一个处理请求结果的事件循环。

---

## L1615：`while True:`

无限循环。这条协程会一直跑到进程退出 / Task 被 cancel。

实践中它退出的合法路径只有两条：

1. 进程关闭时 `task.cancel()` ⇒ `await ...recv_pyobj()` 处抛 `CancelledError` ⇒ 协程结束。
2. 内部某行未捕获异常 ⇒ 由外层 `print_exception_wrapper` 打日志并(通常)让进程退出。

---

## L1616：`with self.soft_watchdog.disable():`

`soft_watchdog` 是 SGLang 的看门狗：定期检查 `last_receive_tstamp` 是否在合理范围内，**很久没收到任何 scheduler/detokenizer 包就触发异常**(怀疑下游 hang 了)。

`disable()` 是个上下文管理器，在 `with` 体内**暂时停掉看门狗计时**。原因：

- `await recv_pyobj()` 是「故意阻塞」——当下没请求时自然不会有消息回来，可以等好久。
- 如果不暂停，看门狗会把这段空闲误判为 hang。

进入 `with` 时 watchdog 被「静音」，退出 `with` 时自动恢复。注意 **它只包了 `recv_pyobj` 这一行**，下面 dispatch 的部分仍然受监控(这部分不该久等)。

---

## L1617：`recv_obj = await self.recv_from_detokenizer.recv_pyobj()`

这是这个函数最核心的一句。

- `self.recv_from_detokenizer` 是 ZMQ 的 PULL socket(`tokenizer_manager.py:344` 构造)，对端是 DetokenizerManager 的 PUSH(流式 token 输出)以及若干控制信号 PUSH。
- `recv_pyobj()` 是 pyzmq 的方法：阻塞拉一条消息，自动 `pickle.loads` 成 Python 对象。
- `await` 前缀：SGLang 用的是 pyzmq 的 asyncio 适配(`zmq.asyncio.Socket`)，所以这是非阻塞 await ——等消息到达期间 event loop 可以调度其他协程(`generate_request`、HTTP handler、cron 任务 …)。
- 结果 `recv_obj` 是反序列化后的 Python 对象，可能类型很多：
  - 流式输出：`BatchStrOutput` / `BatchEmbeddingOutput` / `BatchTokenIDOutput`
  - 控制 / 管理回包：权重更新结果、profile ack、health 信号、abort ack、open/close session ack 等等。
- 这一行 **不会反压 scheduler**：ZMQ 自有 buffer。如果消费太慢，buffer 涨，scheduler 那边 `send_pyobj` 才开始阻塞。

---

## L1618-L1621：类型分支判断

```python
if isinstance(
    recv_obj,
    (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
):
```

判断收到的是不是「批量推理输出」三种类型之一：

- `BatchStrOutput`：detokenizer 已经把 token id 转成文本。**这是 `/generate` 走 tokenizer 时的常规包**。
- `BatchEmbeddingOutput`：embedding 任务(`/encode`)的回包，没有 detokenize 这一步。
- `BatchTokenIDOutput`：`skip_tokenizer_init=True` 时直接给回 token id(没文本)。

这三类有共同形状：含 `rids: List[str]`、按下标对齐的 `output_strs / output_ids / finished_reasons / meta_info` 等。所以可以走统一处理路径。

---

## L1622：`await self._handle_batch_output(recv_obj)`

把这批输出按 rid 分发回 `rid_to_state[rid].out_list`，并 `state.event.set()` 唤醒那些挂在 `_wait_one_response()` 里的请求协程。

`_handle_batch_output` 内部会：

1. 遍历 `recv_obj.rids`，找到对应 `ReqState`；找不到就 log 并跳过(已 abort 的请求)。
2. 拼 `meta_info`、output text、finish reason、token usage、logprobs 等成 out dict。
3. `state.out_list.append(out)`，并(按 batching 策略)`state.event.set()` 通知对应请求。
4. 对 `finished` 请求做计时、metrics、`del self.rid_to_state[rid]`。

之所以是 `await`：函数内部会 `await self.lora_registry.release(...)` 之类异步操作(LoRA 引用计数)。

**这也是为什么必须暂停 watchdog 的精确设计**：watchdog 只测 `recv_pyobj` 之外的时间，这部分必须够快——分发大批 rid + LoRA release 是 CPU 受限的，几毫秒级。

---

## L1623-L1624：非批量输出走独立 dispatcher

```python
else:
    self._result_dispatcher(recv_obj)
```

`recv_obj` 不是三类批量输出之一时，是各种**控制平面消息**：

- `UpdateWeightFromDiskReqOutput` / `UpdateWeightsFromDistributedReqOutput` …
- `OpenSessionReqOutput` / `CloseSessionReqOutput`
- `AbortReq` 的 ack
- `ProfileReq` ack
- `GetWeightsByNameReqOutput`
- 若干 health / state 广播

`_result_dispatcher` 是一个基于类型的分派器(在 `__init__` 用 dict-of-callbacks 注册)。它**不是 async**，是纯同步 dispatch：根据 `type(recv_obj)` 找到对应的 handler(通常去 `set_result()` 某个 future 或 `set()` 某个 event)，同步调一下。

注意对比：

- L1622 `await ...` —— 数据面，可能耗时。
- L1624 同步调用 —— 控制面，量小、纯 set 标志位。

---

## L1625：`self.last_receive_tstamp = real_time()`

更新「最后一次收到下游消息」的时间戳。

`real_time()` 是 monotonic 风格的时间源(一般是 `time.time()` 包装)。这个值被多处使用：

- watchdog 判断是否 hang。
- `/get_load`、`/server_info` 等 endpoint 报告活跃度。
- 一些超时兜底逻辑。

注意它在 dispatch **之后**更新——意味着「我已经处理完了一条消息」。这比放在 recv 之后更新更贴近「健康」的定义(不仅收到了，还消费完了)。

---

## L1626：`self.soft_watchdog.feed()`

「喂狗」：告诉看门狗「我活着、我消费了一条消息」。watchdog 会把计时器重置。

和 L1616 的 `disable()` 配合：

- 进入 recv 前 disable(recv 期间不计)。
- 收到 + dispatch 完一条消息后 feed(重置计时器)。

如果 dispatch 自己卡住(例如 `_handle_batch_output` 死循环)就会在没 feed 的情况下被看门狗发现。

---

## 一轮循环的整体效果

```
┌─ while True ─────────────────────────────────────────────┐
│  ① 暂停 watchdog,await ZMQ recv 拿到 1 条消息           │
│  ② 是「批量推理输出」?                                   │
│       └─ 是 → 异步分发到各 rid 的 ReqState,唤醒协程     │
│       └─ 否 → 同步走控制面 dispatcher                   │
│  ③ 更新 last_receive_tstamp                             │
│  ④ 喂 watchdog                                          │
└──────────────────────────────────────────────────────────┘
回到 ①
```

---

## 几个容易误解的点

1. **「event loop」名称歧义**：函数叫 handle_loop，内部 while True，但它**不是** asyncio 的 event loop，而是 event loop 上的一个 task。整个进程依然只有一条 asyncio loop。

2. **单线程 vs 多请求**：一轮循环只处理 1 条 ZMQ 消息，但**1 条消息可能包含多个 rid 的输出**(scheduler 是按 batch 推回来的)。所以这个「串行 loop」不会成为并发瓶颈——它做的是「拉一个批 → fan-out 到 N 个请求」。

3. **为什么 dispatch 不全做成 async**：批量数据面的清理 / metrics 写入有 IO(虽然轻量)，故 `_handle_batch_output` 是 async；控制面只是 set 标志位，没必要走协程开销，故 `_result_dispatcher` 是同步函数。

4. **异常路径**：任何一行未捕获异常会 bubble out，被外层 `print_exception_wrapper` 接住打日志。如果 `recv_pyobj` 抛 `CancelledError`(进程关闭)，协程干净退出。

5. **背压**：handle_loop 阻塞或变慢 ⇒ ZMQ buffer 涨 ⇒ DetokenizerManager 的 `send_pyobj` 阻塞 ⇒ 进一步上游 scheduler 也会阻塞。这是个天然反压链，但 buffer 容量很大(默认 ZMQ HWM 1000+)，实践中几乎不会反压。
