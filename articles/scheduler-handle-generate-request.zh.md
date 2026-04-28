# `Scheduler.handle_generate_request()` 解析

**位置：** `python/sglang/srt/managers/scheduler.py:1827-2023`

**角色：** `_request_dispatcher` 收到 `TokenizedGenerateReqInput` 时调用——把 IPC 数据结构转成 scheduler 内部的 `Req` 对象，处理多模态 / session / disagg / grammar 等情况，最后入队等调度。这是请求在 scheduler 进程「正式落户」的地方。

---

## 一 函数签名

```python
def handle_generate_request(
    self,
    recv_req: TokenizedGenerateReqInput,
):
```

输入：`TokenizedGenerateReqInput`——TokenizerManager 把用户请求 tokenize 后的产物，含 `rid / input_text / input_ids / sampling_params / stream / mm_inputs / session_params / bootstrap_room ...`。

输出：无(返回值 `None`)。所有副作用通过 `_add_request_to_queue` 或在错误时 `stream_output` 完成。

---

## 二 三条路由

第一段先按 `session_id` 分三种情况：

```python
session_id = (
    recv_req.session_params.id if recv_req.session_params is not None else None
)

if session_id is None:
    # ① 普通无会话请求 —— 构造全新 Req
    ...
elif session_id in self.session_controller and not ...close_on_finish:
    # ② 有效会话 —— 从 session 派生 Req(自动接上历史 prefix)
    ...
else:
    # ③ 会话不存在或正在关闭 —— 构造 Req 并立即 abort
    ...
```

### 2.1 普通无会话请求(最常见)

```python
if recv_req.input_embeds is not None:
    # input_embeds 模式:用 fake input_ids 占位(长度对齐)
    seq_length = len(recv_req.input_embeds)
    fake_input_ids = [1] * seq_length
    recv_req.input_ids = fake_input_ids

if recv_req.bootstrap_port is None:
    recv_req.bootstrap_port = self.server_args.disaggregation_bootstrap_port

req = Req(
    recv_req.rid,
    recv_req.input_text,
    recv_req.input_ids,
    recv_req.sampling_params,
    return_logprob=recv_req.return_logprob,
    top_logprobs_num=recv_req.top_logprobs_num,
    token_ids_logprob=recv_req.token_ids_logprob,
    stream=recv_req.stream,
    lora_id=recv_req.lora_id,
    input_embeds=recv_req.input_embeds,
    positional_embed_overrides=recv_req.positional_embed_overrides,
    token_type_ids=recv_req.token_type_ids,
    custom_logit_processor=recv_req.custom_logit_processor,
    require_reasoning=recv_req.require_reasoning,
    return_hidden_states=recv_req.return_hidden_states,
    return_routed_experts=recv_req.return_routed_experts,
    eos_token_ids=self.model_config.hf_eos_token_id,
    bootstrap_host=recv_req.bootstrap_host,
    bootstrap_port=recv_req.bootstrap_port,
    bootstrap_room=recv_req.bootstrap_room,
    disagg_mode=self.disaggregation_mode,
    routed_dp_rank=recv_req.routed_dp_rank,
    disagg_prefill_dp_rank=recv_req.disagg_prefill_dp_rank,
    vocab_size=self.model_config.vocab_size,
    priority=recv_req.priority,
    metrics_collector=(self.metrics_collector if self.enable_metrics else None),
    routing_key=recv_req.routing_key,
    extra_key=recv_req.extra_key,
    http_worker_ipc=recv_req.http_worker_ipc,
    dllm_config=self.dllm_config,
    time_stats=recv_req.time_stats,
    multi_item_delimiter_indices=recv_req.multi_item_delimiter_indices,
)
req.tokenizer = self.tokenizer
```

`Req` 是 scheduler 进程里**承载请求生命周期的核心对象**——从此刻入队，到完成时释放 KV、log time stats，整个过程都挂在它上面。它和 `TokenizedGenerateReqInput` 的区别：

| `TokenizedGenerateReqInput` | `Req` |
|---|---|
| IPC 数据结构(dataclass + pickle 友好) | 内部状态机 |
| 跨进程传输用 | 仅 scheduler 进程持有 |
| 不可变 | 字段被持续更新(`output_ids`, `prefix_indices`, `kv_indices`, `time_stats`, ...) |

`req.tokenizer = self.tokenizer` 把 detokenize 用的 tokenizer 引用挂上去——给 stop string 检测、incremental decode 用。

#### Disaggregated 模式校验

```python
if self.disaggregation_mode != DisaggregationMode.NULL:
    if (
        recv_req.bootstrap_room is None
        and self.transfer_backend != TransferBackend.FAKE
    ):
        # PD 解耦下必须有 bootstrap_room id 用于 KV 配对
        prepare_abort(req, error_msg, status_code=HTTPStatus.BAD_REQUEST)
        self.stream_output([req], req.return_logprob)
        return
```

PD 解耦下 prefill 和 decode 在不同节点，靠 `bootstrap_room` 把同一个请求的 KV 和元信息配对。没传就直接 abort。

### 2.2 有效会话请求

```python
elif session_id in self.session_controller and not ...close_on_finish:
    session = self.session_controller.get(session_id)
    req = session.create_req(
        recv_req,
        self.tokenizer,
        self.model_config.vocab_size,
        eos_token_ids=self.model_config.hf_eos_token_id,
    )
    if self.enable_metrics:
        req.time_stats.set_metrics_collector(self.metrics_collector)
    if isinstance(req.finished_reason, FINISH_ABORT):
        self.init_req_max_new_tokens(req)
        self._add_request_to_queue(req)
        return
```

`session.create_req` 内部把上一轮保留的 prefix 拼到这一轮的 `origin_input_ids` 前面——这样 RadixCache 自动命中之前的 KV，多轮对话不需要重复 prefill。如果 session 状态已经异常(比如之前请求被 abort)，`finished_reason` 会被设置成 `FINISH_ABORT`，这里只是把它入队让 stream_output 把 abort 信息推回。

#### 2.2.1 多轮对话:client 维护 vs server session(两条独立路径)

之前一篇文章提到「调 `/generate` 时多轮对话和 chat template 都是 client 责任」,而这里又出现了 `session.create_req` 自动拼 prefix——看起来矛盾,**实际上 SGLang 提供两条互斥的可选路径**,client 选哪一条决定走哪个分支。

| 维度 | client 维护(默认) | server session |
|---|---|---|
| 触发条件 | `/generate` 不带 `session_params`、`/v1/chat/completions` | client 主动调 `/open_session` 拿 session_id,后续请求带 `session_params={"id": ...}` |
| Server 状态 | 无 | 有(`SessionController` 持有 prompt history) |
| Client 每轮发什么 | 整段 history(渲染好的 prompt) | 只发当轮新输入 |
| KV 重用机制 | RadixCache 基于 prefix 自动命中 | session prefix 直接拼接 + RadixCache 锦上添花 |
| 网络开销 | 高(每轮发越来越长的 prompt) | 低(只发新 token) |
| Client 复杂度 | 自己拼 history + 渲染模板 | 只管 session_id |
| Server 复杂度 | 简单(无状态) | 复杂(要管 session 生命周期、过期回收) |
| 主流用法 | **OpenAI 兼容、绝大多数生产场景** | 高频长对话、HTTP 带宽敏感场景 |
| Multi-server 兼容 | 完美兼容(任意 server 都能接) | 需要 sticky routing(同 session 必须落到同一 server) |

两条路径在源码里就是 `if session_id is None: ... elif session_id in self.session_controller: ...` 的互斥分支——**不会同时发生**。所以「client 维护」是默认场景下的描述,「session.create_req 拼 prefix」是显式开 session 时的可选优化,两者并不矛盾。

补充:即使是「client 维护」路径,server 端也不会重复 prefill 上一轮内容——因为下一轮 prompt 的前缀和上一轮完全相同,**RadixCache 会自动命中**,只 prefill 新增的 user message + assistant 开口。所以默认路径并不浪费计算,只是浪费一点 HTTP 带宽。

### 2.3 会话不存在或正在关闭

```python
else:
    if session_id in self.session_controller:
        error_msg = f"Invalid request: close was requested for session {session_id}"
    else:
        error_msg = f"Invalid request: session id {session_id} does not exist"
    req = Req(
        recv_req.rid, recv_req.input_text, recv_req.input_ids,
        recv_req.sampling_params,
        vocab_size=self.model_config.vocab_size,
        http_worker_ipc=recv_req.http_worker_ipc,
    )
    req.tokenizer = self.tokenizer
    req.set_finish_with_abort(error_msg)
    self.init_req_max_new_tokens(req)
    self._add_request_to_queue(req)
    return
```

构造一个最小化的 `Req` 仅用于把错误消息回送给 client。

---

## 三 dflash 校验(specdec 变体)

```python
if self.spec_algorithm.is_dflash():
    error_msg = validate_dflash_request(req)
    if error_msg is not None:
        req.set_finish_with_abort(error_msg)
        self.init_req_max_new_tokens(req)
        self._add_request_to_queue(req)
        return
```

dflash 推测解码对请求有额外约束(prompt 长度等)，提前拒绝。

---

## 四 多模态展开

```python
if recv_req.mm_inputs is not None:
    image_inputs = self._get_multimodal_inputs(recv_req.mm_inputs)
    SessionController.adjust_mm_offsets(recv_req, req, image_inputs)

    if self.pad_input_ids_func:
        req.origin_input_ids = self.pad_input_ids_func(
            req.origin_input_ids, image_inputs
        )
    req.extend_image_inputs(image_inputs)
    self._maybe_compute_mrope_positions(req)

    if len(req.origin_input_ids) >= self.max_req_input_len:
        req.set_finish_with_abort(error_msg=...)
        self.init_req_max_new_tokens(req)
        self._add_request_to_queue(req)
        return
```

四步：

1. **`_get_multimodal_inputs`**：把 IPC 里的 `MultimodalDataItem` 集合转成 `MultimodalInputs`(包括跨 rank 广播)。
2. **`adjust_mm_offsets`**：在 session 路径下，prefix tokens 数量影响 mm 在序列中的位置——这里把 `mm_offset` 矫正一遍。
3. **`pad_input_ids_func`**：把单个 image placeholder token 展开成 N 个占位 token——为 vision encoder 输出的 patch embedding 留出位置。`N` 取决于模型(Qwen2-VL: 按图片像素，LLaVA: 固定 576)。函数由 model 自己提供(在 `srt/models/<family>.py` 里定义)。
4. **`_maybe_compute_mrope_positions`**：MRoPE(multi-modal RoPE) 模型(Qwen2-VL 等) 需要每个 token 三维位置索引(text id, h id, w id)，在这里预计算。

如果展开后超长就 abort。

---

## 五 max_new_tokens 与 prompt 长度校验

```python
self.init_req_max_new_tokens(req)

error_msg = validate_input_length(
    req,
    self.max_req_input_len,
    self.server_args.allow_auto_truncate,
)
if error_msg:
    req.set_finish_with_abort(error_msg)
    self._add_request_to_queue(req)
    return
```

`init_req_max_new_tokens`(`scheduler.py:1715`)：

```python
req.sampling_params.max_new_tokens = min(
    (req.sampling_params.max_new_tokens
     if req.sampling_params.max_new_tokens is not None else 1 << 30),
    self.max_req_len - len(req.origin_input_ids) - 1,
)
```

把 user 给的 max_new_tokens 和「context window 剩余空间」取小值——保证 prefill + decode 总长不会超 `max_req_len`。`-1` 是给最后一个 stop/eos token 留位。

`validate_input_length`：检查 prompt 是否超过 `max_req_input_len`(server 配置)，超长且未启用 `allow_auto_truncate` 就 abort。

---

## 六 logprob 起点校准

```python
if not recv_req.return_logprob and recv_req.logprob_start_len != -1:
    recv_req.logprob_start_len = -1

if recv_req.logprob_start_len == -1:
    if recv_req.return_logprob and recv_req.token_ids_logprob is None:
        # 默认:只返回 output token 的 logprob
        req.logprob_start_len = len(req.origin_input_ids)
    elif req.is_prefill_only:
        req.logprob_start_len = len(req.origin_input_ids)
    else:
        req.logprob_start_len = -1
else:
    req.logprob_start_len = recv_req.logprob_start_len

if req.logprob_start_len > len(req.origin_input_ids):
    error_msg = f"{req.logprob_start_len=} is higher than ..."
    req.logprob_start_len = -1
    req.set_finish_with_abort(error_msg)
    self._add_request_to_queue(req)
    return
```

`logprob_start_len` 控制从 prompt 哪个位置开始返回每个 token 的 logprob。`-1` 表示「只算 output 的」。这段把用户的输入归一化成 0..len(prompt) 范围内的合法值。

### 6.1 背景:LLM 推理可以返回 logprob

`logprob`(log-probability) 是模型采样某个 token 时,该 token 在词表上的对数概率。常见用途:

- **评估**:eval 工具用 logprob 算 perplexity / NLL。
- **debug**:看模型为什么选了这个词。
- **重排**:beam search、speculative decoding 校验、best-of-N 重排。
- **classification trick**:用 LLM 做分类时,看候选标签的 logprob 哪个最高。

logprob 来自两个阶段:

| 阶段 | logprob 含义 | 来自哪次 forward |
|---|---|---|
| **input(prompt) logprob** | prompt 里每个 token 在它前面 prefix 条件下的概率 | 一次 prefill forward 顺便算出来(logits 是现成的) |
| **output(completion) logprob** | 生成的每个 new token 的采样概率 | 每个 decode step 算 |

prompt 第一个 token 没法算 logprob(没有 prefix 条件),从第二个开始。

### 6.2 `logprob_start_len` 控制从哪开始返回 input logprob

它是 OpenAI/SGLang API 暴露的一个参数,**单位是 token 索引**——告诉 server「我只要 input 序列中从位置 X 开始的 logprob」。

| 值 | 含义 |
|---|---|
| `-1`(默认) | **不返回 input logprob**,只返回 output logprob |
| `0` | 从 prompt 第 1 个 token(没法算)/ 第 2 个开始全算 |
| `len(prompt)` | 只算最后一个 prompt token 的 logprob(等价于「评估这段 prompt 的最后一步」) |
| 任意 0 < N < len(prompt) | 从 prompt 第 N 个 token 开始算到结尾 |

为什么需要起点参数:算 input logprob 不便宜——需要从 logits 里取出对应位置的 vocab 维度概率(vocab 几十万维),大批量请求下数据传输也是负担。让 client 指定起点,允许它「**只算我关心的尾段,跳过前面的 system prompt 之类**」。

### 6.3 「校准」具体校的是什么

「校准」就是把**用户原始传入的 `logprob_start_len`** 经过一套规则规范化成**实际可用的合法值**。三件事:

1. **冗余清除**:用户没要 logprob 却给了 start_len → 强制改成 `-1`(防御性,避免参数自相矛盾)。
2. **默认值填充**:用户要 logprob 但没指定 start_len(`-1`) → 自动选一个合理默认(只算 output 部分,因为这是 99% 的常见用法,不会浪费算力算 input logprob)。
3. **越界保护**:start_len > prompt 长度 → 直接 abort 请求(不报错地传到下游会让 logprob 数组对不齐)。

| 原始值(client 给的) | 校准后(`req.logprob_start_len`) | 行为 |
|---|---|---|
| `return_logprob=False, start_len=5` | `-1` | 不算 logprob(用户参数自相矛盾,server 修正) |
| `return_logprob=True, start_len=-1` | `len(prompt)` | 只算 output logprob(默认行为) |
| `return_logprob=True, start_len=10`(prompt 100 token) | `10` | 从位置 10 开始算到结尾 |
| `return_logprob=True, start_len=200`(prompt 100 token) | abort | 越界 |
| `is_prefill_only=True, start_len=-1` | `len(prompt)` | 跳过 input 也跳过 output(没 output) |

> **「校准」 = 把 client 给的 `logprob_start_len` 与 `return_logprob` 等其他参数交叉验证,填充默认值,处理边界情况,产出 scheduler 内部 `req` 真正使用的合法 start_len**——后续 `process_batch_result_prefill` / `stream_output` 都按这个校准后的值来算和发 logprob。

简而言之:**输入参数规范化**,让下游不用考虑各种异常组合。

---

## 七 Grammar 编译与最终入队

```python
added_to_grammar_queue = self.grammar_manager.process_req_with_grammar(req)
if not added_to_grammar_queue:
    self._add_request_to_queue(req)
```

如果请求带结构化输出约束(JSON schema、regex、EBNF)，先送入 `grammar_manager` 异步编译——编译完才能跑(否则采样时没有 mask)。`process_req_with_grammar` 返回 True 表示「我接管了这个请求，编译完成后会自己再调 `_add_request_to_queue`」；返回 False 表示「无 grammar 约束，请你自己入队」。

---

## 八 退出后的等待

`handle_generate_request` 返回时，`req` 已经在三个去处之一：

| 去处 | 何时 |
|---|---|
| `waiting_queue`(NULL 模式) | 普通请求 / session 失败 abort / grammar 编译完 |
| `disagg_prefill_bootstrap_queue` | PD 解耦 PREFILL 节点 |
| `disagg_decode_prealloc_queue` | PD 解耦 DECODE 节点 |
| `grammar_manager` 编译队列 | 带 grammar 还在编译 |

接下来 `get_next_batch_to_run` 会从这些队列里挑请求组 batch。

---

## 九 设计要点小结

| 决策 | 原因 |
|---|---|
| `Req` 与 `TokenizedGenerateReqInput` 解耦 | IPC 结构不可变,内部对象需要持续 mutate |
| session 路径独立分支 | 多轮对话需要拼接 prefix + 共享 KV |
| 多模态 padding 在 scheduler 端 | scheduler 是构造 KV slot 的唯一进程,统一在这里展开避免重复 |
| max_new_tokens 在此 clamp | 等 input_ids 完整(可能多模态展开后变长)再算可用空间 |
| grammar 异步编译 | 编译耗时不能阻塞主循环,编译完再回流入队 |
| PD bootstrap_room 必填 | 配对 prefill 与 decode 节点的 KV 传输 |
