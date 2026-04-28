# `Scheduler.stream_output()` / `stream_output_generation()` 解析

**位置：** `python/sglang/srt/managers/scheduler_output_processor_mixin.py:910-1217`

**角色：** scheduler 进程把这一批的 token 输出 / embedding 输出打包成 `BatchTokenIDOutput` / `BatchEmbeddingOutput`,通过 ZMQ PUSH 给 DetokenizerManager。这是 scheduler 把数据交还出去的最后一步。

---

## 一 顶层分派

```python
def stream_output(
    self: Scheduler,
    reqs: List[Req],
    return_logprob: bool,
    skip_req: Optional[Req] = None,
):
    """Stream the output to detokenizer."""
    if self.is_generation:
        self.stream_output_generation(reqs, return_logprob, skip_req)
    else:  # embedding or reward model
        self.stream_output_embedding(reqs)

    if envs.SGLANG_TEST_CRASH_AFTER_STREAM_OUTPUTS.get() > 0:
        self._trigger_crash_for_tests(...)
```

`is_generation` 走 generation 路径(GPT 类),否则走 embedding 路径(BGE / reward model 等)。后面的 crash 钩子是 chaos 测试用。

---

## 二 stream interval:决定本批要不要 send

每个请求是否在本轮被发送由 `should_output` 决定:

```python
if req.finished():
    if req.finished_output:
        continue                 # 同一个 finished 请求别重复发(overlap 双输出兜底)
    req.finished_output = True
    if req.finished_len is None:
        req.finished_len = len(req.output_ids)
    should_output = True         # 完成必发
else:
    if req.stream:
        stream_interval = (
            req.sampling_params.stream_interval or self.stream_interval
        )
        should_output = (
            len(req.output_ids) % stream_interval == 1
            if stream_interval > 1
            else len(req.output_ids) % stream_interval == 0
        )
        if should_output:
            should_output &= not req.check_match_stop_str_prefix()
    else:
        should_output = (
            len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
        )
```

要点:

- **finished 请求必发**(且只发一次)。`finished_output` 标志防 overlap 模式下「请求 finish 后还跑了一轮 dummy decode」造成重复发送。
- **stream=True**:按 `stream_interval` 节流。常见配置 `stream_interval=1`(每 token 都发,SSE 流式);也可以设 4、8 之类合并发送降 ZMQ frame 数。
- **stop string 前缀检查**:`check_match_stop_str_prefix` 判断 detokenize 后的 tail 是否可能匹配 stop_str 的某个前缀——如果可能,**暂时不发**(等下一个 token 揭晓是否真停;否则可能把已经发出去的 stop_str 前缀显示给用户,后面又「撤回」,client 体验差)。
- **stream=False**:平时不发,但每隔 `DEFAULT_FORCE_STREAM_INTERVAL`(默认 50) 还是要发一次——目的是让 detokenizer 进程**逐步消化 token id**,不要等到 finish 才一次性吐 N 千 token,会引起短暂内存高峰。这也是非流式请求 client 看到的最终 text 是「累积的」。

---

## 三 字段并行 list:批化打包

```python
rids = []
http_worker_ipcs = []
finished_reasons: List[BaseFinishReason] = []

decoded_texts = []
decode_ids_list = []
read_offsets = []
output_ids = []

skip_special_tokens = []
spaces_between_special_tokens = []
no_stop_trim = []
prompt_tokens = []
reasoning_tokens = []
completion_tokens = []
cached_tokens = []
cached_tokens_details = []
spec_verify_ct = []
spec_accepted_tokens = []
spec_acceptance_histogram = []
retraction_counts = []
output_hidden_states = None
load = self.get_loads(GetLoadsReqInput(include=["core"]))
routed_experts = None
customized_info = {}
time_stats = []
```

每个字段都是按 batch 索引并列的 list——这是「**批次出**」结构(对照 `BatchTokenIDOutput` 字段定义),让一次 ZMQ send 携带 N 个请求的输出。

`load` 在外面调一次(这一批的 KV/req pool 利用率快照),也跟着发出去——给 router / load balancer 当反馈。

---

## 四 增量 send 状态:`send_token_offset` / `send_decode_id_offset`

```python
send_token_offset = req.send_token_offset
send_output_token_logprobs_offset = req.send_output_token_logprobs_offset

decode_ids, read_offset = req.init_incremental_detokenize()
decode_ids_list.append(decode_ids[req.send_decode_id_offset:])

output_ids_ = req.output_ids_through_stop
req.send_decode_id_offset = len(decode_ids)
read_offsets.append(read_offset)
output_ids.append(output_ids_[send_token_offset:])
req.send_token_offset = len(output_ids_)
```

每个 req 维护两个**已发送 offset**：

| offset | 作用 |
|---|---|
| `send_token_offset` | 上次 stream_output 发到第几个 output_id 了 |
| `send_decode_id_offset` | 上次发到 decode_ids(供 incremental detokenize 用) 第几个了 |
| `send_output_token_logprobs_offset` | logprob 也一样 |

每次只发自上次以来**新增的部分**——不是把整段累积输出重发。这是流式增量传输的核心:O(总长) 总流量 而非 O(总长²)。

`init_incremental_detokenize`:返回从某个 prefix offset 开始的 token id 列表 + read_offset(detokenizer 用来对齐 special token 边界)。

`output_ids_through_stop`:如果命中 stop_str,这里返回的是「截到 stop_str 之前的部分」——保证 client 看到的 text 不包含 stop_str 本身(行为符合 OpenAI API)。

---

## 五 logprob 字段(可选)

```python
if return_logprob:
    if (
        req.return_logprob
        and not req.input_logprob_sent
        and self.disaggregation_mode != DisaggregationMode.DECODE
        and req.input_token_logprobs_val is not None
    ):
        # 输入 logprobs 只发一次(prefill 完成后)
        input_token_logprobs_val.append(req.input_token_logprobs_val)
        ...
        req.input_logprob_sent = True
    else:
        # 占位空 list
        input_token_logprobs_val.append([])
        ...

    if req.return_logprob:
        # 输出 logprobs 增量发
        logprob_end = max(len(output_ids_), 1)
        output_token_logprobs_val.append(
            req.output_token_logprobs_val[
                send_output_token_logprobs_offset:logprob_end
            ]
        )
        ...
        req.send_output_token_logprobs_offset = logprob_end
```

要点：

- **input_logprobs 只发一次**:prefill 完成后那一次发,后续 decode 都是空 list。`input_logprob_sent` 标志位防重发。
- **DECODE 节点不发 input_logprobs**:PD 解耦下 input_logprobs 是 prefill 节点算的,decode 节点没有这部分数据。
- **output_logprobs 按窗口增量发**:`[send_output_token_logprobs_offset:logprob_end]`,机制和 token 一样。
- **整 batch 内字段按 req 对齐**:某些 req 没 logprob 时填空 list 占位(让 N 个 req 的所有字段长度一致)。

---

## 六 其他 optional 字段

```python
if req.return_hidden_states:
    if output_hidden_states is None:
        output_hidden_states = []
    output_hidden_states.append(req.hidden_states)

if req.return_routed_experts:
    if routed_experts is None:
        routed_experts = []
    routed_experts.append(req.routed_experts)

if req.customized_info is not None:
    for k, v in req.customized_info.items():
        if k not in customized_info:
            customized_info[k] = []
        customized_info[k].append(
            v[send_token_offset : len(output_ids_)]
        )
```

- **hidden_states**:embedding 相关任务可以让 client 拿到中间层 hidden states。
- **routed_experts**:MoE 模型让 client 看路由情况,用于 debug / expert profiling。
- **customized_info**:用户在 model 实现里自定义的额外输出(如 reasoning chain 中间态)。

每个都按 batch 列对齐,有就 append,没有保持 None。

---

## 七 finish 时记录 time stats

```python
if (
    req.finished()
    and self.attn_tp_rank == 0
    and self.server_args.enable_request_time_stats_logging
):
    req.log_time_stats()
```

完成的请求在主 rank 上把完整的 time_stats(从入队到完成各阶段时间) log 出来,用于 latency profiling。仅 attn_tp_rank=0 做(避免每个 rank 重复 log)。

---

## 八 ZMQ send 出去

```python
dp_ranks = [self.dp_rank] * len(rids) if rids else None

if reqs or is_idle_batch:
    self.send_to_detokenizer.send_output(
        BatchTokenIDOutput(
            rids=rids,
            http_worker_ipcs=http_worker_ipcs,
            spec_verify_ct=spec_verify_ct,
            spec_accepted_tokens=spec_accepted_tokens,
            spec_acceptance_histogram=spec_acceptance_histogram,
            time_stats=time_stats,
            finished_reasons=finished_reasons,
            decoded_texts=decoded_texts,
            decode_ids=decode_ids_list,
            read_offsets=read_offsets,
            output_ids=output_ids,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            no_stop_trim=no_stop_trim,
            prompt_tokens=prompt_tokens,
            reasoning_tokens=reasoning_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            cached_tokens_details=cached_tokens_details,
            input_token_logprobs_val=input_token_logprobs_val,
            input_token_logprobs_idx=input_token_logprobs_idx,
            output_token_logprobs_val=output_token_logprobs_val,
            output_token_logprobs_idx=output_token_logprobs_idx,
            input_top_logprobs_val=input_top_logprobs_val,
            input_top_logprobs_idx=input_top_logprobs_idx,
            output_top_logprobs_val=output_top_logprobs_val,
            output_top_logprobs_idx=output_top_logprobs_idx,
            input_token_ids_logprobs_val=input_token_ids_logprobs_val,
            input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
            output_token_ids_logprobs_val=output_token_ids_logprobs_val,
            output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
            output_token_entropy_val=None,
            output_hidden_states=output_hidden_states,
            routed_experts=routed_experts,
            customized_info=customized_info,
            placeholder_tokens_idx=None,
            placeholder_tokens_val=None,
            retraction_counts=retraction_counts,
            load=load,
            dp_ranks=dp_ranks,
        )
    )
```

要点：

- **`dp_ranks`**:每个 rid 标记来自哪个 DP rank——detokenizer 和 tokenizer manager 用这个做 DP 路由。
- **`is_idle_batch`**:DP attention idle batch 也要发(rids 是空的),这样下游 detokenizer 知道这一帧没有数据要 detokenize,但本轮调度走过了——和 `last_receive_tstamp` 等心跳逻辑配合。
- **`reqs or is_idle_batch`**:剩下的情况(全 skip 没 should_output 的 req)就不发了,节省一次 ZMQ frame。

---

## 九 `stream_output_embedding`(简述)

走类似的 fan-out 逻辑,但只有 `embeddings / pooled_hidden_states` 等字段,装进 `BatchEmbeddingOutput`。

embedding 任务一次性出结果(没有流式),所以每个完成的 req 只发一次。

---

## 十 设计要点小结

| 决策 | 原因 |
|---|---|
| 一次 ZMQ send 装 N 个 rid | 减少 ZMQ frame 数,降低 IPC overhead |
| 增量 offset(send_token_offset 等) | 流式 O(n) 总流量,而不是 O(n²) |
| stream_interval 节流 | 平衡 SSE 实时性与 IPC 开销 |
| stop_str 前缀暂缓发送 | 避免 client 看到 stop_str 残段 |
| input_logprobs 只发一次 | 它在 prefill 后就是定值,后续 decode 重发是浪费 |
| 发 token id,detokenize 留给独立进程 | 把 CPU 重活从 scheduler 主循环剥离 |
| finished 标志防重复发 | overlap 模式下 finish 后会跑一个 dummy decode |
| idle batch 也发 | 心跳作用,下游 watchdog 能看到 scheduler 还在跑 |

---

## 十一 衔接

`BatchTokenIDOutput` → ZMQ → DetokenizerManager → 文本化 → `BatchStrOutput` → ZMQ → TokenizerManager 的 `handle_loop` → `_handle_batch_output` → `rid_to_state[rid].out_list.append + state.event.set()` → 唤醒 `_wait_one_response` → SSE → HTTP client。

整条链路在 [Scheduler 整体流程](scheduler-recv-and-run.zh.md) 末尾的 ASCII 图里有完整呈现。
