# `_handle_batch_output()` 函數解析

**位置：** `python/sglang/srt/managers/tokenizer_manager.py:1628-1863`

這是 `handle_loop` 收到「批量輸出」消息後的核心分發函數。負責：
- 把 scheduler/detokenizer 一個批內的多個 rid 結果**拆開**；
- 為每個 rid 構造 `out_dict`（將被 yield 給 HTTP 響應）；
- 喚醒對應的請求協程；
- 對 `finished` 請求做計時、metrics、清理 `rid_to_state`、釋放 LoRA。

---

## 一、三類批量輸出結構的含義

它們都是 `python/sglang/srt/managers/io_struct.py` 中的 dataclass，是 **scheduler/detokenizer → tokenizer_manager** 跨進程消息的標準載體。共同點：所有字段都是按 `rids` 索引對齊的列表。

### 1. `BatchStrOutput` —— 含文本的生成結果（最常見）

走 `/generate`、`/v1/chat/completions` 等需要解碼成字符串的請求時，**DetokenizerManager 處理完 token→text 轉換後**回傳。

主要字段：

| 字段 | 含義 |
|---|---|
| `rids: List[str]` | 這個批裡每個請求的 id |
| `output_strs: List[str]` | 本步增量生成的文本（delta） |
| `output_ids: List[List[int]]` | 本步增量生成的 token id |
| `finished_reasons: List[Optional[dict]]` | 為 None 表示還在跑；非 None 表示結束（type=stop/length/abort，含 status_code/message） |
| `prompt_tokens / completion_tokens / cached_tokens / reasoning_tokens` | 計費 / metrics 用 |
| `cached_tokens_details` | RadixCache / HiCache 命中分層細節 |
| `retraction_counts` | 因 KV 不足回退（preempt-and-retract）發生過的次數 |
| `output_hidden_states` | 若請求要求返回隱藏狀態 |
| `routed_experts` | MoE 模型按需返回的專家路由信息 |
| `time_stats` | scheduler 端的時間採樣（queue/prefill/decode 等） |
| `load` | DP 模式下這個 rank 的當前負載快照 |
| `dp_ranks` | 哪個 DP rank 服務了這個請求 |
| `customized_info` | 模型/插件自定義信息 |

語義：**「scheduler 一輪 batch 內所有產生新增 token 的請求」的合集**，文本側已被 detokenizer 處理。

### 2. `BatchTokenIDOutput` —— 純 token id（跳過 detokenize）

當 server 啟動時 `--skip-tokenizer-init`，TokenizerManager 不持有 tokenizer，detokenizer 也不會把 id 轉文本。此時 scheduler 直接回傳 token id。

字段相比 `BatchStrOutput` **少一個 `output_strs`**，其它幾乎一樣。客戶端要自行 detokenize（適合 token-in / token-out 的下游服務、benchmark、speculative decoding 編排層）。

### 3. `BatchEmbeddingOutput` —— Embedding 推理結果

`/encode`、`/v1/embeddings`、cross-encoder 等不做生成、只取池化向量的任務。每個請求只產生**一次性結果**。

主要字段：

| 字段 | 含義 |
|---|---|
| `rids: List[str]` | 請求 id |
| `embeddings: List[List[float]]` | 池化後的向量（或 `[CLS]` 向量等） |
| `pooled_hidden_states` | 可選：池化前的 hidden states |
| `prompt_tokens / finished_reasons / retraction_counts / time_stats` | 與生成類共有的計費/狀態字段 |

它沒有 `output_ids / completion_tokens / cached_tokens` 這類**生成類**字段，這也是函數內 `if not isinstance(recv_obj, BatchEmbeddingOutput): meta_info.update({...completion_tokens...})` 的判斷依據。

---

## 二、函數逐段解析

### 1. 開頭：批量喚醒緩衝

```python
pending_notify: dict[str, ReqState] = {}
batch_notify_size = self.server_args.batch_notify_size
```

不是收到一條就立刻 `state.event.set()`，而是攢一小批再批量 set，以降低 asyncio 喚醒抖動。`batch_notify_size` 是可調的（默認小批量）。

### 2. 主循環：按 rid 拆分

```python
for i, rid in enumerate(recv_obj.rids):
    state = self.rid_to_state.get(rid, None)
    if state is None:
        logger.error(...)
        continue
```

對 batch 中每一條按 `rid` 找到對應的 `ReqState`。找不到通常是請求已被 abort 並從 `rid_to_state` 刪除——這種「孤兒」結果直接丟棄並打 error 日誌。

### 3. 構造通用 `meta_info`

```python
meta_info = {
    "id": rid,
    "finish_reason": recv_obj.finished_reasons[i],
    "prompt_tokens": recv_obj.prompt_tokens[i],
    "weight_version": self.server_args.weight_version,
    "total_retractions": recv_obj.retraction_counts[i],
}
```

`weight_version` 用於熱更新權重後客戶端能識別「這個結果來自哪個權重版本」。`total_retractions` 在做穩定性監控時很關鍵——retraction 多通常意味着 KV 池過載。

### 4. 可選擴展字段

- `enable_metrics + time_stats`：把 scheduler 階段的時間採樣（recv→queue→prefill→decode→send 等）拍進 `meta_info`，最終 client 能看到端到端 + 各階段耗時。
- `return_logprob`：調 `convert_logprob_style` 把 token 級 logprob、top-k logprob 補進 `meta_info`。如果 `return_text_in_logprobs=True` 且 server 沒 skip tokenizer，還會把 token id 反 detokenize 成文本片段。
- 非 embedding：補 `reasoning_tokens / completion_tokens / cached_tokens(_details)`。其中 `cached_tokens_details` 反映 RadixCache 命中、HiCache L1/L2、disk 等的命中量。
- `output_hidden_states`、`routed_experts`、`customized_info`、`dp_ranks` 都是按需追加。

### 5. 標記是否完成

```python
state.finished = recv_obj.finished_reasons[i] is not None
```

**這是「請求是否結束」的單點寫入**。後續 `_wait_one_response` 看的就是這個 flag。

### 6. 三類分支構造 `out_dict`

#### 6.1 `BatchStrOutput` 分支

```python
delta_text = recv_obj.output_strs[i]
delta_output_ids = recv_obj.output_ids[i]
state.append_text(delta_text)
state.output_ids.extend(delta_output_ids)
```

把這一步的 delta 累積到 `state` 裡。然後分三種模式產出：

- **incremental streaming**（`incremental_streaming_output=True 且 stream=True`）：直接吐 delta 文本+id，並通過 `_slice_streaming_output_meta_info` 把 logprob 等切片到本步，更新 `last_output_offset`。客戶端拿到的是真正的「增量幀」。
- **non-incremental streaming**（默認 stream 模式）：
  - 中間幀 `text=None, output_ids=state.output_ids（引用，不拷貝）`，**把字符串重建延遲到 `_wait_one_response` 那邊一次性 join**。這是為了避免每步 `''.join(many_chunks)` 的 O(n²) 累積開銷。
  - 結束幀才 `state.get_text()` 真正拼成完整字符串。
- **non-stream**：只在 `state.finished` 時返回完整文本+id；中間步驟 `out_dict = None`，根本不入 `out_list`。

#### 6.2 `BatchTokenIDOutput` 分支

和上面類似，**少了 text 字段**。同樣有 incremental / non-incremental / 結束三種模式。

#### 6.3 `BatchEmbeddingOutput` 分支

```python
out_dict = {
    "embedding": recv_obj.embeddings[i],
    "meta_info": meta_info,
}
if recv_obj.pooled_hidden_states is not None and recv_obj.pooled_hidden_states[i] is not None:
    out_dict["pooled_hidden_state"] = recv_obj.pooled_hidden_states[i]
```

embedding 任務沒有「增量」概念，一次出結果就是終態。

### 7. 計時與完成處理

```python
if state.time_stats.first_token_time == 0.0:
    state.time_stats.set_first_token_time()
```

整個請求生命週期裡 **first token 時間只寫一次**（用來算 TTFT）。

```python
if state.finished:
    # 寫 trace、e2e_latency
    # speculative decoding 額外指標
    # metrics
    del self.rid_to_state[rid]
    if state.obj.lora_path:
        asyncio.create_task(self.lora_registry.release(state.obj.lora_id))
```

完成時：
- 記錄 e2e latency、scheduler 階段拼起來的詳盡時間樣本。
- 從 `rid_to_state` **刪除條目**（防止內存洩漏）。
- 異步釋放 LoRA 引用計數（不阻塞當前 dispatch）。

### 8. 入隊 + 批量喚醒

```python
if out_dict is not None:
    state.out_list.append(out_dict)
    pending_notify[rid] = state

    if len(pending_notify) >= batch_notify_size:
        for s in pending_notify.values():
            s.event.set()
        pending_notify = {}
        await asyncio.sleep(0)
```

- 把 `out_dict` 塞進 `state.out_list`，等 `_wait_one_response` 來取。
- 攢夠 `batch_notify_size` 就批量 `event.set()`，並 `await asyncio.sleep(0)` **主動讓出**——讓被喚醒的請求協程能立刻被調度走（否則一直在 for 循環裡，整批處理完才有機會跑）。
- 這個讓出是 SGLang 對「**SSE 首 token 延遲**」的關鍵優化：高並發時不讓任何單個請求被一個大 batch 攔住。

### 9. 每 rid 末尾：可選 metrics / dump

```python
if self.enable_metrics and state.obj.log_metrics:
    self.collect_metrics(state, recv_obj, i)
if self.dump_requests_folder and state.finished and state.obj.log_metrics:
    self.dump_requests(state, out_dict)
if self.crash_dump_folder and state.finished and state.obj.log_metrics:
    self.record_request_for_crash_dump(state, out_dict)
```

- Prometheus / 自定義 metrics 採集。
- 結束時可選 dump 整條請求 → 用於離線回放、糾錯、分析。
- crash dump：保留最近 N 條請求，崩潰後可幫助復現。

### 10. 收尾

```python
for s in pending_notify.values():
    s.event.set()
```

把循環尾部還沒攢滿 `batch_notify_size` 的剩餘 set 出去。

```python
if (
    self.server_args.dp_size > 1
    and isinstance(recv_obj, (BatchStrOutput, BatchTokenIDOutput))
    and recv_obj.load is not None
):
    load_update_req = WatchLoadUpdateReq(loads=[recv_obj.load])
    self.send_to_scheduler.send_pyobj(load_update_req)
```

DP（data parallel）模式下，把這個 rank 報的最新負載**回送**給 scheduler / 路由層。下次 DP 路由就有更新後的負載信息可用。embedding 沒有這條（DP load 由生成路徑驅動）。

---

## 三、整體數據流總結

```
DetokenizerManager / Scheduler                      handle_loop()
  └─ send_pyobj(BatchStrOutput|BatchTokenIDOutput|   └─ recv_pyobj()
                BatchEmbeddingOutput, batch=N)            │
                                                          ▼
                                        _handle_batch_output(recv_obj)
                                                          │
                          ┌───────────────────────────────┴──────────────────────────┐
                          │  for i, rid in enumerate(recv_obj.rids):                 │
                          │     state = rid_to_state[rid]                            │
                          │     meta_info = {...rid 共有...}                          │
                          │     if BatchStrOutput   → 累積 delta 文本/id, 構造 out_dict│
                          │     if BatchTokenIDOutput → 累積 delta id, 構造 out_dict   │
                          │     if BatchEmbeddingOutput → 一次性 embedding out_dict    │
                          │     if state.finished: 寫 metrics / del rid_to_state[rid] │
                          │     state.out_list.append(out_dict); pending_notify[rid]  │
                          │     if 攢夠 batch_notify_size: event.set() + sleep(0)     │
                          └───────────────────────────────┬──────────────────────────┘
                                                          ▼
                                       剩餘 pending_notify.values().event.set()
                                                          ▼
                          DP 模式: 把 recv_obj.load 回送給 scheduler 做負載感知路由
```

---

## 四、設計亮點 & 易踩坑點

1. **「一個 batch 消息對應 N 個請求結果」**。這是 SGLang 高吞吐的關鍵——前後端解耦，每次跨進程通信不是 1:1，而是 1:N。
2. **`pending_notify` 批量喚醒 + `await asyncio.sleep(0)`**：在公平性和事件循環抖動之間取得平衡，避免「先收到結果的請求要等所有 rid 處理完才被喚醒」。
3. **non-incremental streaming 故意把字符串拼接延後**：把 O(n²) 變成 O(n)。
4. **`del self.rid_to_state[rid]` 必須只在 `state.finished` 分支裡發生一次**，否則會出現孤兒喚醒事件再也沒人接收的狀況。
5. **abort 分支在這裡不單獨處理**——abort 是 scheduler 端把請求標 `finished + finish_reason.type="abort"`，本函數依然走 finish 分支正常 yield 給 client（client 就能看到 abort 原因）。
6. **`BatchEmbeddingOutput` 不走 `state.append_text` / `output_ids` 累積**——避免在 embedding 任務上跑生成側狀態機。
7. **`load` 字段只在生成路徑回傳**——這也是為什麼 DP 路由邏輯主要照顧生成負載。
