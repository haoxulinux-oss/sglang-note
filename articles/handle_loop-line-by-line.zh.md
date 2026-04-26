# `handle_loop()` 每行代碼分析

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

整體角色：**TokenizerManager 進程裏唯一的後台消費者協程**，從 ZMQ 入站 socket 拉取下游（DetokenizerManager / Scheduler）發回來的所有消息，分發到對應的請求狀態或控制路徑上。

---

## L1613：`async def handle_loop(self)`

- `async def` ⇒ 這是個 coroutine function。它的存在前提是後面有人 `await` 或 `loop.create_task(handle_loop())` 把它調度起來。
- 不接受任何參數（除 `self`）：所有狀態都掛在 `self` 上（socket、watchdog、`rid_to_state`）。
- 返回類型隱式 `None` ——但這個函數**永不返回**（它是個 `while True`），返回類型不重要。
- 所謂「event loop」是指**這個函數內部自己組織的處理循環**，不是 asyncio 的 event loop。注意命名歧義：asyncio 的 event loop 是調度者，`handle_loop` 是被調度的協程之一。

---

## L1614：docstring `"""The event loop that handles requests"""`

純文檔，無運行時行為。提示讀者：這是一個處理請求結果的事件循環。

---

## L1615：`while True:`

無限循環。這條協程會一直跑到進程退出 / Task 被 cancel。

實踐中它退出的合法路徑只有兩條：

1. 進程關閉時 `task.cancel()` ⇒ `await ...recv_pyobj()` 處拋 `CancelledError` ⇒ 協程結束。
2. 內部某行未捕獲異常 ⇒ 由外層 `print_exception_wrapper` 打日誌並（通常）讓進程退出。

---

## L1616：`with self.soft_watchdog.disable():`

`soft_watchdog` 是 SGLang 的看門狗：定期檢查 `last_receive_tstamp` 是否在合理範圍內，**很久沒收到任何 scheduler/detokenizer 包就觸發異常**（懷疑下游 hang 了）。

`disable()` 是個上下文管理器，在 `with` 體內**暫時停掉看門狗計時**。原因：

- `await recv_pyobj()` 是「故意阻塞」——當下沒請求時自然不會有消息回來，可以等好久。
- 如果不暫停，看門狗會把這段空閒誤判為 hang。

進入 `with` 時 watchdog 被「靜音」，退出 `with` 時自動恢復。注意 **它只包了 `recv_pyobj` 這一行**，下面 dispatch 的部分仍然受監控（這部分不該久等）。

---

## L1617：`recv_obj = await self.recv_from_detokenizer.recv_pyobj()`

這是這個函數最核心的一句。

- `self.recv_from_detokenizer` 是 ZMQ 的 PULL socket（`tokenizer_manager.py:344` 構造），對端是 DetokenizerManager 的 PUSH（流式 token 輸出）以及若干控制信號 PUSH。
- `recv_pyobj()` 是 pyzmq 的方法：阻塞拉一條消息，自動 `pickle.loads` 成 Python 對象。
- `await` 前綴：SGLang 用的是 pyzmq 的 asyncio 適配（`zmq.asyncio.Socket`），所以這是非阻塞 await ——等消息到達期間 event loop 可以調度其他協程（`generate_request`、HTTP handler、cron 任務 …）。
- 結果 `recv_obj` 是反序列化後的 Python 對象，可能類型很多：
  - 流式輸出：`BatchStrOutput` / `BatchEmbeddingOutput` / `BatchTokenIDOutput`
  - 控制 / 管理回包：權重更新結果、profile ack、health 信號、abort ack、open/close session ack 等等。
- 這一行 **不會反壓 scheduler**：ZMQ 自有 buffer。如果消費太慢，buffer 漲，scheduler 那邊 `send_pyobj` 才開始阻塞。

---

## L1618-L1621：類型分支判斷

```python
if isinstance(
    recv_obj,
    (BatchStrOutput, BatchEmbeddingOutput, BatchTokenIDOutput),
):
```

判斷收到的是不是「批量推理輸出」三種類型之一：

- `BatchStrOutput`：detokenizer 已經把 token id 轉成文本。**這是 `/generate` 走 tokenizer 時的常規包**。
- `BatchEmbeddingOutput`：embedding 任務（`/encode`）的回包，沒有 detokenize 這一步。
- `BatchTokenIDOutput`：`skip_tokenizer_init=True` 時直接給回 token id（沒文本）。

這三類有共同形狀：含 `rids: List[str]`、按下標對齊的 `output_strs / output_ids / finished_reasons / meta_info` 等。所以可以走統一處理路徑。

---

## L1622：`await self._handle_batch_output(recv_obj)`

把這批輸出按 rid 分發回 `rid_to_state[rid].out_list`，並 `state.event.set()` 喚醒那些掛在 `_wait_one_response()` 裏的請求協程。

`_handle_batch_output` 內部會：

1. 遍歷 `recv_obj.rids`，找到對應 `ReqState`；找不到就 log 並跳過（已 abort 的請求）。
2. 拼 `meta_info`、output text、finish reason、token usage、logprobs 等成 out dict。
3. `state.out_list.append(out)`，並（按 batching 策略）`state.event.set()` 通知對應請求。
4. 對 `finished` 請求做計時、metrics、`del self.rid_to_state[rid]`。

之所以是 `await`：函數內部會 `await self.lora_registry.release(...)` 之類異步操作（LoRA 引用計數）。

**這也是為什麼必須暫停 watchdog 的精確設計**：watchdog 只測 `recv_pyobj` 之外的時間，這部分必須夠快——分發大批 rid + LoRA release 是 CPU 受限的，幾毫秒級。

---

## L1623-L1624：非批量輸出走獨立 dispatcher

```python
else:
    self._result_dispatcher(recv_obj)
```

`recv_obj` 不是三類批量輸出之一時，是各種**控制平面消息**：

- `UpdateWeightFromDiskReqOutput` / `UpdateWeightsFromDistributedReqOutput` …
- `OpenSessionReqOutput` / `CloseSessionReqOutput`
- `AbortReq` 的 ack
- `ProfileReq` ack
- `GetWeightsByNameReqOutput`
- 若干 health / state 廣播

`_result_dispatcher` 是一個基於類型的分派器（在 `__init__` 用 dict-of-callbacks 註冊）。它**不是 async**，是純同步 dispatch：根據 `type(recv_obj)` 找到對應的 handler（通常去 `set_result()` 某個 future 或 `set()` 某個 event），同步調一下。

注意對比：

- L1622 `await ...` —— 數據面，可能耗時。
- L1624 同步調用 —— 控制面，量小、純 set 標誌位。

---

## L1625：`self.last_receive_tstamp = real_time()`

更新「最後一次收到下游消息」的時間戳。

`real_time()` 是 monotonic 風格的時間源（一般是 `time.time()` 包裝）。這個值被多處使用：

- watchdog 判斷是否 hang。
- `/get_load`、`/server_info` 等 endpoint 報告活躍度。
- 一些超時兜底邏輯。

注意它在 dispatch **之後**更新——意味着「我已經處理完了一條消息」。這比放在 recv 之後更新更貼近「健康」的定義（不僅收到了，還消費完了）。

---

## L1626：`self.soft_watchdog.feed()`

「餵狗」：告訴看門狗「我活着、我消費了一條消息」。watchdog 會把計時器重置。

和 L1616 的 `disable()` 配合：

- 進入 recv 前 disable（recv 期間不計）。
- 收到 + dispatch 完一條消息後 feed（重置計時器）。

如果 dispatch 自己卡住（例如 `_handle_batch_output` 死循環）就會在沒 feed 的情況下被看門狗發現。

---

## 一輪循環的整體效果

```
┌─ while True ─────────────────────────────────────────────┐
│  ① 暫停 watchdog,await ZMQ recv 拿到 1 條消息           │
│  ② 是「批量推理輸出」?                                   │
│       └─ 是 → 異步分發到各 rid 的 ReqState,喚醒協程     │
│       └─ 否 → 同步走控制面 dispatcher                   │
│  ③ 更新 last_receive_tstamp                             │
│  ④ 餵 watchdog                                          │
└──────────────────────────────────────────────────────────┘
回到 ①
```

---

## 幾個容易誤解的點

1. **「event loop」名稱歧義**：函數叫 handle_loop，內部 while True，但它**不是** asyncio 的 event loop，而是 event loop 上的一個 task。整個進程依然只有一條 asyncio loop。

2. **單線程 vs 多請求**：一輪循環只處理 1 條 ZMQ 消息，但**1 條消息可能包含多個 rid 的輸出**（scheduler 是按 batch 推回來的）。所以這個「串行 loop」不會成為並發瓶頸——它做的是「拉一個批 → fan-out 到 N 個請求」。

3. **為什麼 dispatch 不全做成 async**：批量數據面的清理 / metrics 寫入有 IO（雖然輕量），故 `_handle_batch_output` 是 async；控制面只是 set 標誌位，沒必要走協程開銷，故 `_result_dispatcher` 是同步函數。

4. **異常路徑**：任何一行未捕獲異常會 bubble out，被外層 `print_exception_wrapper` 接住打日誌。如果 `recv_pyobj` 拋 `CancelledError`（進程關閉），協程乾淨退出。

5. **背壓**：handle_loop 阻塞或變慢 ⇒ ZMQ buffer 漲 ⇒ DetokenizerManager 的 `send_pyobj` 阻塞 ⇒ 進一步上游 scheduler 也會阻塞。這是個天然反壓鏈，但 buffer 容量很大（默認 ZMQ HWM 1000+），實踐中幾乎不會反壓。
