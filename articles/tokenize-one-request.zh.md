# `_tokenize_one_request()` 与 `_tokenize_texts()` 解析

**位置：** `python/sglang/srt/managers/tokenizer_manager.py:692-799`、`tokenizer_manager.py:612-690`

这是 TokenizerManager 把一个 HTTP 单请求(`is_single=True`)从「文本/token id/embedding」转换成可跨进程发送的 `TokenizedGenerateReqInput / TokenizedEmbeddingReqInput` 的核心步骤。逻辑分两层：

- `_tokenize_one_request(obj)` —— 处理「输入到底是什么」的分支决策、多模态、长度校验，最终把所有片段拼成一个 `Tokenized*ReqInput`。
- `_tokenize_texts(texts, is_cross_encoder)` —— 真正调 HuggingFace tokenizer 把字符串变成 `input_ids`。

下面分两节剖析。

---

## 一、`_tokenize_one_request(obj)` 逐段解析

### 1. 入口与初始化

```python
input_embeds = None
input_text = obj.text
token_type_ids = None
is_cross_encoder_request = (
    isinstance(obj, EmbeddingReqInput) and obj.is_cross_encoder_request
)
```

- `obj` 可能是 `GenerateReqInput`(生成)或 `EmbeddingReqInput`(向量)。
- `is_cross_encoder_request`：BERT 风格的 cross-encoder（输入是 `[query, doc]` 对，需要 `token_type_ids` 标段）。这是 embedding 路径独有的分支条件。
- `input_embeds / token_type_ids` 默认为空，按情况填充。

### 2. 三种输入来源的优先级

```python
if obj.input_embeds is not None:
    if not self.server_args.disable_radix_cache:
        raise ValueError(...)
    input_embeds = obj.input_embeds
    input_ids = obj.input_ids
elif obj.input_ids is not None:
    input_ids = obj.input_ids
else:
    if self.tokenizer is None:
        raise ValueError(...)
    if not input_text and self.mm_processor and obj.contains_mm_input():
        input_ids = []
    else:
        input_ids, token_type_ids = await self._tokenize_texts(
            input_text, is_cross_encoder_request
        )
```

三条路径按优先级排：

| 来源 | 触发条件 | 处理 |
|---|---|---|
| **`input_embeds`** | 用户直接给 hidden vector 当输入 | 必须 `--disable-radix-cache`(因为 RadixCache 用 token id 做 key，embed 直接 bypass)；保留 `input_ids` 用于占位长度 |
| **`input_ids`** | 用户已经预先 tokenize 好 | 直接用，跳过 tokenizer 调用——常见于 benchmark、token-in 服务、或 server 启动时 `--skip-tokenizer-init` 的下游 |
| **`text`** | 最常见 | 必须有 tokenizer；若是「纯音频/纯图像」请求(text 为空 + 有 mm 输入)，先用 `input_ids = []` 占位，后面 mm_processor 会覆盖；其余情况调 `_tokenize_texts` |

「skip tokenizer」分支处的 `ValueError` 是清晰的边界：server 启动时若选择不加载 tokenizer，那本进程内就没法 encode 文本，让客户端自己 tokenize。

### 3. 多模态处理

```python
if self.mm_processor and obj.contains_mm_input():
    # 把 image_data/video_data/audio_data 包成 list
    self._validate_mm_limits(obj)
    mm_inputs = None
    if (not language_only) or encoder_transfer_backend in ["zmq_to_tokenizer", "mooncake"]:
        if language_only:
            mm_inputs = await self.mm_receiver.recv_mm_data(...)
        if mm_inputs is None:
            mm_inputs = await self.mm_processor.process_mm_data_async(...)
    elif language_only and encoder_transfer_backend == "zmq_to_scheduler" and not need_wait_for_mm_inputs:
        mm_inputs = await self.mm_processor.process_mm_data_async(...)
```

要点：

- **`contains_mm_input()`** 看请求里是否有 image/video/audio 字段。
- SGLang 支持 **EPD 解耦**(Encoder–Prefill–Decode 分离)：在 `language_only=True` 模式下，编码器跑在另一台机器上，本进程只承担语言模型部分。
  - `encoder_transfer_backend="zmq_to_tokenizer"` 或 `"mooncake"`：tokenizer 这一侧需要从 mm_receiver 拉编码后的多模态特征。
  - `encoder_transfer_backend="zmq_to_scheduler"`：编码器输出会被直接发到 scheduler 端，tokenizer 这边只在「单图」之类未走 dispatch 的情况下本地处理。
- 非 `language_only` 模式下，本地直接 `mm_processor.process_mm_data_async(...)` 把图/音/视频转成 `mm_inputs`(含 patch embedding、特殊 token 占位等)。

```python
if mm_inputs and mm_inputs.input_ids is not None:
    input_ids = mm_inputs.input_ids
if mm_inputs and mm_inputs.token_type_ids is not None:
    token_type_ids = mm_inputs.token_type_ids
    if not isinstance(token_type_ids, list):
        token_type_ids = token_type_ids.flatten().tolist()
if envs.SGLANG_MM_PRECOMPUTE_HASH.get() and mm_inputs and mm_inputs.mm_items:
    for item in mm_inputs.mm_items:
        if isinstance(item, MultimodalDataItem):
            item.set_pad_value()
```

mm_processor 会**重写** `input_ids`(把原始文本里的占位符替换成 `<|image_pad|>` 等真正的 token id 序列)。如果开了 `SGLANG_MM_PRECOMPUTE_HASH`，预先计算多模态片段的 hash 用来给 RadixCache 做 multimodal-aware 命中。

无 mm 时直接 `mm_inputs = None`，下游 scheduler 知道这是纯文本。

### 4. 长度校验

```python
self._validate_one_request(obj, input_ids)
```

调用 `_validate_one_request`(`tokenizer_manager.py:801`)：

- 计算 `input_token_num = len(input_ids) + num_reserved_tokens`。
- 如果 `>= context_len`：
  - 配了 `allow_auto_truncate` 就 `del input_ids[max_req_len:]` 截断。
  - 否则抛 `ValueError`，请求被拒。

「保留 token」是因为 sampling/special token 也要占空间(EOS、特殊起止符、speculative decoding 多预留若干等)。

### 5. 最终封装

```python
return self._create_tokenized_object(
    obj, input_text, input_ids, input_embeds, mm_inputs, token_type_ids
)
```

`_create_tokenized_object`(`tokenizer_manager.py:944`)的工作：

1. **解析 sampling_params**：和 `preferred_sampling_params`(server 启动时传的默认值)合并 → 实例化 `SamplingParams` → `normalize(tokenizer)`(把 stop strings 转成 token id) → `verify(vocab_size)`。
2. **构造跨进程对象**：
   - 生成请求：`TokenizedGenerateReqInput(input_text, input_ids, mm_inputs, sampling_params, ..., rid, lora_id, session_params, custom_logit_processor, ...)`。
   - 向量请求：`TokenizedEmbeddingReqInput(input_text, input_ids, mm_inputs, token_type_ids, sampling_params, ..., dimensions, ...)`。
3. **生成 `bootstrap_room`**：Disaggregation 模式下用于 PD 路由的房间号；fake backend 直接自增。
4. **解析 embed overrides**：embedding 任务里的 `embed_override_token_id` 现在已经知道 `input_ids` 了，可以解析具体位置。

返回值就是后面 `_send_one_request` 直接 `send_pyobj` 出去的载体。

---

## 二、`_tokenize_texts(texts, is_cross_encoder)` 逐段解析

这是真正接触 HuggingFace tokenizer 的地方，目标是把字符串(单条或一批)变成 `input_ids`。

### 1. 入参形态

支持四种：

```text
普通：
  "How are you?"
  ["Hello", "World", "How are you?"]

Cross-encoder(BERT 类对偶):
  [["query text", "document text"]]
  [["q1", "d1"], ["q2", "d2"], ["q3", "d3"]]
```

`is_cross_encoder=True` 时返回 `token_type_ids`(段标记 0/1)，否则不返回。

### 2. 早期校验

```python
if not texts or self.tokenizer is None:
    raise ValueError("texts cannot be empty and tokenizer must be initialized")
```

空字符串 / 空列表 / 没有 tokenizer 都拒绝。

### 3. 输入格式判定

```python
input_format = self._detect_input_format(texts, is_cross_encoder)
tokenizer_input = self._prepare_tokenizer_input(texts, input_format)
original_batch_size = len(texts) if not isinstance(texts, str) else 1
```

`_detect_input_format` 用 `InputFormat` 枚举(`tokenizer_manager.py:207`)：

| 输入 | InputFormat | 说明 |
|---|---|---|
| `"hi"` | `SINGLE_STRING` | 单条文本 |
| `["a", "b"]` | `BATCH_STRINGS` | 普通批 |
| cross-encoder + `[["q","d"], ...]` | `CROSS_ENCODER_PAIRS` | 句对(段 0 / 段 1) |

`_prepare_tokenizer_input` 把单字符串包成 `["..."]`，让下游永远以 list 形式喂 tokenizer，简化代码。

### 4. tokenizer 参数

```python
tokenizer_kwargs = (
    {"return_token_type_ids": is_cross_encoder} if is_cross_encoder else {}
)
```

只有 cross-encoder 才需要 `token_type_ids`，普通生成 / 单段 embedding 都不需要。

### 5. 选择 tokenization 策略

```python
use_async_tokenizer = (
    self.async_dynamic_batch_tokenizer is not None
    and input_format == InputFormat.SINGLE_STRING
)
```

两条路：

#### 5.1 `async_dynamic_batch_tokenizer`(动态批 tokenizer)

只在以下条件同时满足时使用：
1. 启动参数 `--enable-dynamic-batch-tokenizer`，且没有 `--skip-tokenizer-init`(`tokenizer_manager.py:329`)。
2. **当前是单字符串**输入。

它内部维护一个微小的 buffer：把多个**并发到达的单条 tokenize 请求**自动凑成一个 batch 一次性 encode（受 `max_batch_size` 和 `batch_wait_timeout` 控制），再把结果按调用方分发回去。原因：HF tokenizer 对 list 输入有内部并行(尤其是 fast tokenizer 用 Rust)，**一次 encode 多条远比一条一条 encode 快**。

```python
result = await self.async_dynamic_batch_tokenizer.encode(tokenizer_input[0], **tokenizer_kwargs)
input_ids = [result["input_ids"]]
token_type_ids = (
    [result["token_type_ids"]]
    if is_cross_encoder and result.get("token_type_ids")
    else None
)
```

返回值统一封成 batch 形态，方便下面统一处理。

#### 5.2 普通 tokenizer

```python
encoded = self.tokenizer(tokenizer_input, **tokenizer_kwargs)
input_ids = encoded["input_ids"]
token_type_ids = encoded.get("token_type_ids") if is_cross_encoder else None
```

直接调 `transformers` 的 `PreTrainedTokenizer.__call__`：

- 支持单条与批量；
- fast tokenizer 自带多线程；
- `encoded` 是个 `BatchEncoding`，类似 dict。

两种策略最后都得到 `input_ids: List[List[int]]` + 可选 `token_type_ids`。

### 6. 抽取并对齐返回形态

```python
return self._extract_tokenizer_results(
    input_ids, token_type_ids, input_format, original_batch_size
)
```

`_extract_tokenizer_results`(`tokenizer_manager.py:588`)做的事：**把统一的 batch 结果还原到调用方期望的形态**：

- 单条字符串输入 → 返回 `(List[int], Optional[List[int]])` —— 一维。
- 单 cross-encoder 对(`[[q, d]]` 长度为 1) → 同样一维。
- 真正的 batch → 返回 `(List[List[int]], Optional[List[List[int]]])` —— 二维。

这个抽取保证了上层 `_tokenize_one_request` 拿到的是「单条 → 一维」「批量 → 二维」的稳定 contract。

---

## 三、`_tokenize_one_request` 与 `_tokenize_texts` 的关系图

```
_tokenize_one_request(obj)
   │
   ├─ obj.input_embeds 优先 → 直接拿 (input_embeds, input_ids), 跳过 tokenizer
   ├─ 否则若 obj.input_ids → 直接用,跳过 tokenizer
   └─ 否则取 obj.text:
         │
         └─ await _tokenize_texts(text, is_cross_encoder)
                │
                ├─ _detect_input_format → SINGLE_STRING / BATCH_STRINGS / CROSS_ENCODER_PAIRS
                ├─ _prepare_tokenizer_input → 统一包成 list 形式
                ├─ if 单条 + 启用 dynamic batch tokenizer:
                │    await async_dynamic_batch_tokenizer.encode(...)   ← 跨请求自动合批
                │  else:
                │    self.tokenizer(...)                                ← 直接 HF tokenizer
                └─ _extract_tokenizer_results → 还原成单条/批量的形态
   │
   ├─ 多模态 mm_processor.process_mm_data_async(...) 重写 input_ids,
   │  注入 image/audio/video patch token; 可选 SGLANG_MM_PRECOMPUTE_HASH
   ├─ _validate_one_request → 长度校验/截断
   └─ _create_tokenized_object → 返回 TokenizedGenerateReqInput / TokenizedEmbeddingReqInput
```

---

## 四、设计要点 & 易混淆处

1. **三入口优先级是有意为之**：`input_embeds > input_ids > text`。一旦客户端给了上层数据，就**不再退回去** tokenize，避免 server 端「多走一步」的开销和潜在不一致。

2. **`input_embeds` 必须 `disable_radix_cache`**：RadixCache 的 key 是 token id，绕过 tokenize 用 hidden vector 做输入会让 prefix 命中失效。系统宁可显式拒绝，也不静默降级。

3. **空文本 + 多模态**也是合法的(纯语音 Whisper、纯图描述)。这条路径里 `input_ids = []` 是占位，靠 mm_processor 在后续步骤把真正的 token 序列写进去。

4. **`async_dynamic_batch_tokenizer` 只服务单条字符串**：因为 HF tokenizer 一次 batch 的开销摊到很多并发请求才划算；批量请求本身已经在一次 `tokenizer(...)` 里 batched 了，再合批就是多此一举。

5. **`_extract_tokenizer_results` 的对齐很关键**：`_tokenize_texts` 内部一律把单字符串包成 list 跑 batch tokenizer，但**返回类型必须和调用方传入的原始 shape 对应**，否则上层 `input_ids = ...` 拿到 list-of-list 会全部出错。

6. **token_type_ids 的传播**：cross-encoder → `_tokenize_texts` 返回 → `_tokenize_one_request` 直接保留 → `_create_tokenized_object` 写进 `TokenizedEmbeddingReqInput`。生成路径压根用不到这个字段。

7. **EPD 解耦的多分支**：`language_only` + `encoder_transfer_backend` 这套是 SGLang 多模态远程编码的关键。读这部分代码时需要把它放进「这个进程是 L 还是 EPD」的上下文里。

8. **长度校验依赖 `num_reserved_tokens`**：是为 sampling/特殊 token/speculative draft 留的空间。和 scheduler 端 `_validate` 不完全一致(代码里有 `# FIXME: unify the length validation logic`)，意味着边界情况可能在 scheduler 才被发现，要小心。

---

## 五、`async_dynamic_batch_tokenizer` 与 `self.tokenizer` 的关系

两条路径**底层用的是同一个 HuggingFace tokenizer 实例**，`AsyncDynamicbatchTokenizer` 只是它的一个异步包装层，并不替代或绕过 HF tokenizer。

### 1. 实例关系：是包装，不是替代

`tokenizer_manager.py:329-340`：

```python
if (
    server_args.enable_dynamic_batch_tokenizer
    and not server_args.skip_tokenizer_init
):
    self.async_dynamic_batch_tokenizer = AsyncDynamicbatchTokenizer(
        self.tokenizer,                                   # ← 把同一个 self.tokenizer 传进去
        max_batch_size=...,
        batch_wait_timeout_s=...,
    )
else:
    self.async_dynamic_batch_tokenizer = None
```

而在 `AsyncDynamicbatchTokenizer.__init__` 里：

```python
def __init__(self, tokenizer, max_batch_size=32, batch_wait_timeout_s=0.002):
    self.tokenizer = tokenizer       # ← 持有 TokenizerManager 传进来的同一个引用
    ...
```

两个引用指向**同一个 Python 对象**，不会复制一份。

### 2. 真正调 tokenizer 的地方

在 `AsyncDynamicbatchTokenizer._process_dynamic_batch`：

```python
if can_batch and len(prompts) > 1:
    encode_fn = partial(self.tokenizer, prompts, **kwargs)        # ← HF tokenizer
    results = await loop.run_in_executor(self._executor, encode_fn)
else:
    encode_fn = lambda ...: [
        self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs_list)  # ← HF tokenizer
    ]
    results = await loop.run_in_executor(self._executor, encode_fn)
```

最终的「encode 行为」就是 `self.tokenizer(...)`：

- `self.tokenizer(["hi", "hello"])` —— TokenizerManager 直接同步调 HF tokenizer。
- `await self.async_dynamic_batch_tokenizer.encode("hi")` —— 通过队列攒一批，再丢到 thread pool 里去同样调 `self.tokenizer(...)`。

### 3. `self.tokenizer` 是什么

它是 `transformers` 加载出来的 tokenizer 对象，根据模型不同可能是：

- `PreTrainedTokenizerFast`(Rust 后端，默认且推荐，速度快、支持 batch 并行)
- `PreTrainedTokenizer`(Python 后端，老模型/无 fast 实现时回退)
- 多模态模型时是 `processor`，里面也有一个 `tokenizer` 子对象

加载在 `TokenizerManager.__init__`(`tokenizer_manager.py:311-327` 一带)：

```python
self.tokenizer = get_tokenizer(...)                    # 文本模型
# 或
self.processor = get_processor(...)
self.tokenizer = get_tokenizer_from_processor(...)     # 多模态模型
```

`get_tokenizer / get_processor` 内部就是 `transformers.AutoTokenizer.from_pretrained(...)` 一类的标准入口，跟 SGLang 没有任何特殊魔法。

### 4. 两条路径到底差什么

两条路径调到底都是 HF tokenizer，但**调度策略**不同：

| 维度 | `self.tokenizer(...)` 直接调 | `async_dynamic_batch_tokenizer.encode(...)` |
|---|---|---|
| 同步还是异步 | 同步阻塞 event loop | async + 线程池，非阻塞 event loop |
| 触发线程 | 当前 asyncio 线程 | 单线程 `ThreadPoolExecutor`(`max_workers=1`) |
| 是否合批 | 调用方自己决定(传 list 就 batched) | 自动跨请求合批：在 `batch_wait_timeout_s`(默认 2ms)内最多收 `max_batch_size`(默认 32)个单字符串请求一起 encode |
| 适用场景 | 已经是 batch 的请求，或单条但调用方要立刻拿结果 | 大量并发的**单条**请求，把多次小调用合并成一次大调用 |
| 串行化 | 多协程同时调会在 GIL/Rust lock 上排队 | 用单线程 executor 显式串行，避免 fast tokenizer 的 Rust 线程竞争 |

### 5. 为什么用线程池

HF fast tokenizer 是 Rust 实现，`tokenizer(...)` 是同步阻塞调用，在主 asyncio 线程里调用会**卡住整个 event loop**(其它请求都不能调度)。`run_in_executor(self._executor, ...)` 把它丢到一个**单线程**的 thread pool：

- 主 event loop 仍然能调度其它协程；
- 单线程意味着 tokenizer 调用还是串行的，不会触发 Rust 端潜在的并发问题；
- 单条请求的合批是「跨请求合并」，不是「跨线程并行」。

### 6. 结论

是的，两边底层用的是**同一个 HuggingFace tokenizer 对象**(`self.tokenizer`，通常是 `transformers.PreTrainedTokenizerFast`)。`AsyncDynamicbatchTokenizer` 只是它的一个**异步包装层**：

- 把 `tokenizer(...)` 的同步调用挪到一个单线程 thread pool 里，避免阻塞 asyncio。
- 在毫秒级窗口内对**单条字符串请求**做跨请求合批，靠一次 batch encode 摊薄 Python ↔ Rust 的调用开销。

它不会替代或绕过 HF tokenizer，也不引入新的分词算法。
