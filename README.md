# sglang-note

SGLang 源码阅读笔记。English version: [README.en.md](README.en.md)。

## 文章目录

### 基础知识

- [Prefill 和 Decode](articles/basics-prefill-decode.zh.md)
- [`forward` 在 LLM 领域的含义](articles/basics-forward.zh.md)
- [`forward_mode` 是什么](articles/basics-forward-mode.zh.md)
- [CUDA Graph 是什么](articles/basics-cuda-graph.zh.md)
- [CUDA 是什么](articles/basics-cuda.zh.md)

### TokenizerManager

- [`handle_loop()` 每行代码分析](articles/handle_loop-line-by-line.zh.md)
- [`_handle_batch_output()` 解析](articles/handle_batch_output.zh.md)
- [`_tokenize_one_request()` 与 `_tokenize_texts()` 解析](articles/tokenize-one-request.zh.md)

### KV Cache

- [KV cache 存什么、HBM 是什么、`_prefetch_kvcache()` 在做什么](articles/kvcache-prefetch-and-storage.zh.md)

### Scheduler

#### Rank

- [Scheduler / Worker 进程拓扑与各种 rank](articles/scheduler-rank-and-process-topology.zh.md)

#### 主循环

> 🗺 **[函数调用关系图(可点击节点跳转)→](articles/scheduler-call-graph.zh.md)** — 用 Mermaid 图把下面这些文章按调用顺序串起来,适合先看再读单篇。

- [Scheduler 如何从 ZMQ 拿请求并跑推理](articles/scheduler-recv-and-run.zh.md)
  - [`recv_requests()` 解析](articles/scheduler-recv-requests.zh.md)
  - [`process_input_requests()` 解析](articles/scheduler-process-input-requests.zh.md)
  - [`handle_generate_request()` 解析](articles/scheduler-handle-generate-request.zh.md)
  - [`_add_request_to_queue()` 解析](articles/scheduler-add-request-to-queue.zh.md)
  - [`get_next_batch_to_run()` 解析](articles/scheduler-get-next-batch-to-run.zh.md)
    - [`get_next_batch_to_run()` 深度解读(宏观 / 概念 / 三状态关系 / chunked_req 生命周期)](articles/scheduler-get-next-batch-to-run-deep-dive.zh.md)
    - [`self.chunked_req` vs `self.last_batch.chunked_req`(PP 下两者会分歧)](articles/scheduler-chunked-req-vs-last-batch-chunked-req.zh.md)
  - [`get_new_batch_prefill()` 解析](articles/scheduler-get-new-batch-prefill.zh.md)
  - [`update_running_batch()` 解析](articles/scheduler-update-running-batch.zh.md)
  - [`run_batch()` 解析](articles/scheduler-run-batch.zh.md)
    - [`forward_batch_generation()` 解析(初学者向)](articles/forward-batch-generation.zh.md)
      - [`ModelRunner` 是什么(成员介绍)](articles/model-runner-overview.zh.md)
      - [`ModelRunner.forward()` 方法详解(初学者向)](articles/model-runner-forward.zh.md)
  - [`process_batch_result()` 解析](articles/scheduler-process-batch-result.zh.md)
  - [`stream_output()` / `stream_output_generation()` 解析](articles/scheduler-stream-output.zh.md)
