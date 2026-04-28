# sglang-note

SGLang 源码阅读笔记。English version: [README.en.md](README.en.md)。

## 文章目录

### TokenizerManager

- [`handle_loop()` 每行代码分析](articles/handle_loop-line-by-line.zh.md)
- [`_handle_batch_output()` 解析](articles/handle_batch_output.zh.md)
- [`_tokenize_one_request()` 与 `_tokenize_texts()` 解析](articles/tokenize-one-request.zh.md)

### Scheduler

- [Scheduler 如何从 ZMQ 拿请求并跑推理](articles/scheduler-recv-and-run.zh.md)
  - [`recv_requests()` 解析](articles/scheduler-recv-requests.zh.md)
  - [`process_input_requests()` 解析](articles/scheduler-process-input-requests.zh.md)
  - [`handle_generate_request()` 解析](articles/scheduler-handle-generate-request.zh.md)
  - [`_add_request_to_queue()` 解析](articles/scheduler-add-request-to-queue.zh.md)
  - [`get_next_batch_to_run()` 解析](articles/scheduler-get-next-batch-to-run.zh.md)
  - [`get_new_batch_prefill()` 解析](articles/scheduler-get-new-batch-prefill.zh.md)
  - [`update_running_batch()` 解析](articles/scheduler-update-running-batch.zh.md)
  - [`run_batch()` 解析](articles/scheduler-run-batch.zh.md)
  - [`process_batch_result()` 解析](articles/scheduler-process-batch-result.zh.md)
  - [`stream_output()` / `stream_output_generation()` 解析](articles/scheduler-stream-output.zh.md)
