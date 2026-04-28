# CUDA Graph 是什么

---

## 一 出场动机:CPU launch 开销

GPU 跑一个深度学习模型时,CPU 要不停地下发指令——每个 kernel 都要:

```
CPU:                              GPU:
┌─ for op in 模型的 N 个 op:        ┌─ kernel 0 跑
│    准备参数(stride/shape/ptr)    │
│    cudaLaunchKernel(kernel)  ──→  │  kernel 1 排队
│    cudaLaunchKernel(kernel)  ──→  │  kernel 2 排队
│    ...                            │  ...
└─ 最后 cudaStreamSynchronize        └─ 全部跑完
```

每次 `cudaLaunchKernel` 大约 **3-10 微秒** CPU 时间(参数封装 + 驱动调用 + 提交到 GPU 命令队列)。

LLM 一次 forward 包含**几千个 kernel**(几十层 × 每层多个 matmul/elementwise/attn),CPU launch 累计起来:

- 一次 prefill forward:几千个 kernel × 5 µs = **十几毫秒** CPU 时间。
- 一次 decode forward:同样几千个 kernel,但 GPU 算得快——可能 GPU 本身只要 0.5 ms,CPU launch 反而成了瓶颈。

特别是 **decode 阶段**,一次 forward 只产出 1 个 token,GPU 计算量小,CPU launch 开销甚至能占到总时间的 30-70%——这就是「**launch overhead**」问题。

---

## 二 CUDA Graph:把一段 kernel 序列「录下来」

CUDA Graph(CUDA 10 引入,2018) 让你可以:

1. **录制(capture)**:把一段 kernel 调用序列录到一个 graph 对象里,只录调用关系和参数指针,不真的执行。
2. **实例化(instantiate)**:把 graph 编译成可执行实例 `cudaGraphExec_t`,GPU 一次性预解析好整段执行计划。
3. **回放(replay)**:之后只需要一次 `cudaGraphLaunch(exec)`,GPU 就把整段序列**自己**按预解析的计划跑完——CPU 只下发了 1 次指令。

```
传统:
  CPU: launch_0 → launch_1 → ... → launch_N    (N 次系统调用)
  GPU:                                          (并行执行)

CUDA Graph 录制后:
  CPU: launch_graph_exec()                      (1 次系统调用)
  GPU: 自己按预录的序列跑完整段                  (节省 CPU launch overhead)
```

类比理解:

| 方式 | 比喻 |
|---|---|
| 普通 launch | 老板每分钟来你工位 fork 一个新任务,你做完一个,老板再来一次 |
| CUDA Graph | 老板早上一次给你打印一张「**今日全部任务清单**」,你照单做完才汇报 |

---

## 三 录制和回放代码长什么样

```cpp
// 录制
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
forward_pass_kernels(stream);   // 这里的所有 kernel 都被录下来,不真跑
cudaStreamEndCapture(stream, &graph);

// 实例化
cudaGraphInstantiate(&graphExec, graph, ...);

// 回放(可以重复多次)
for (int i = 0; i < 100; ++i) {
    cudaGraphLaunch(graphExec, stream);
}
```

PyTorch 包装成更高层 API:

```python
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    # 这里所有 GPU op 都被录到 g 里
    output = model(input_buffer)

# 回放:只要 input_buffer 内容更新一下就能再跑一次
g.replay()
```

---

## 四 适用条件(也是它的限制)

CUDA Graph 要求录制时和回放时的 op 序列**完全一致**——这是它的硬约束:

- **kernel 序列固定**:if/else 走不同路径会破坏录制。
- **Tensor 形状固定**:matmul 的 M/N/K、attention 的 seq_len 都要和录制时一样。
- **指针地址固定**(或用 graph update API 替换):录的是「读这个 GPU 地址」,所以输入要事先放在固定 buffer 里,每次 replay 前把新数据**拷贝进 buffer**。
- **不能有动态分配**:`torch.empty` 不能在录制中调用,要预分配。
- **CPU↔GPU 同步会破坏录制**:`tensor.item()` / `.cpu()` 这种会让 graph capture 失败。

所以 CUDA Graph 不是万能的——只能用在「形状固定、kernel 序列固定」的场景。

---

## 五 LLM 推理为什么是 CUDA Graph 的天选场景

把上面限制和 LLM 推理对比:

| 限制 | LLM decode 阶段 |
|---|---|
| kernel 序列固定 | ✅ 模型结构是静态的,几十层 transformer 永远是同一组 op |
| 形状固定 | 🟡 可以做到——按 batch_size 分桶 capture(详见下) |
| 指针固定 | ✅ 用预分配 buffer + 每次 replay 前拷贝新输入 |
| 无 CPU↔GPU 同步 | ✅ decode forward 全程在 GPU |

唯一的麻烦是**形状不固定**:每个 batch 的 batch_size 可能不同。SGLang / vLLM 的解决办法是**按 batch_size 分桶**:

```
预先 capture:
  batch_size = 1   → graph_1
  batch_size = 2   → graph_2
  batch_size = 4   → graph_4
  batch_size = 8   → graph_8
  batch_size = 16  → graph_16
  ...
  batch_size = 256 → graph_256

运行时:
  当前 batch_size = 7 → 找最小的 ≥ 7 的桶 → graph_8
                       → 把 7 个请求 padding 到 8(填假数据)
                       → graph_8.replay()
                       → 输出取前 7 个
```

代价:多 capture 几个 graph 占显存,decode batch 偶尔 padding 浪费一点算力。
收益:**每次 decode 节省几千个 launch**,decode latency 通常降 30%-50%。

---

## 六 SGLang 中的 CUDA Graph

SGLang 在 `srt/model_executor/cuda_graph_runner.py` 里实现:

- **decode graph**:对常用 batch_size 各 capture 一份(由 `--cuda-graph-bs` 参数指定)。
- **draft-extend graph**:speculative decoding 的 draft model 也按 size 分桶 capture。
- **target-verify graph**:spec decode 校验阶段用。
- **不 capture prefill**:prefill 的 token 总数变化太大,分桶代价不划算,prefill 是 compute-bound 也不太需要省 launch overhead。

启动时打印一行典型日志:

```
Capture cuda graph begin. This can take up to several minutes. avail mem=...
[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, ...]
```

每个数字就是一个被 capture 的 batch_size 桶。capture 完后 decode 阶段几乎所有调度路径都走 graph replay。

可以通过 `--disable-cuda-graph` 关掉(debug 时方便看 stack trace),代价是 decode 慢 30%+。

---

## 七 一句话总结

> **CUDA Graph** 是 NVIDIA 提供的「把一段固定的 kernel 调用序列录下来,后面一次 launch 就能在 GPU 上整段重放」的机制——目标是消除 CPU 一个 kernel 接一个 kernel 下发指令的开销。LLM 的 decode 阶段 kernel 序列固定 + 形状可分桶,是 CUDA Graph 的完美应用场景,SGLang/vLLM/TRT-LLM 都用它把 decode 提速 30%+,代价是要预先 capture 各 batch_size 的 graph、形状变化时要 padding。
