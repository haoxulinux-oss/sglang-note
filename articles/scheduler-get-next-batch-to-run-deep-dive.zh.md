# `get_next_batch_to_run()` 深度解读

[`get_next_batch_to_run()` 解析](scheduler-get-next-batch-to-run.zh.md) 偏代码逐行,本文从**宏观视角、概念关系、生命周期**四个方向补充,帮助建立整体心智模型。

---

## 一 宏观视角:这个函数做了哪些事情

一句话:**每轮主循环它要回答一个问题——「这一帧 forward 要跑哪些请求?prefill 还是 decode?如果什么都没,返回 None」**。

它是 SGLang **continuous batching** 的核心调度器。具体来说,它把以下散乱的状态整合成「一个 ScheduleBatch」:

```
        ┌─ 上一轮 prefill 的产物(last_batch)
        │       └─ 这些请求 prefill 完了,要进 decode
        │
        ├─ 当前正在 decode 的请求(running_batch)
        │       └─ finished 的清掉,KV 不够时 retract
        │
        ├─ 等待中的新请求(waiting_queue)
        │       └─ 挑一些组成新 prefill 批
        │
        └─ 上一轮没跑完的 chunked 请求(self.chunked_req)
                └─ 继续接着 prefill 它
                                                    ↓
                            ┌──────────────────────┴─────────────────────┐
                            ▼                                            ▼
                    返回 prefill batch(优先)                     返回 decode batch
                                                  │
                                                  ▼
                                      要么返回 None(完全空闲)
```

它做的具体动作可以归纳为 5 步:

1. **timeout 治理**:扫一遍 waiting / running 队列,abort 等太久的请求(`_abort_on_waiting_timeout / _abort_on_running_timeout`)。
2. **状态过渡**:把上一轮 prefill 完的请求(`last_batch`)合并到 `running_batch`(prefill→decode 的边界)。
3. **优先组 prefill 新批**:从 `waiting_queue` 挑请求,组成 prefill batch(`get_new_batch_prefill`)。
4. **没新 prefill 就跑 decode**:走 `update_running_batch`,清 finished、retract、prepare decode tensor。
5. **DP attention / 空闲兜底**:idle batch 让没活的 rank 陪跑通信原语。

---

## 二 涉及的重要概念

| 概念 | 含义 | 哪里用到 |
|---|---|---|
| **ScheduleBatch** | 一次 forward 要跑的请求集合 + 元数据(input_ids, positions, attn metadata) | 函数返回值 |
| **forward_mode** | 这次 forward 是 prefill(EXTEND) 还是 decode(DECODE)、还是 idle | 决定走哪个 attn kernel / CUDA graph |
| **waiting_queue** | 还没开始跑的 prefill 请求列表(普通 Python list) | 新 prefill 的来源 |
| **running_batch** | 正在 decode 的请求集合(持续存在的状态对象) | decode 的主体 |
| **last_batch** | 上一轮跑过的 batch(prefill 或 decode) | prefill→decode 边界过渡用 |
| **chunked_req** | 当前正被 chunked prefill 拆分的那个请求(单个,不是列表) | 长 prompt 的分批 prefill |
| **PrefillAdder** | 「在 KV 预算内尽量塞请求」的状态机 | `get_new_batch_prefill` 内部 |
| **batch_is_full** | running_batch 的饱和度标志,True 时本轮跳过 prefill | 减少无效 prefill 尝试 |
| **continuous batching** | 任何时刻都允许新请求加入正在跑的 batch,不必等当前批跑完 | 整个调度策略的总名称 |
| **chunked prefill** | 把超长 prompt 的 prefill 拆成多个 chunk | 控制单次 prefill 占用 forward 时间 |
| **mixed chunked prefill** | 同一次 forward 同时跑 prefill + decode | `is_mixed_chunk = True` 时启用 |
| **retract** | KV 不够时把 decode 中的请求踢回 waiting_queue,让出 KV | `update_running_batch` 中处理 |
| **RadixCache prefix 命中** | 已有 KV 的 prompt 段不重复 prefill | `init_next_round_input` 计算 |
| **DP attention mlp_sync** | DP 各 rank 必须一致地跑 prefill/decode,否则 NCCL all-gather 死锁 | `maybe_prepare_mlp_sync_batch` |

---

## 三 三者关系:waiting_queue / last_batch / running_batch

这是理解这个函数的最关键概念。一图总览:

```
   client 新请求                    ┌──────── prefill ────────┐  ┌── decode 自循环 ──┐
        │                           │                         │  │                   │
        ▼                           ▼                         ▼  ▼                   │
  ┌────────────┐  ① 入队     ┌──────────────┐ ② 组 prefill batch  ┌──────────────┐    │
  │waiting_queue├────────────►│   prefill 中  ├───────────────────►│  running_batch├───┘
  │  (List)    │             │  (一帧 forward) │  ③ prefill 完进 decode│   (持久)    │
  └────────────┘             └──────────────┘                    └──────────────┘
                                  ↑                                   │
                                  │ ④ chunked 没跑完,留 self.chunked_req│
                                  └──────────────────────────────────────┘
                                  ⑤ KV 不够 → retract,放回 waiting_queue
                                  ↓
                             ┌──────────────┐
                             │waiting_queue │
                             └──────────────┘
```

### 3.1 三者各自的角色

| | 是什么 | 数据结构 | 持续时间 | 跟 prefill/decode 的关系 |
|---|---|---|---|---|
| **waiting_queue** | 等待被首次调度的请求列表 | `List[Req]`(普通 Python list) | 跨多轮主循环持续累积 | 这里的请求**还没跑过** prefill,等着被组进 prefill batch |
| **last_batch** | 上一轮主循环跑过的那批 | `ScheduleBatch` 或 None | 一轮的临时快照 | 既可能是 prefill batch(EXTEND),也可能是 decode batch(DECODE) |
| **running_batch** | 当前正在 decode 的请求集合 | `ScheduleBatch` | 跨多轮主循环持续存在(decode 的本质就是反复对它跑 forward) | 里面的请求**已经 prefill 完了**,在反复 decode |

### 3.2 一个请求的一生

```
client 来一个新请求
    │
    ▼
process_input_requests → handle_generate_request → _add_request_to_queue
    │
    ▼
   入 waiting_queue 等着                     [ 阶段:waiting ]
    │
    ▼
get_next_batch_to_run → get_new_batch_prefill
    │   PrefillAdder 选中它
    │   waiting_queue 里把它移除
    │   组成 prefill ScheduleBatch
    ▼
本轮 batch (forward_mode=EXTEND, prefill) ←─ 这一轮的 last_batch 就是它
    │
    ▼
run_batch + process_batch_result_prefill
    │   prefill 完成,KV 写入,生成第一个 token
    ▼
下一轮 get_next_batch_to_run
    │   last_batch.is_extend() → 把它 merge 进 running_batch  [ 阶段:进入 decode ]
    │
    ▼
running_batch 里反复 decode                 [ 阶段:running ]
    │   每轮 update_running_batch
    │   filter finished、retract、prepare_for_decode
    │   每轮 run_batch 跑一次 forward,生成 1 个 token
    ▼
EOS / max_new_tokens 触发
    │
    ▼
process_batch_result_decode 把它标 finished  [ 阶段:finished ]
    │
    ▼
update_running_batch.filter_batch 把它从 running_batch 移除
KV 释放回 token_to_kv_pool
stream_output 推回 DetokenizerManager
```

### 3.3 三者之间的「转移」时机

发生在 `get_next_batch_to_run` 这个函数里的就两种过渡:

#### 过渡 A:waiting_queue → 新 prefill batch(本轮 ret)

由 `get_new_batch_prefill` 完成:

```python
# 简化
for req in self.waiting_queue:
    if adder.add_one_req(req): can_run_list.append(req)
self.waiting_queue = [x for x in self.waiting_queue if x not in can_run_list]
new_batch = ScheduleBatch.init_new(can_run_list, ...)
return new_batch
```

#### 过渡 B:last_batch(若是 prefill) → running_batch

```python
if self.last_batch and self.last_batch.forward_mode.is_extend():
    self.last_batch.filter_batch(chunked_req_to_exclude=...)   # 排除 chunked 请求
    if not self.last_batch.is_empty():
        if self.running_batch.is_empty():
            self.running_batch = self.last_batch              # 整体接管
        else:
            self.running_batch.merge_batch(self.last_batch)   # 合并
```

注意 `last_batch.is_decode()` 时**不需要 merge**——它的请求本来就在 running_batch 里跑,只是上一轮做了一次 decode forward 而已。

### 3.4 为什么需要 last_batch 这个中间态

为什么不直接「prefill 完立刻 merge 进 running_batch」?

因为 SGLang 的主循环结构是:

```
recv → process_input → get_next_batch → run_batch → process_batch_result
                          ↑                                      │
                          └──────────────────────────────────────┘
                          这里要把上一轮的 batch(last_batch) 接续好
```

`run_batch` 跑完只是生成了第一个 token,**process_batch_result 后请求才算「prefill 完成」**。但 process_batch_result 之后这一轮就结束了,要等下一轮主循环再开始时,才有机会把它合并进 running_batch。

`last_batch` 就是「上一轮 batch」的引用,作为这一过渡的载体。

### 3.5 三者的「请求所有权」

任意时刻一个请求**只能在三者之一**里:

- 在 waiting_queue:还没 prefill 过。
- 在 running_batch.reqs:在 decode。
- 是 self.chunked_req:还在 chunked prefill 中(单个请求)。

不会同时在两处——`get_new_batch_prefill` 把请求从 waiting_queue 物理移除时就是把所有权移交给 batch。

---

## 四 chunked_req 的完整生命周期

### 4.1 为什么要有 chunked prefill

考虑一个 32k token 的长 prompt:如果一次 prefill 全跑,这一次 forward 大约要 1-3 秒。在这 3 秒里其他 decode 中的请求**全部停滞**——每个请求 ITL(inter-token latency) 抖动到 3 秒,SLO 基本完蛋。

解决:把这条 prefill 拆成 N 个 chunk,每个 chunk 比如 2k token。这样每次 forward 大约只跑 2k 的 prefill + 其他 decode,两边都能动。

实现方式:同一个请求在 N 轮 forward 里被反复扔进 prefill,每轮跑一段;`self.chunked_req` 就是「**当前正在被拆的那一个请求**」的引用。

### 4.2 设计:为什么是「单个 self.chunked_req」而不是列表

SGLang 限制**一次只能有一个 chunked 请求**——这是个简化设计。理由:

- 一次只 chunk 一个,代码逻辑简单(不用维护 chunked 优先级队列)。
- chunk size(`chunked_prefill_size`) 通常是 8k token 量级,单个请求几个 chunk 就跑完了,不需要并发多个。
- 多个 chunked 请求并发反而抢 KV,容易触发 retract。

### 4.3 完整生命周期(状态机)

下面用「Q1:32k token 长 prompt」举例,假设 `chunked_prefill_size = 8k`。

```
状态 0:Q1 是新请求
        │
        ▼
process_input_requests → 入 waiting_queue
self.chunked_req = None
self.waiting_queue = [Q1]
running_batch = []
last_batch = None


轮 N:get_next_batch_to_run → get_new_batch_prefill
  │
  │ adder 决定塞 Q1 进 batch
  │ Q1 一共 32k token,但 chunked_prefill_size=8k,只能跑前 8k
  │ adder.add_one_req 检测到 token 过多,触发 chunking 逻辑
  │   - Q1.extend_input_len = 8000
  │   - Q1.is_chunked = 1
  │   - adder.new_chunked_req = Q1                ← 标记为新 chunked 请求
  │
  │ 函数末尾:
  │ self.chunked_req = adder.new_chunked_req      ← 记到 scheduler 状态
  │ self.waiting_queue = []                        ← Q1 从 waiting 移除
  │
  ▼
run_batch:跑 8k token 的 prefill forward(forward_mode=EXTEND)
  │   Q1 的 prefix_indices 现在指向已 prefill 的 0..8k token 的 KV
  │
  ▼
process_batch_result_prefill:
  │   注意:chunked req 的中间轮次,通常 NOT sample(还没 prefill 完)
  │   只更新 prefix_indices, output 留空
  ▼
状态:
  self.chunked_req = Q1
  self.waiting_queue = []
  running_batch = []
  last_batch = (含 Q1 的 prefill batch)


轮 N+1:get_next_batch_to_run
  │
  │ ① 顶部:把 self.chunked_req 加入 chunked_req_to_exclude
  │     for stash:把 Q1 暂时「藏起来」不让它进 running_batch
  │
  │ ② merge last_batch 到 running_batch
  │     last_batch 经过 filter_batch(排除 Q1) 后变空
  │     running_batch 还是空
  │
  │ ③ 调 get_new_batch_prefill
  │     adder.add_chunked_req(self.chunked_req) → 优先把 Q1 塞回来
  │     Q1.extend_input_len = 8000(下一段 8k)
  │     Q1.is_chunked = 2
  │     can_run_list = [Q1, ...(还可能塞别的新请求)]
  │     ScheduleBatch.init_new(can_run_list, chunked_req=Q1)
  │
  ▼
run_batch:跑 第二段 8k 的 prefill,Q1 的 prefix_indices 现在是 0..16k
  ...

轮 N+2:同上,跑第三段 8k → prefix_indices = 0..24k

轮 N+3:get_new_batch_prefill
  │ adder.add_chunked_req(Q1) 跑最后 8k(0..32k)
  │ adder 检测到这是最后一段(剩余长度 < chunked_prefill_size)
  │   - Q1.is_chunked = 4(累计被拆 4 次)
  │   - adder.new_chunked_req = None              ← 不再设新 chunked
  │
  │ 函数末尾:
  │ self.chunked_req = None                       ← 清空
  │
  ▼
run_batch:跑最后 8k token + sample,生成 Q1 的第一个生成 token
  │
  ▼
process_batch_result_prefill:
  │   chunked req 最后一段:正常 sample,Q1 prefill 完成
  │   Q1 进入「decode」阶段
  ▼
状态:
  self.chunked_req = None
  last_batch = (含 Q1 的最后 prefill batch,Q1 已 sample 完第一个 token)


轮 N+4:get_next_batch_to_run
  │ last_batch.is_extend() → True
  │ filter_batch:Q1 已经不是 chunked 了(self.chunked_req = None),不会被排除
  │ running_batch.merge_batch(last_batch)         ← Q1 进 running_batch
  │
  ▼
后续:Q1 进入 decode,反复跑 forward,直到 EOS / max_new_tokens
```

### 4.4 几个关键细节

#### 4.4.1 chunked_req 跨轮的「进出 batch」

每一轮 `get_next_batch_to_run` 开头的两段:

```python
# 顶部:把 chunked_req 从 last_batch 排除
if self.chunked_req is not None:
    chunked_req_to_exclude.add(self.chunked_req)
    self.stash_chunked_request(self.chunked_req)

...
self.last_batch.filter_batch(chunked_req_to_exclude=list(chunked_req_to_exclude))

# get_new_batch_prefill 内:
if self.chunked_req is not None:
    self.chunked_req.init_next_round_input()
    self.chunked_req = adder.add_chunked_req(self.chunked_req)   # 把它接回来
```

所以 chunked_req 的状态是**「跨轮一直挂在 self.chunked_req 字段上,每轮在 get_new_batch_prefill 里被 adder 接回新 batch」**——它在 last_batch 里短暂存在,在 running_batch 里**从不存在**。

#### 4.4.2 chunked_req 不进 running_batch 的原因

running_batch 是「正在 decode 的请求」。chunked_req 还在 prefill 中(只是被拆了),没 prefill 完不能 decode——所以必须**单独存 self.chunked_req 字段**,不能跟 running_batch 混。

#### 4.4.3 chunked_req 的优先级最高

`adder.add_chunked_req` 是在 `add_one_req` 之前调的——保证 chunked 请求**永远第一个被塞进新 batch**。原因:它已经占了 KV slot(prefix_indices 持续指向 GPU 上的 KV),如果不能继续 prefill,这部分 KV 就闲置浪费。

#### 4.4.4 chunked_req 不能并发的硬约束

```python
# get_new_batch_prefill 里:
if adder.new_chunked_req is not None:
    assert self.chunked_req is None    # ← 同时只能有一个
    self.chunked_req = adder.new_chunked_req
```

这个 assert 保证全局只有一个 chunked。如果当前还有 chunked_req,adder 不会再产生新的 chunked req(具体在 `PrefillAdder.add_one_req` 里看 `has_chunked_req` 标志)。

#### 4.4.5 chunked req 中间轮次不 sample

`process_batch_result_prefill` 内:

```python
if req.is_chunked > 0 and 还没 prefill 完:
    # 不 sample,不 append output,只更新 prefix_indices
    continue
```

只有最后一段(prefill 完整个 prompt) 才 sample——因为 next_token 必须基于完整的 prompt logits。

#### 4.4.6 chunked req 完成后才进 running_batch

```python
轮 N+3 末尾:Q1 prefill 完成,sample 完
轮 N+4 开头:last_batch.is_extend() = True
            self.chunked_req = None(已清),所以 Q1 不在 chunked_req_to_exclude 里
            filter_batch 不会排除 Q1
            Q1 跟着 last_batch 一起 merge 进 running_batch
            进入 decode
```

注意这一步:**Q1 进入 running_batch 的时机是「prefill 完成的下一轮」**,不是 prefill 完那一刻——这是「last_batch」中介结构的体现。

### 4.5 chunked_req 与 PP/PD 的额外耦合

PP > 1 时,chunked req 可能在某个 microbatch 里 prefill 一半,另一个 microbatch 接着——这时:

```python
if self.last_batch.chunked_req is not None:
    chunked_req_to_exclude.add(self.last_batch.chunked_req)
```

`last_batch.chunked_req`(batch 级) 和 `self.chunked_req`(scheduler 级) 是两个不同字段,PP 下都要排除。详见 [`self.chunked_req` vs `self.last_batch.chunked_req`](scheduler-chunked-req-vs-last-batch-chunked-req.zh.md)。

---

## 五 一句话总结

> **`get_next_batch_to_run` 是 SGLang 调度器的「指挥棒」**——每轮主循环它根据 `waiting_queue / running_batch / last_batch / chunked_req` 四个状态源,决定本轮 forward 的 batch:**优先组 prefill 新批,没就跑 decode,都没就 idle**。
>
> **三者关系**:waiting_queue 是「没跑过的新请求」,running_batch 是「正在 decode 的持久集合」,last_batch 是「上一轮快照」——prefill 完的请求通过 `last_batch (EXTEND) → merge 进 running_batch` 这条边过渡到 decode 阶段。
>
> **chunked_req 生命周期**:某请求 prompt 太长被 PrefillAdder 标记为 chunked 后,它从 waiting_queue 移出但**不进 running_batch**,而是挂在 `self.chunked_req` 字段上,跨轮反复被 adder.add_chunked_req 拉回新 prefill batch,每轮跑一段;最后一段 prefill 完成 + sample 后,chunked_req 清空,请求跟着 last_batch 正常 merge 进 running_batch 开始 decode。同一时刻全局只允许一个 chunked_req,优先级最高。
