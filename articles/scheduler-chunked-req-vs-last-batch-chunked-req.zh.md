# `self.chunked_req` vs `self.last_batch.chunked_req`

两者都指向「正在被 chunked prefill 拆分的请求」,但是**作用域和时间维度不同**。在单流水线下它们通常是同一个,在 PP(pipeline parallel) 模式下会出现分歧。

---

## 一 简短对照

| | `self.chunked_req` | `self.last_batch.chunked_req` |
|---|---|---|
| 字段所属对象 | `Scheduler`(全局) | `ScheduleBatch`(单个 batch) |
| 含义 | **当前**整个 scheduler 跨轮跟踪的那个 chunked 请求 | **上一轮跑过的那个 batch 当时**关联的 chunked 请求 |
| 时间维度 | 「现在的状态」 | 「上一帧的快照」 |
| 何时被设置 | `get_new_batch_prefill` 末尾:`self.chunked_req = adder.new_chunked_req`(或保留旧值) | `ScheduleBatch.init_new(can_run_list, ..., chunked_req=self.chunked_req)`——构造 batch 时从 scheduler 复制一份引用 |
| 何时被清空 | 当 chunked req 完成最后一段 prefill,adder 不再 new_chunked,`self.chunked_req = None` | 与 last_batch 一起,被本轮新 batch 替换时整体丢弃 |
| 是否唯一 | 是,**全局只能有一个** | 每个 batch 各自有一个字段(可以为 None) |

---

## 二 设置时机

### 2.1 `self.chunked_req` 的赋值

在 `_get_new_batch_prefill_raw` 末尾(`scheduler.py:2598-2604`):

```python
if adder.new_chunked_req is not None:
    assert self.chunked_req is None
    self.chunked_req = adder.new_chunked_req

if self.chunked_req is not None:
    self.chunked_req.is_chunked += 1
```

只有 PrefillAdder 在本批生成了新的 chunked req 时,才把它挂到 scheduler 上。如果上一轮已有 chunked req(还没跑完),adder 会跳过新建,继续用 `self.chunked_req`。

### 2.2 `last_batch.chunked_req` 的赋值

在 `ScheduleBatch.init_new`(`get_new_batch_prefill` 调用) 时:

```python
new_batch = ScheduleBatch.init_new(
    can_run_list,
    self.req_to_token_pool,
    self.token_to_kv_pool_allocator,
    self.tree_cache,
    self.model_config,
    self.enable_overlap,
    self.spec_algorithm,
    chunked_req=self.chunked_req,    # ← 把 scheduler 当前的引用复制给 batch
)
```

**构造瞬间**:`new_batch.chunked_req == self.chunked_req`(同一个对象的引用)。

之后这个 new_batch 会被 `run_batch` 跑掉,然后 `self.last_batch = batch`。所以「**`last_batch.chunked_req` 就是上一轮组 batch 时,scheduler 当时的 `self.chunked_req` 的快照**」。

---

## 三 单流水线下:两者通常一致

走完一轮:

```
轮 N 开头  : self.chunked_req = Q1
轮 N 内    : new_batch = init_new(..., chunked_req=Q1)
            → run_batch(new_batch)
            → self.last_batch = new_batch    # last_batch.chunked_req == Q1
            → self.chunked_req 在 get_new_batch_prefill 末尾继续保持 Q1(还没跑完)
                                       或 None(如果 Q1 跑完最后一段)

轮 N+1 开头: 此时
            self.chunked_req = Q1(或 None)
            self.last_batch.chunked_req = Q1
```

**两个字段值相等**——单流水线下 `self.chunked_req` 是 `last_batch.chunked_req` 的「**保留版本**」,只要 Q1 没跑完最后一段,scheduler 字段会持续保留它。

代码里两处分别排除:

```python
if self.chunked_req is not None:
    chunked_req_to_exclude.add(self.chunked_req)
    self.stash_chunked_request(self.chunked_req)
...
if self.last_batch.chunked_req is not None:
    chunked_req_to_exclude.add(self.last_batch.chunked_req)
```

是因为 set.add 同一个对象幂等,即使两者相等也不会出错;但 PP 下两者可能不等,所以**两处都不能省**。

---

## 四 PP(pipeline parallel) 下:两者会分歧

注释里专门提到这一点(`scheduler.py:2340-2343`):

```python
if self.last_batch.chunked_req is not None:
    # In the context pipeline parallelism, after the last chunk, the current microbatch still track outdated chunked_req.
    # We need to discard it.
    chunked_req_to_exclude.add(self.last_batch.chunked_req)
```

PP 流水线下,**多个 microbatch 同时在不同 stage 跑**——可能出现:

- microbatch A:Q1 的最后一段 chunk 已跑完,`self.chunked_req` 已被清成 `None`。
- microbatch B(也就是这一轮的 last_batch):它构造时 `last_batch.chunked_req = Q1`(过时的引用)。

也就是说:

```
self.chunked_req         = None    (scheduler 已经知道 Q1 跑完最后一段,清空了)
self.last_batch.chunked_req = Q1   (但 last_batch 是更早构造的,还存着 Q1 的旧引用)
```

这种情况下 last_batch 里实际上**还含有 Q1 的最后一段 prefill 结果**——按理说 Q1 已经 prefill 完整段 prompt,应该正常 merge 进 running_batch。但因为 PP 的乱序,我们需要保守地把这种「**陈旧 chunked_req 引用对应的请求**」也排除掉,等下一轮再处理(避免在 last_batch 还没完全清算时就把它当 decode 请求处理)。

注释的核心意思:**「上一个 microbatch 还在用过时的 chunked_req 标记,需要把它丢弃」**。

---

## 五 形象类比

把两个字段类比成「现实状态 vs 信纸快照」:

- `self.chunked_req` = scheduler **现在脑子里**记着「我正在拆分 Q1」。
- `last_batch.chunked_req` = 上一轮发出去的「**便条**」,上面写着「这一帧涉及 Q1 的某段 chunk」。

单流水线下,scheduler 同步更新自己的脑子和发出便条,两者一致。

PP 下,便条已经发出去了,但 scheduler 因为收到了别的 microbatch 的反馈,**脑子里的状态先一步更新了**——这时便条上的内容就「过期」,但物理上 last_batch 里还引用着它,所以也要排除一下。

---

## 六 一句话总结

> **`self.chunked_req`** 是 Scheduler 的全局字段,**「当前跨轮跟踪的那个 chunked 请求」**;**`self.last_batch.chunked_req`** 是 ScheduleBatch 的字段,**「上一轮 batch 构造时从 scheduler 复制下来的快照」**。
>
> 单流水线下两者通常指向同一对象,代码同时排除是为了 set 幂等地多保一道险;**PP 流水线下因为多 microbatch 异步推进,两者会分歧**——scheduler 已清空但 last_batch 还有过期引用,所以两个字段必须分别检查、分别排除,确保「上一轮 batch 里所有跟 chunked 沾边的请求」都不会在这一轮误进 running_batch。
