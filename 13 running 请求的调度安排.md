# 13 running 请求的调度安排

本篇对应第 **14 轮**，覆盖 **A01～A04**，于 **2026-10-26** 完成机制检验。承接上一篇的候选处理量计算，连通一个完整行为：**running 请求怎样在本轮预算、模型长度与 KV 资源约束下，成为本轮安排执行的请求，并让后续请求看到剩余额度。**

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)。本轮沿用 V1、离线文本生成、单候选 `n=1`、普通 Scheduler、同步客户端、单 GPU、UniProcExecutor、MRV2 与 eager 的教学路径。当前请求具备本轮调度资格；无多模态、Mamba、投机或延迟调度影响。固定 `num_output_placeholders = 0`、`draft_slots = 0`，普通生成路径的 `num_sampled_tokens_per_step = 1`。这些是本轮前提，不代表所有配置的默认行为。

## 1. 场景、范围与结果去向

现在处在 `EngineCore.step()` 调用的 `Scheduler.schedule()` 中。请求已经有 token 序列，但已有 token ID 不等于相应位置已经完成前向计算。例如 prompt 已被分词，仍可能有一段位置尚待处理。attention 计算产生的 KV 是后续复用的键和值状态；调度器要先为这些位置准备可用的块资源。

本轮主线为：

```text
已有序列与调度计算进度的差额
    → 本轮候选处理量及各项上限
    → KV 块资源申请
    → 成功登记本轮安排并扣减共享预算
    → 后续请求使用剩余额度
```

| 阅读位置 | 作用 |
| --- | --- |
| [634～644 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L634-L644) | 创建本轮名单、映射和共享预算 |
| [712～717 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L712-L717) | 进入 running 遍历，检查本轮余额 |
| [758～776 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L758-L776) | 候选量、阈值、共享预算和模型长度限制 |
| [810～841 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L810-L841) | 零量跳过；非零时申请 KV 资源 |
| [901～913 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L901-L913) | 最终失败的出口，或成功安排后的记账 |

842～900 行在本轮只按接口理解：首次分配失败后，可能经过资源检查、抢占和重试而成功，也可能最终仍失败。抢占内部及记账恢复不在本篇展开。

本段自身不执行模型。`schedule()` 后续还要考虑 waiting 请求并组织 `SchedulerOutput`。在 [EngineCore.step 650～665 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L650-L665)中，先取得调度输出，再交给 `execute_model()`，随后取得模型结果并更新请求状态。

## 2. 先区分跨轮请求状态与本轮执行计划

以下计数单位均为 token 位置数；`req_index` 是列表下标，request ID 是请求标识。

| 状态或变量 | 含义与生命周期 |
| --- | --- |
| `self.running`、`request` | Scheduler 跨轮保存的活跃请求列表，以及本次扫描到的请求对象。在 running 中不保证本轮必然入选。 |
| `request.num_tokens_with_spec` | 已有序列长度加投机 token 数。本轮无投机，因此是 prompt 与已生成 token 的总长度。 |
| `request.num_computed_tokens` | 请求跨轮保存的调度计算进度，用于定位接着处理哪些位置。不能无条件等同于 GPU 已实际完成的数量，因为后续调度代码会提前记账。 |
| `num_new_tokens` | 当前请求的局部候选处理量；逐项受限，资源满足后才写入本轮安排。 |
| `token_budget` | 本次调用共享的处理量余额，从 `self.max_num_scheduled_tokens` 初始化。 |
| `input_budget` | 本次调用共享的输入位置余额，从 `max_num_batched_tokens` 初始化；一般还需容纳每个入选请求的 `draft_slots`。本轮该预留为 0。 |
| `scheduled_running_reqs` | 本次调用新建的已选 running 请求名单，保存 Request 对象。 |
| `req_to_new_blocks` | 本次调用新建的“请求 ID → 此次成功分配返回的新增块资源信息”映射。 |
| `num_scheduled_tokens` | 本次调用新建的“请求 ID → 本轮安排处理量”映射，值不是已生成输出数量。 |

`long_prefill_token_threshold` 是上游已经算好的本轮有效阈值，0 表示不应用这一限制，不一定等于配置原值。`self.max_model_len` 来自模型最大序列长度。这里只需要使用这些前提，不为读本段再追完整初始化路径。

## 3. 从候选量到可安排的请求

### 3.1 先看整轮还有没有容量

running 循环的条件为：

```python
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    if input_budget <= draft_slots:
        break
```

两个预算都属于当前 `schedule()` 调用，不会在每条请求开始时重新初始化。`draft_slots = 0` 时，内部判断相当于检查输入余额是否已经耗尽。

### 3.2 待处理差额还要经过上限约束

本轮无投机且输出占位为 0，起始候选量可读成：

```python
num_new_tokens = (
    request.num_tokens_with_spec - request.num_computed_tokens
)
```

它表示准备从已有调度进度之后继续处理多少位置，可以全部来自 prompt。随后依次应用有效阈值、两个共享预算，以及模型长度上限。

预算约束为：

```python
num_new_tokens = min(
    num_new_tokens, token_budget, input_budget - draft_slots
)
```

这句只给当前请求的候选量赋值，**不修改 `token_budget` 或 `input_budget`**。候选量与剩余预算一般不相等；候选量为零，也不能倒推出共享预算必然为零。

模型长度进一步给出上限：

```python
self.max_model_len
- request.num_computed_tokens
- self.num_sampled_tokens_per_step
```

本轮最后一项为 1，为本步采样后追加的位置留余量。因此，预算允许处理更多位置，也不能越过这一长度限制。

### 3.3 零量只跳过当前请求

到第 810 行候选量为零时，代码推进 `req_index` 并 `continue`。这条请求本轮不能安排，并不足以证明后面的请求也不能安排。

以分支入口为已知条件：A 的候选量为零，后面的 B 候选量大于零，剩余预算足够且 KV 可分配成功。原代码允许继续尝试 B；把这里改成 `break`，会直接结束 running 遍历，使 B 失去本轮机会。这个反例只说明控制流后果，不要求展开零量来自哪种高级配置。

### 3.4 KV 申请确认资源条件

非零候选量进入 `allocate_slots()` 调用。它为请求拟计算的位置及 `num_lookahead_tokens` 指定的额外位置准备 KV 存储空间，本身不运行模型。接口说明可查 [KVCacheManager.allocate_slots 371～407 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/kv_cache_manager.py#L371-L407)。

- 成功时，更新块分配及相关管理记录，返回 `KVCacheBlocks`，内容是此次新增块资源信息。
- 已有容量够用时，成功结果可以没有新增块。因此判断是 `new_blocks is not None`，不能改成要求结果“非空”。
- `None` 表示此次无法满足分配要求，但不保证完全没有状态变化：[546～579 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/kv_cache_manager.py#L546-L579)可能先释放不再需要的块，再检查容量并返回失败。这里只了解契约，内部管理机制尚未走读。

外层的 `record_function_or_nullcontext` 为这一段加可选性能分析标记，不负责分配资源。

首次返回 `None` 时不会走成功条件下的第 841 行 `break`。它进入中间处理，可能再试，也可能结束分配循环。只有最终仍为 `None` 到达 901 行时，903 行才结束本轮 **整个 running 遍历**；这并不是 `schedule()` 立即返回。

### 3.5 成功后登记安排并即时扣减

成功落到 905～913 行时，调度器把这些状态一起推进：

```python
scheduled_running_reqs.append(request)
req_to_new_blocks[request_id] = new_blocks
num_scheduled_tokens[request_id] = num_new_tokens
token_budget -= num_new_tokens
input_budget -= num_new_tokens + draft_slots
req_index += 1
```

资源已成功准备、请求已进入本轮计划之后，才正式扣除相应额度。扣减必须在继续扫描下一条请求前完成，让下一请求的 `min()` 使用真实的剩余额度。

这里还有一个局部标记：`prefill_scheduled |= request.is_prefill_chunk`。它读取 Request 已有的非最终 prefill chunk 标记并累计；请求标记稍后才更新，且这个累计量只覆盖 running 遍历，不能概括本轮所有 prefill 请求。

## 4. 为什么局部 `min()` 必须配合即时记账

在无投机、且当前推演没有抢占撤销安排的条件下，每次继续扫描前应满足：

```text
共享剩余额度 = 本轮初始额度 - 当前已经保留的安排总量
```

本轮 `draft_slots = 0`，两个预算都按成功安排量扣减，虽然它们的初始值可以不同。后续有撤销安排时，还必须恢复相应记录与额度；具体恢复机制留到下一篇。

导师用以下反例说明即时扣减的必要性：两个预算初始都为 8，A、B 各有 6 个待处理位置，KV 充足，阈值和模型长度不构成额外限制。

| 实现 | A 安排后可见预算 | B 能安排的量 | 本轮总安排量 |
| --- | ---: | ---: | ---: |
| 原代码即时扣减 | 2 | 2 | 8 |
| 将扣减推迟到遍历结束 | 8 | 6 | 12 |

反事实版本中，每条请求都没有超过自己看到的 8，却重复使用了同一份额度。最后统一扣减，无法阻止已经形成的超额安排。KV 资源足够也不能代替本轮计算量的预算约束。

这不是算术考点，而是共享状态的一致性要求：**已经作出的安排，必须及时反映到下一次决策读取的余额中。**

## 5. 资源分配成功、调度成功与模型结果

这三个阶段不能合并理解：

| 阶段 | 本段能确认什么 |
| --- | --- |
| KV 分配成功 | 拟安排位置所需的块资源已准备，返回对应资源信息。 |
| 本轮安排登记完成 | 请求已进入本轮计划，安排量已记录，共享预算已扣减。 |
| 模型执行及结果处理 | 要等后续执行器执行、采样与请求状态更新；不能从当前安排量直接推出实际输出数量。 |

例如本轮安排处理 4 个 prompt 位置，只能说明计划继续计算这 4 个已有输入位置，不能声称已生成 4 个输出 token。即便安排的是 decode 计算，到当前源码位置也还没有发生模型前向；判断依据是调用阶段，不能只看 prompt/decode 标签。

`req_to_new_blocks` 保存的也不是“待申请的请求 ID 集合”或已经算好的 KV 数值 tensor，而是按请求 ID 保存成功分配返回的资源信息。

## 6. 实际纠错与通过依据

本轮最初把“资源不够”概括成直接中断调度，之后结合源码分清了候选量限制、零量跳过、首次分配失败与最终失败的不同后果。

| 检验点 | 实际回答与确认结果 |
| --- | --- |
| 共享预算 | 初答仍认为不会超额，没有沿用“循环内不扣预算”的反事实。经反例解释后，学习者指出 765～767 只限制候选量，911～912 才扣减；A 成功安排后必须立即扣额度，让 B 看到真实余额。预算因果链已确认。 |
| 零量与后续机会 | 学习者说明把 `continue` 改成 `break` 会退出 running 的 while，让后续请求失去本轮机会；A 不能安排不代表其他请求也不能安排。该机制已确认。 |
| 执行阶段 | 学习者明确“确实执行了”指 KV 分配成功，并非声称模型已经执行。阶段歧义已澄清。 |
| 首次失败与最终失败 | 修正后的回答说明首次 `None` 后进入中间处理、尝试重试，最终失败才到 903。导师规范最后一句：903 退出整个 running 遍历，不能只理解成跳过当前请求；此前对 while 退出范围的回答已经提供依据。 |

此前对 blocks 映射含义的误解已由导师按源码解释。本轮没有另行要求学习者独立复述这一映射，不能把“解释已经给出”伪写成一次不存在的独立作答。

本轮通过的是上述完整调度机制，不代表 KV manager、抢占内部或 `SchedulerOutput` 完整打包已经走读。核对方式为阅读源码与手工推演，没有运行模型、安装依赖或执行测试。

上一篇：[12 候选处理量与预算限制](<12 候选处理量与预算限制.md>)。下一入口是资源不足时的抢占与记账恢复：已有安排可能被撤回时，怎样让请求状态、计划记录和共享预算重新保持一致。
