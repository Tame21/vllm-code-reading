# 15 waiting 请求的准入与恢复

本篇对应第 **16 轮**，覆盖 **A07～A10**，于 **2026-11-07** 完成机制检验。承接上一篇的抢占与记账恢复：被抢占的请求回到等待队列，还需要再次满足准入条件，才能成为本轮安排执行的请求。

本轮核心问题是：**running 请求使用预算之后，等待队列中的新请求或恢复请求，怎样经过就绪、预算和 KV 资源检查，变成配套记录完整的本轮安排；未准入的请求又去了哪里。**

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)。采用同步、普通 Scheduler、纯文本 decoder-only 路径，处于 `UNPAUSED`，无投机、LoRA、connector、流式输入、Mamba 或 DP 延迟影响。本次 running 调度未发生抢占；`draft_slots`、额外 lookahead、前缀重放均为 0，`scheduler_reserve_full_isl = False`。分别比较 chunked prefill 开启与关闭，前缀缓存只读查询和使用契约，不深入缓存内部。这些是教学条件，不代表所有配置默认如此。

## 1. 位置、状态与阅读路线

当前仍在 `Scheduler.schedule()` 内，模型尚未执行。两个共享预算是 running 调度后的剩余额度，不是给等待请求重新初始化的一份额度。

等待段入口要求 `not preempted_reqs`，还要排除全部暂停状态。`preempted_reqs` 是本次调用新建的名单，表示本次 running 处理中是否发生抢占；它不等于 waiting 队列里是否存在历史上被抢占的请求。因此，某请求已回到 waiting，不代表同一次 `schedule()` 必须立刻再安排它；也不代表以后永远进不了等待准入段。

| 必要状态 | 本轮含义 |
| --- | --- |
| `self.waiting`、`self.kv_holding_waiting`、`request_queue` | 普通等待队列、持 KV 等待队列，以及本次实际选择的队列。先处理持 KV 队列；本轮普通路径下它通常为空。 |
| `request.status` | 请求当前状态。处于等待队列，不表示枚举必定是 `WAITING`；恢复请求可以是 `PREEMPTED`。 |
| `len(self.running)`、`max_num_active_reqs` | 已占用的活跃名额与上限。不能用本轮已入选人数代替 running 列表长度。 |
| `request.num_tokens` | 请求已有 prompt 与已生成输出 token 的总长度。抢占不会清空这些 token ID。 |
| 请求与局部的 `num_computed_tokens` | 前者是 Request 跨轮进度；后者是本次准入决定采用的可复用进度。两者不能与新执行的模型结果混为一谈。 |
| `new_computed_blocks`、`num_new_local_computed_tokens` | 本次查询得到的本地可复用块信息与命中位置数。命中不等于已经准入。 |
| `request_token_budget`、`num_new_tokens` | 当前请求可使用的预算上限，以及待安排的候选处理量；单位均为 token 位置数。 |
| `step_skipped_*`、`deferred_waiting` | 本轮暂存的跳过项，以及跨轮保留的暂缓跟踪；都不表示请求已完成。 |

主读范围如下，跨过的专用分支按本轮前提处理：

| 阶段 | 源码位置 | 因果连接 |
| --- | --- | --- |
| 入口与候选 | [954～988 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L954-L988) | 是否进入等待段、选择队列、检查名额、查看队首、就绪与暂时跳过。 |
| 复用进度 | [1009～1022 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1009-L1022)、[1099～1103 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1099-L1103)、[1140～1147 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1140-L1147) | 进度为 0 时查命中，否则沿用已有进度。 |
| 候选量与预算 | [1172～1184 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1172-L1184)、[1210～1225 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1210-L1225)、[1266～1276 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1266-L1276) | 总序列减复用进度，再应用阈值、切块开关、预算与零量边界。 |
| KV 申请 | [1278～1284 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1278-L1284)、[1310～1339 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1310-L1339) | 准备申请参数，失败时结束本次等待扫描。 |
| 成功准入 | [1371～1375 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1371-L1375)、[1406～1433 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1406-L1433) | 记录命中统计、成功出队、入 running、分类与记账、更新状态。 |
| 放回跳过项 | [1462～1466 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1462-L1466) | 本次暂存项放回相应等待队列，闭合未准入请求的去向。 |

## 2. 查看队首不是取出请求

循环先检查共享预算、队列是否为空以及活跃名额。持 KV 等待队列优先；本轮处于 `UNPAUSED` 时，没有持 KV 等待项才使用普通 waiting 队列。

第 981 行通过 `peek_request()` 查看策略队首，此时请求仍在队列中。真正成功取出是后面的第 1406 行 `pop_request()`。这个时机是理解失败后请求去向的关键。

`_handle_blocked_waiting_request()` 检查远端 KV、语法编译或后续输入等阻塞条件是否就绪。返回 True 只允许继续准入检查，不代表已经成为 `RUNNING`；普通本地请求在本轮直接就绪。内部实现不是本轮考点。

若需要暂时跳过，请求通过局部 `skip_request()` 被取出，加入 `deferred_waiting`，再按当前是否持 KV 块放入本轮的 `step_skipped_*` 暂存队列。随后 `continue`，允许检查后面的请求。

扫描结束后，第 1462～1466 行把暂存项放回适当队列。因此暂时取出不等于成功准入，更不等于丢弃请求。FCFS 路径中，暂存与放回的次序配合保留本轮跳过项的相互顺序；PRIORITY 队列仍按自身策略排序。

## 3. 恢复请求要使用总序列与可复用进度

请求的 `num_computed_tokens == 0` 时，调度器调用 `_get_local_prefix_cache_hit()` 查询本地可复用块和位置数。这是缓存查询及复用信息，不是本次刚刚运行模型得到的结果，也不是最终准入承诺。

本轮外部命中为 0，所以局部采用的计算进度来自本地命中。如果请求已经保留非零进度，则走另一分支沿用该进度，而不是无条件重新当作完全未计算的请求。

被抢占的请求也会出现进度为 0，因为上一篇已经看到抢占重置了计算进度。但它仍保留已有 prompt 和输出 token ID，身份也不是一个全新的请求。因此第 1184 行使用：

```python
num_new_tokens = request.num_tokens - num_computed_tokens
```

这里刻意使用 `request.num_tokens`，不能换成 `request.num_prompt_tokens`。否则恢复时会漏掉此前已经生成、但仍需要在重建计算状态中考虑的输出位置。

本轮还看到两个专用调用：`_mark_prefix_replay()` 记录某些缓存类型需要重算的范围，当前重放量为 0；`_reserve_prefill_lookahead()` 保护特殊投机的切块边界，当前不改变候选量。它们只作为接口背景，不表示相关实现已经走读。

## 4. 阈值、切块开关与预算的先后顺序

当前请求最多能使用：

```python
request_token_budget = min(token_budget, input_budget - draft_slots)
```

候选量从总序列减去复用进度得到，先经过有效 `long_prefill_token_threshold` 限制，再判断关闭 chunked prefill 时是否仍超过当前请求预算。

若关闭切块，且**经过阈值后的候选量**仍超过 `request_token_budget`，第 1222 行直接 `break`，结束本次 waiting 扫描。不能把后面的第 1224 行 `min()` 当成关闭切块时也无条件允许缩短处理量。

允许继续时，再把候选量限制到当前预算以内。若专用边界处理后得到零量，则走 `skip_request()` 和 `continue`，保留后面请求本轮被检查的机会；本轮纯文本、无相关特殊限制的主线不深入零量来源。

这里停止的是本次扫描。因为候选请求仍只被 `peek`，尚未成功 `pop`，队首及后面尚未检查的请求都还在等待队列中。后面请求即使所需预算更少，也不会在这次 `break` 之后继续被检查；但这不等于它们已经消失。

## 5. 缓存命中之后仍要申请 KV 资源

`allocate_slots()` 接收当前待处理量、本次命中长度和命中块，为本次安排准备资源。本轮额外预留、重放和外部计算量为 0，也不要求整段输入一次预留；仍须以实际返回结果判断资源是否满足。

缓存命中只说明有一段状态可复用，不能推出其余位置所需的块资源一定充足。成功返回新增块容器，已有容量足够时可以没有新增块；失败返回 `None`。

失败不保证整个调用完全没有资源状态变化，内部可能已释放无用块；但这不构成成功准入。第 1332～1339 行结束当前 waiting 扫描，**没有 running 段中同样的抢占重试循环**。候选仍在等待队列，尚未加入成功名单，也未扣除本次安排预算。

失败分支中还有 encoder 缓存清理，它释放的是该请求相关的缓存占用，不是清空请求的 encoder 输入。本轮纯文本主线不触发这项操作，内部没有展开。

成功路径上的 `get_blocks(request_id)` 则查询请求此刻持有的完整块信息。因此第 1425～1427 行写入 `req_to_new_blocks` 的是完整请求块信息；与 running 成功路径直接保存 `allocate_slots()` 返回的新增结果有所不同。不能只凭变量名字就把两处值都理解成同一范围。

## 6. 成功准入需要配套推进哪些状态

本轮本地同步路径中，KV 申请成功之后才走到成功出队和状态更新。若发生前缀查询，第 1371～1375 行先记录此次命中统计；此时还在 `pop` 与转 `RUNNING` 之前，不能把统计顺序说成“已经转入 running 后才统计”。

随后第 1406～1433 行按以下顺序推进：

1. 从实际候选队列 `pop_request()`，取消其暂缓跟踪，并加入 `self.running`。
2. 根据**进入时的旧状态**分类：`WAITING` 加入 `scheduled_new_reqs`，`PREEMPTED` 加入 `scheduled_resumed_reqs`。
3. 登记请求的完整块信息，以及本轮 `num_new_tokens` 安排位置数。
4. 从两个共享预算中扣除本次安排量；本轮 `draft_slots = 0`。
5. 将请求状态设为 `RUNNING`，保存本次采用的可复用计算进度，并清理相应观测阶段标记。

旧状态必须先用于区分新请求与恢复请求，再改成 `RUNNING`。等待队列位置、请求状态和本轮计划名单是三种相关但不同的记录，不能用其中一种代替另外两种。

第 1432 行为 Request 保存的仍是本次采用的复用进度，尚未加上本轮安排量。模型还没有执行；本篇也不展开之后的统一进度记账和执行指令打包。

## 7. 三种去向与 A/B 边界案例

令 FCFS 等待队列为 `[A, B]`，本轮尚有名额与预算。以下分别给定不同的分支入口条件：

| 情况 | 对 A 的操作 | 本次是否继续检查 B | 扫描后的去向 |
| --- | --- | --- | --- |
| A 暂不就绪，走 `skip_request()` | 从原队列取出并暂存 | 继续；B 仍需通过自己的检查 | A 扫描末尾放回；B 若成功则进入 running。 |
| 关闭切块，A 经过阈值后的候选仍超预算 | 只查看 A，第 1222 行停止扫描 | 不继续 | A、B 都未成功出队，仍在 waiting。 |
| A 部分命中，但 KV 申请返回 `None` | 只查看 A，第 1339 行停止扫描 | 不继续，即使给定 B 的资源足够 | A、B 都仍在 waiting，等待以后再次获得机会。 |
| A 通过各项检查，KV 分配成功 | 成功出队、入 running、登记安排并扣预算 | 继续与否取决于剩余名额和预算 | A 已准入；B 若尚未安排仍在 waiting。 |

“以后有机会”仍取决于当次调度的门控、名额、预算、就绪状态和 KV 条件，不能保证下一次一定准入。这里证明的是请求被保留下来，而不是许诺后续必然成功。

这三种去向体现不同决策：暂时绕过队首继续找候选；停止本次扫描但保留队列；资源与计算量确认后正式准入。只有成功准入路径配套改变 running 名单、安排记录、预算和请求状态。

## 8. 实际回答、纠错与通过依据

本轮依据两题主体和最后一次真实失败场景综合通过，保留以下实际发生的理解变化：

| 检验内容 | 实际证据 |
| --- | --- |
| 跳过 A 后继续 B | 学习者说明 A 进入本轮临时跳过队列，继续检查 B，扫描结束后再放回 A，已连接暂存与归还的完整路径。 |
| 恢复量使用总序列 | 学习者说明被抢占请求保留已有输出，恢复所需处理量不能只看 prompt。 |
| 命中与资源确认 | 学习者说明缓存命中不保证资源足够，成功后才取出请求、转为 `RUNNING` 并扣预算。 |
| 停止扫描与请求保留 | 关闭切块且 A 超预算的题目中，曾猜测 B 会丢失。导师用 `peek`、`break` 和成功才 `pop` 的位置解释；最后 A 部分命中但 KV 失败的场景中，学习者明确回答“不检查 B，A/B 都还在 waiting，下次有机会处理”，保留队列的纠错已经确认。 |

encoder 释放范围、成功路径各字段的准确赋值顺序由导师按源码规范；不能把这些补充伪写成学习者已经逐项独立答全。通过依据是能解释准入与未准入的因果链，不要求背诵容器清单。

本轮只读源码并手工推演，没有修改源码、安装依赖、运行模型或执行测试。前缀缓存内部、异步远端加载和模型执行器内部仍未深入；本篇只使用已核对的接口契约。

上一篇：[14 抢占与记账恢复](<14 抢占与记账恢复.md>)。本轮产物是已选请求的本轮安排，随后才由调度器组织执行指令并交给引擎；等待中的未准入请求则保留，供后续调度继续考虑。
