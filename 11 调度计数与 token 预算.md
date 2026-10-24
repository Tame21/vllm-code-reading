# 11 调度计数与 token 预算

本篇对应第十二轮阅读，承接上一轮的请求 token 累积与长度限制，进入 `Scheduler.schedule()` 的开头。重点是：**请求当前已有的序列长度、调度器记录的计算进度、本轮可安排的处理量，是三个不同的量。** 本轮已完成阅读与纠错检验；目录从 `00` 开始编号，因此第十二轮的文件编号为 `11`。

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)，文中行号对应这一提交。继续按 V1、离线文本生成、单候选 `n=1`、非流式输入阅读。同步多进程客户端不等同于关闭异步调度；下文的计数示例另外限定为没有推测 token、没有异步占位的普通逐轮生成，并在每轮执行和结果更新完成后观察。

## 1. 阅读位置与范围

| 用途 | 源码位置 | 阅读重点 |
| --- | --- | --- |
| 主读 | [Scheduler.schedule 第 621–641 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L621-L641) | 算法注释、本轮容器初始化与处理预算 |
| 辅助 | [Request 的三个长度属性，第 294–304 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/request.py#L294-L304) | 区分已有序列、推测 token 与累计输出 |
| 答疑核对 | [_update_after_schedule 第 1696–1709 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1696-L1709) | 计算进度增加的数量与时机 |

本轮先理解调度开头的计数关系，不展开完整调度循环、KV cache 分配、抢占和等待请求的准入逻辑。投机解码只记录问答中澄清的概念边界。

## 2. 当前序列长度与计算进度

上一轮读过：

```python
num_tokens = len(self._all_token_ids)
num_output_tokens = len(self._output_token_ids)
```

本轮再看一个属性：

```python
@property
def num_tokens_with_spec(self) -> int:
    return len(self._all_token_ids) + len(self.spec_token_ids)
```

在当前普通文本场景中，各个量的含义是：

| 名称 | 含义 |
| --- | --- |
| `num_tokens` | prompt token 数加上累计生成 token 数 |
| `num_output_tokens` | 累计生成 token 数 |
| `num_tokens_with_spec` | `num_tokens` 再加上当前保存的推测 token 数 |
| `num_computed_tokens` | 调度器维护的计算进度，按 token 位置计 |

本轮基本例子中 `spec_token_ids = []`，因此 `num_tokens_with_spec == num_tokens`。

`schedule()` 的注释说明：每一步尝试给请求分配 token 处理量，让 `num_computed_tokens` 追上 `num_tokens_with_spec`。这里的“追上”是计算进度与当前序列长度之间的关系，不是让累计生成数量达到某个值，也不表示一次就能处理完全部差额。

### 计算进度的记账时机

答疑时辅助核对了下面这行：

```python
request.num_computed_tokens += num_scheduled_token
```

它位于 `_update_after_schedule()` 中。[schedule 第 1634–1636 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1634-L1636) 在返回调度安排前调用这个函数；随后外层才执行模型。

因此，**本轮调度安排确定后、模型执行前，就会先推进这个计数**。它不能随时被当成 GPU 已经实际完成的 token 数，也不是历史上累计执行过的 token 总次数。后续还可能因拒绝草稿等情况修正计数。本轮用执行和结果更新完成后的简单状态解释其数值。

## 3. 为什么累计生成 2 个，计算进度却是 5

设 prompt 为 `[P1, P2, P3, P4]`，没有前缀缓存命中，也没有推测 token 和异步占位。模型每轮采样一枚新 token，请求尚未结束。

| 观察时刻 | 本轮作为输入处理的位置 | 本轮新输出 | `num_computed_tokens` | 累计输出数 | 当前总长度 |
| --- | --- | --- | --- | --- | --- |
| 初始 | 尚未处理 | 无 | 0 | 0 | 4 |
| 第一轮执行与更新后 | `P1、P2、P3、P4` | `O1` | 4 | 1 | 5 |
| 第二轮执行与更新后 | `O1` | `O2` | 5 | 2 | 6 |

第一轮处理 prompt 后得到 `O1`；第二轮把 `O1` 作为后续输入，得到 `O2`。前面位置的 KV 计算结果可以继续使用，第二轮处理的是新增输入位置。

第二轮结束时，可以把两组 token 并排写出来：

```text
输入计算覆盖的位置：P1、P2、P3、P4、O1  → 5 个
累计生成的输出：    O1、O2              → 2 个
当前完整序列：      P1、P2、P3、P4、O1、O2 → 6 个
```

**生成出 `O2`，发生在把 `O1` 作为输入计算之后。`O2` 已经存在于请求里，但还没有作为下一轮输入处理。** 所以，计算进度为 `5` 和累计生成 `2` 枚 token 可以同时成立。

这里的 `5` 可以理解为这个简单例子中 prompt 与后续输入位置的计算进度，不能解释成生成了 `5` 枚 token。也不要把例子中的数值关系推广为所有配置都适用的计数公式。

## 4. “尚未处理几个位置”的含义

在上述没有推测 token、没有异步占位，且一轮执行与结果更新已完成的观察时刻，可以用下面的差值理解已有序列中尚待安排计算的输入位置：

```text
待处理位置数 = num_tokens - num_computed_tokens
```

第二轮结束后的例子是 `6 - 5 = 1`，对应最后新产生的 `O2`。下一轮把它作为输入处理，才能继续得到后续 token。

不同状态下，这个差值的表现不同：

- **Prefill 尚未完成、没有生成输出时**：当前序列只有 prompt，差值就是 prompt token 数减去计算进度。例如新请求有 10 枚 prompt token，本轮只处理前 4 个位置，剩余是 `10 - 4 = 6`。
- **Prefill 完成并采样出第一枚输出后**：完整序列多了一枚新输出，它还没有作为后续输入处理，因此这个简单例子中差值变成 `1`。
- **普通逐 token decode，且请求继续生成时**：在本轮执行与结果更新完成后，通常也是最后新产生的那一枚尚未作为后续输入处理，差值为 `1`。

这里用 prefill 和 decode 两类状态帮助理解；源码注释通过统一的计算进度关系描述调度。

### 为什么不能直接按请求数量计算

如果本轮所有请求都属于上述继续生成的普通 decode 情况，每个请求各留一枚新 token，那么 N 个这样的请求，待处理位置总数恰好为 N。

但这取决于请求的状态。上面的 prompt 例子中，本轮只处理了 **1 个请求**，它却还剩 **6 个 token 位置**。因此不能把待处理位置数直接等同于本轮请求数，也不能把这个差值无条件用于正在调度或执行中的状态。

## 5. 本轮处理预算与调度记录

`schedule()` 开头创建本轮容器，其中包括：

```python
num_scheduled_tokens: dict[str, int] = {}
token_budget = self.max_num_scheduled_tokens
```

`num_scheduled_tokens` 用于记录每个请求本轮被安排的 token 处理量，形式为 `req_id -> 数量`。刚初始化时它为空，只说明本轮的安排尚未写入这个字典，不能据此认定调度器里没有请求。

`token_budget` 是本轮各请求共享的 token 处理预算。若 `self.max_num_scheduled_tokens = 8`，执行完第 641 行时：

```python
token_budget = 8
num_scheduled_tokens = {}
```

这表示本轮所有请求共用这份处理额度，而不是每个请求各自有 8。这里的额度也不能解释成“本轮一定生成 8 枚新 token”：安排的计算可能包括 prompt 位置，实际安排还受后续预算和其他条件限制。

## 6. 投机解码答疑：采纳数不等于尚未处理数

讨论计算进度时，曾进一步提出：投机解码中，尚未处理的位置数是否等于本轮采纳的 token 数？这个推断不成立，因为**被目标模型采纳的草稿 token，已经在验证时作为输入参与过计算**。

以没有异步占位、两个草稿都通过、且没有触发停止的一次普通验证为例：

```text
本轮目标模型的输入：  O1、D1、D2
验证、采样后新增输出：D1、D2、B
```

`D1、D2` 是通过验证的草稿，已经参与目标模型计算；`B` 是最后新采样出来的补充 token，尚未作为下一轮输入处理。因此本例采纳了 2 枚草稿，尚未作为后续输入处理的却只有 `B`，共 1 枚。

这与两处答疑时核对的源码一致：

- [调度后推进计数，第 1706–1709 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L1706-L1709)：先把本轮安排的处理量计入计算进度。
- [处理验证结果，第 2196–2212 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L2196-L2212)：根据采纳数求出拒绝的草稿数，再回退被拒绝的部分，采纳部分仍保留在计算进度中。

[RejectionSampler 的术语说明，第 54–69 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/sample/rejection_sampler.py#L54-L69) 说明了采纳、替换与 bonus 输出；草稿全部通过时，会在末尾加入 bonus token。本轮只用这个例子澄清输入与输出的区别，不把“总是只剩 1 个”推广到异步占位、多个在途步骤或其他生成机制。

## 7. 自查题与参考答案

**问题一：请求有 4 枚 prompt token，累计生成 2 枚 token，没有推测 token，当前 `num_computed_tokens = 5`。`num_tokens_with_spec` 是多少，计算进度还差几个位置？这里的 `5` 能解释成累计生成数吗？**

<details>
<summary>参考答案</summary>

`num_tokens_with_spec = 4 + 2 = 6`，与计算进度相差 `6 - 5 = 1` 个位置。在上述普通逐轮生成例子中，计算进度覆盖 `P1、P2、P3、P4、O1`，尚未作为后续输入处理的是 `O2`。累计生成数是 `2`，不是 `5`。计数在调度时可能提前推进，因此也不能随时把它当作 GPU 实际完成量。

</details>

**问题二：`self.max_num_scheduled_tokens = 8`，刚执行完第 641 行，`token_budget` 和 `num_scheduled_tokens` 分别是什么？预算属于每个请求还是本轮所有请求？**

<details>
<summary>参考答案</summary>

`token_budget = 8`，`num_scheduled_tokens = {}`。预算由本轮所有请求共享，限制的是安排的 token 处理量，不是每个请求各生成 8 枚 token。

</details>

上一轮：[10 请求 token 累积与长度限制](<10 请求 token 累积与长度限制.md>)。本轮已检验通过，下一轮源码尚未布置。
