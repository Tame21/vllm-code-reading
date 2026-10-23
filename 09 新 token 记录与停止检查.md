# 09 新 token 记录与停止检查

上一轮读到 `scheduler.update_from_output()`，明确它既要组织内部输出，也要更新核心侧保存的请求状态。本轮沿着其中一个辅助函数，观察新生成的 token 怎样记入请求，以及停止条件怎样写入请求状态。

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)，文中行号对应这一提交。继续固定在 V1、离线文本生成、单候选 `n=1`、非流式输入、同步多进程客户端路径；本轮示例假设本次 `new_token_ids` 只有一个 token。

## 1. 阅读位置与执行顺序

| 顺序 | 源码位置 | 阅读重点 |
| --- | --- | --- |
| ① | [`_update_request_with_output()` 第 2582–2602 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L2582-L2602) | 先追加新 token，再检查停止，最后返回 token 列表和停止标志 |
| ② | [`check_stop()` 第 98–119 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/utils.py#L98-L119) | EOS、停止 token 和长度限制的判断顺序，以及请求状态赋值 |

```text
_update_request_with_output(request, new_token_ids)
  → stopped = False
  → 遍历本次新 token
      → request.append_output_token_ids(output_token_id)
      → check_stop(request, self.max_model_len)
          → 检查停止条件，命中时更新 request.status
      → 如果 stopped 为 True：退出遍历
  → 返回 (new_token_ids, stopped)
```

第 2596–2600 行包含一批多个 token 提前停止后的计算进度修正和列表裁剪，本轮先略过。这里的执行顺序图也省略了这部分细节。

## 2. 先记录 token，再检查停止

主线是：

```python
stopped = False
for num_new, output_token_id in enumerate(new_token_ids, 1):
    request.append_output_token_ids(output_token_id)
    stopped = check_stop(request, self.max_model_len)
    if stopped:
        # 多 token 时的修正与裁剪逻辑暂略
        break
return new_token_ids, stopped
```

**新 token 先记入请求，再用于判断是否停止。** 因此，`check_stop()` 读取最后一个输出 token 时，能看到刚刚追加的 token；长度判断也使用追加后的数量。

例如，核心侧 `request` 已保存的输出 token 是 `[101, 102]`，本次收到 `new_token_ids = [103]`：

1. 将 `103` 追加到请求，请求中累计的输出 token 变成 `[101, 102, 103]`。
2. 调用 `check_stop()`，检查追加后的状态。
3. 如果返回 `True`，退出当前遍历。
4. 返回 `([103], True)`。

即使 `103` 触发了停止，它也已经记入请求。这个顺序不能理解成“先检查停止，只有没停止才追加 token”。本轮只讨论核心请求的记录与状态，用户最终文本的后处理不在这一段中展开。

## 3. 请求中的累计记录与函数返回值

这个函数涉及两份用途不同的数据：

| 数据 | 在上述示例中的内容 | 含义 |
| --- | --- | --- |
| `request` 保存的输出 token | `[101, 102, 103]` | 该请求累计生成并记入的输出 token |
| 本次的 `new_token_ids` | `[103]` | 本次交给函数处理的新 token 列表 |

最后一行是：

```python
return new_token_ids, stopped
```

返回值是一个二元组，包含**本次处理后的 token 列表**和**是否停止的布尔值**。在本轮单 token 示例中：

- `new_token_ids = [103]`，`check_stop()` 返回 `True`：返回 `([103], True)`。
- `new_token_ids = [104]`，`check_stop()` 返回 `False`：返回 `([104], False)`。

学习时曾把第一种情况的返回值只写成 `[103]`。列表的内容正确，但漏掉了 `stopped`。同时，不能把函数返回的本次新 token 列表与请求中保存的累计输出混为一谈。

## 4. 停止条件按代码顺序检查

在本轮读到的 `check_stop()` 第 98–119 行中，三类条件依次出现：

| 顺序 | 条件 | 写入的状态 | 其他动作 |
| --- | --- | --- | --- |
| ① | 最后一个 token 等于 `sampling_params.eos_token_id` | `FINISHED_STOPPED` | 立即 `return True` |
| ② | 最后一个 token 属于 `sampling_params.stop_token_ids` | `FINISHED_STOPPED` | 设置 `request.stop_reason = last_token_id`，立即 `return True` |
| ③ | `request.num_tokens >= max_model_len`，或 `request.num_output_tokens >= request.max_tokens` | `FINISHED_LENGTH_CAPPED` | 立即 `return True` |

这里的顺序会影响最终状态。某个分支命中并 `return True` 后，下面的判断不会再执行。

例如，最后一个 token 不是 EOS，但既属于 `stop_token_ids`，又恰好使输出数量达到 `request.max_tokens`：

```python
request.status = RequestStatus.FINISHED_STOPPED
request.stop_reason = last_token_id
return True
```

执行到停止 token 分支就已经返回，因此不会继续进入长度分支，也不会把状态设成 `FINISHED_LENGTH_CAPPED`。本轮问答中曾先判断成长度上限，随后通过代码的先后顺序纠正。

这个优先顺序只概括本轮实际阅读的三个分支。其他停止机制和参数处理不在这里扩展。

## 5. 从“停止标志”看到“状态更新”

`check_stop()` 不只是计算一个布尔值。命中上述停止条件时，它还会执行：

```python
request.status = ...
```

这就是上一轮“更新核心请求状态”职责的一处具体实现。外层拿到 `stopped` 后可以决定是否继续遍历本次新 token，而 `request.status` 则在请求对象中记录它已经达到哪类结束条件。

源码第 2592–2593 行也特别说明：检查停止并更新请求状态，必须发生在构造 `EngineCoreOutput` 之前。本轮先掌握这个顺序，输出对象的构造细节留到后续阅读。

## 6. 自查题与参考答案

**问题一：请求原先保存的输出 token 是 `[101, 102]`，本次 `new_token_ids = [103]`，`check_stop()` 返回 `True`。执行后请求中的累计输出是什么？函数返回什么？**

<details>
<summary>参考答案</summary>

先追加 `103`，再检查停止，因此请求中的累计输出是 `[101, 102, 103]`。检查返回 `True` 后退出遍历，函数返回 `([103], True)`，而不是仅返回 `[103]`，也不是返回累计列表。

</details>

**问题二：最后一个 token 不是 EOS，但同时属于 `stop_token_ids`，并使输出数量达到 `request.max_tokens`。最终状态和 `stop_reason` 分别是什么？**

<details>
<summary>参考答案</summary>

状态为 `RequestStatus.FINISHED_STOPPED`，`request.stop_reason` 被设为 `last_token_id`。停止 token 分支位于长度判断之前，命中后立即返回 `True`，不会再执行长度判断。

</details>

**补充确认：本次 `new_token_ids = [104]`，且 `check_stop()` 返回 `False`，函数最终返回什么？**

<details>
<summary>参考答案</summary>

返回 `([104], False)`。这道补充题用于确认返回值包含列表和布尔值两项，第 10 轮已检验通过。

</details>

上一轮：[08 单步调度与模型执行](<08 单步调度与模型执行.md>)。下一轮：[10 请求 token 累积与长度限制](<10 请求 token 累积与长度限制.md>)。
