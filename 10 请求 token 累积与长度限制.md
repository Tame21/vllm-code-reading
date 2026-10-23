# 10 请求 token 累积与长度限制

第十一轮承接新 token 的记录与停止检查，进一步区分核心 `Request` 中的两个 token 列表，以及 `check_stop()` 使用的两种长度。重点是：**同一个请求会连续累积生成 token，总长度与生成长度分别统计不同的内容。**

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)，文中行号对应这一提交。继续按 V1、离线文本生成、单候选 `n=1`、非流式输入阅读；本轮例子使用普通文本 token 输入，初始化时走 `prompt_token_ids` 非空且 `prompt_is_token_ids is None` 的分支。本轮已完成阅读与纠错检验。

## 1. 阅读位置与顺序

| 顺序 | 源码位置 | 阅读重点 |
| --- | --- | --- |
| ① | [Request 初始化，第 146–150 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/request.py#L146-L150) | 输出列表初始为空，完整列表从 prompt 副本开始 |
| ② | [append_output_token_ids 第 272–281 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/request.py#L272-L281) | 单个 token 用 `append()`，token 列表用 `extend()`，两份列表同时更新 |
| ③ | [num_tokens 第 294–296 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/request.py#L294-L296) 与 [num_output_tokens 第 302–304 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/request.py#L302-L304) | 两种计数分别取哪个列表的长度 |
| ④ | [check_stop 第 114–119 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/utils.py#L114-L119) | 对应总长度限制与生成长度限制 |

主读范围内的哈希更新、结构化输出和 `num_tokens_with_spec` 本轮略过。

## 2. 两个列表怎样初始化

当前普通文本输入分支中的关键赋值是：

```python
self._output_token_ids: list[int] = []
# 当前分支：prompt_token_ids 存在，且 prompt_is_token_ids is None。
self._all_token_ids = self.prompt_token_ids.copy()
```

这是摘出当前分支后的初始化片段，完整条件见源码链接。

如果 `prompt_token_ids = [11, 12, 13]`，初始化后：

```python
_output_token_ids = []
_all_token_ids = [11, 12, 13]
```

`_output_token_ids` 只保存生成的 token；`_all_token_ids` 从 prompt 的副本开始，后续还会包含生成的 token。

## 3. 连续调用会在同一个请求上追加

`append_output_token_ids()` 根据参数类型选择追加方式：

```python
if isinstance(token_ids, int):
    self._output_token_ids.append(token_ids)
    self._all_token_ids.append(token_ids)
else:
    self._output_token_ids.extend(token_ids)
    self._all_token_ids.extend(token_ids)
```

传入整数时，将一枚 token 加到两份列表末尾；传入列表时，将其中各枚 token 依次加到两份列表末尾。已有内容保留，调用之间也不会重新初始化请求。

沿着同一个 `request` 连续执行：

```python
request.append_output_token_ids(101)
request.append_output_token_ids([102, 103])
request.append_output_token_ids(104)
```

列表与计数变化如下：

| 时刻 | `_output_token_ids` | `_all_token_ids` | `num_tokens` | `num_output_tokens` |
| --- | --- | --- | --- | --- |
| 初始化后 | `[]` | `[11, 12, 13]` | 3 | 0 |
| 追加 `101` 后 | `[101]` | `[11, 12, 13, 101]` | 4 | 1 |
| 再追加 `[102, 103]` 后 | `[101, 102, 103]` | `[11, 12, 13, 101, 102, 103]` | 6 | 3 |
| 再追加 `104` 后 | `[101, 102, 103, 104]` | `[11, 12, 13, 101, 102, 103, 104]` | 7 | 4 |

本轮最初把第二次调用后的结果写成了 `_output_token_ids = [102, 103]`，并把完整列表中的 `101` 漏掉。纠正点是：题目中的两次调用作用于同一个请求，`extend()` 会在已有 `[101]` 后追加 `[102, 103]`。补题中再追加 `104`，确认得到累计输出 `[101, 102, 103, 104]` 和计数 `7`、`4`。

## 4. 两种数量分别统计什么

两个属性的实现分别是：

```python
@property
def num_tokens(self) -> int:
    return len(self._all_token_ids)

@property
def num_output_tokens(self) -> int:
    return len(self._output_token_ids)
```

在当前普通文本场景中：

| 属性 | 统计内容 |
| --- | --- |
| `num_tokens` | prompt token 数加上累计生成 token 数 |
| `num_output_tokens` | 累计生成 token 数 |

这里统计的是请求当前保存的累计内容，不能把 `num_output_tokens` 当成本轮新增量。生成 token 同时存在于两份列表中，因此也不能把两份列表的长度相加来计算请求总长度。

## 5. 对应到两种长度限制

上一轮读过的 `check_stop()` 中有以下判断：

```python
if (
    request.num_tokens >= max_model_len
    or request.num_output_tokens >= request.max_tokens
):
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    return True
```

- `request.num_tokens >= max_model_len`：检查包含 prompt 在内的总长度。
- `request.num_output_tokens >= request.max_tokens`：检查累计生成长度。

两项通过 `or` 连接，任意一项成立都会进入长度结束分支。生成数量还未达到 `max_tokens`，也可能因为包含 prompt 的总长度达到 `max_model_len` 而停止。

例如，请求有 5 枚 prompt token，累计生成 2 枚 token，设置 `max_model_len = 7`、`request.max_tokens = 5`。假设最后一枚 token 不是 EOS，也不属于 `stop_token_ids`，从而没有在前面的停止分支提前返回：

```text
num_tokens = 5 + 2 = 7
num_output_tokens = 2
7 >= 7 为 True
2 >= 5 为 False
  → 总长度条件成立
  → status = FINISHED_LENGTH_CAPPED
  → check_stop() 返回 True
```

这次状态变化发生在核心的 `Request` 上。本轮讨论的列表和计数也属于核心请求，与前面输出处理器中 `req_state.detokenizer` 保存的解码状态分别位于不同对象中。

## 6. 自查题与参考答案

**问题一：普通文本请求的 `prompt_token_ids = [11, 12, 13]`。初始化后，依次调用 `append_output_token_ids(101)` 和 `append_output_token_ids([102, 103])`，最终两份列表和两种数量分别是什么？接着再追加 `104` 呢？**

<details>
<summary>参考答案</summary>

第二次调用后，`_output_token_ids = [101, 102, 103]`，`_all_token_ids = [11, 12, 13, 101, 102, 103]`，`num_tokens = 6`，`num_output_tokens = 3`。

再追加 `104` 后，`_output_token_ids = [101, 102, 103, 104]`，`_all_token_ids = [11, 12, 13, 101, 102, 103, 104]`，两个计数分别为 `7`、`4`。连续调用会保留并累积此前的内容。

</details>

**问题二：请求有 5 枚 prompt token，累计生成 2 枚 token，`max_model_len = 7`、`request.max_tokens = 5`。最后一枚 token 不是 EOS，也不属于 `stop_token_ids`。`check_stop()` 返回什么，设置什么状态，由哪个长度条件触发？**

<details>
<summary>参考答案</summary>

返回 `True`，设置 `RequestStatus.FINISHED_LENGTH_CAPPED`。总长度为 `5 + 2 = 7`，因此 `request.num_tokens >= max_model_len` 成立。累计生成长度只有 `2`，尚未达到生成长度上限 `5`。

</details>

上一轮：[09 新 token 记录与停止检查](<09 新 token 记录与停止检查.md>)。下一处阅读入口：[`Scheduler.schedule()` 第 621–641 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/core/sched/scheduler.py#L621-L641)。该入口属于第 12 轮，已布置阅读，尚未完成问答检验，本篇不展开其实现。
