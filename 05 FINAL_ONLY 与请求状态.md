# 05 FINAL_ONLY 与请求状态

第六轮继续追踪输出后处理：使用 `FINAL_ONLY` 时，中间结果暂不返回，生成进度保存在哪里？请求结束后，最终结果又怎样包含之前多轮生成的 token？

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)，文中行号对应这一提交。本轮聚焦单个候选答案（`n=1`）的普通文本生成，按同步接口、非流式输入、`stream_interval=1` 的路径阅读，先略过 pooling、父子请求聚合和异步队列分支。

## 1. 从哪里开始，怎样读

本轮围绕同一请求的状态，按“找到状态 → 更新状态 → 判断是否输出 → 构造最终结果”的顺序阅读。

| 顺序 | 源码位置 | 阅读重点 |
| --- | --- | --- |
| ① | [process_outputs 第 755–762 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/output_processor.py#L755-L762) | 新建本轮结果列表，按请求 ID 查找请求状态 |
| ② | [第 823–833 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/output_processor.py#L823-L833) | 用新 token 更新解码器状态，检查停止字符串 |
| ③ | [make_request_output 第 354–369 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/output_processor.py#L354-L369) | 定位 `FINAL_ONLY` 暂不返回结果的实际判断 |
| ④ | [调用处第 849–866 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/output_processor.py#L849-L866) | 区分辅助函数返回 `None` 与外层继续处理其他请求 |
| ⑤ | [_new_completion_output 第 483–489 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/output_processor.py#L483-L489) | 最终结果从解码器中取得累积的输出 token |

## 2. 请求状态与本轮结果列表是不同的容器

`process_outputs()` 每次调用都会新建结果列表，再逐条处理执行核心返回的内部结果：

```python
request_outputs = []
reqs_to_abort = []

for engine_core_output in engine_core_outputs:
    req_id = engine_core_output.request_id
    req_state = self.request_states.get(req_id)
    if req_state is None:
        continue
```

这里的 `req_state` 是对应请求的 `RequestState`，其中保存了输出类型、解码器等状态。可结合 [RequestState 的初始化](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/output_processor.py#L183-L228) 确认这些字段的归属。

| 对象 | 保存的内容 | 生命周期 |
| --- | --- | --- |
| `self.request_states` | 按请求 ID 索引的请求状态 | 跨多次输出处理调用 |
| `req_state.detokenizer` | 该请求的增量解码状态和累积输出 token | 随该请求跨轮保留 |
| `request_outputs` | 本次调用准备向上返回的用户结果 | 每次 `process_outputs()` 新建 |

跨轮生成进度保存在请求的解码器状态中，不能把本次调用的 `request_outputs` 列表当作保存全部历史 token 的地方。

## 3. 先更新生成状态，再判断是否返回结果

对于文本生成分支，输出处理器先执行：

```python
stop_string = req_state.detokenizer.update(
    new_token_ids, finish_reason == FinishReason.STOP
)
```

`new_token_ids` 来自本次执行核心结果。`update()` 更新该请求的解码器状态，同时检查停止字符串。

需要分清状态变化与返回值：

- 新 token 和解码进度由 `req_state.detokenizer` 内部保存。
- `stop_string` 接收的是停止字符串检查结果，不是累积 token 列表。

如果命中停止字符串，后续代码还会更新 `finish_reason`、`stop_reason`，并按需要裁剪超过停止位置的 token。本轮的关键是执行顺序：解码器更新发生在调用 `make_request_output()` 之前。

## 4. FINAL_ONLY 在哪里暂缓用户输出

`RequestState.make_request_output()` 开头的判断是：

```python
finished = finish_reason is not None
final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

if not finished and final_only:
    return None
```

当请求尚未结束，而且输出类型是 `FINAL_ONLY` 时，这个辅助函数返回 `None`，表示本轮暂不构造可向用户返回的结果。

这不会撤销之前已经执行的解码器更新，也不会因为暂不输出而清空生成进度。它控制的是本轮是否产生用户结果。

## 5. 辅助函数返回 None，外层仍可继续处理

调用处使用了赋值表达式：

```python
if request_output := req_state.make_request_output(...):
    ...
```

可以按两步理解：

```python
request_output = req_state.make_request_output(...)
if request_output:
    ...
```

在本轮同步接口的路径中，构造出结果时会将其加入 `request_outputs`。如果返回 `None`，本次不会追加该请求的用户结果，但外层 `for` 可以继续处理后面的请求。

例如，一次调用收到 A、B 两个请求的内部结果，且都使用 `FINAL_ONLY`：

```text
A 尚未结束
  → 更新 A 的解码器状态
  → make_request_output() 返回 None
  → 不把 A 加入本轮用户结果列表
B 已经结束
  → 更新 B 的解码器状态
  → 构造 B 的最终结果
  → 把 B 加入本轮用户结果列表
```

因此，本轮 `request_outputs` 可以只包含 B。辅助函数返回 `None`，并不等于整个 `process_outputs()` 立即返回。

## 6. 最终结果怎样包含此前多轮 token

请求结束后，会继续进入生成候选结果的构造流程。`_new_completion_output()` 中有以下代码：

```python
finished = finish_reason is not None
delta = self.output_kind == RequestOutputKind.DELTA

text = self.detokenizer.get_next_output_text(finished, delta)
if not delta:
    token_ids = self.detokenizer.output_token_ids
```

`FINAL_ONLY` 不等于 `DELTA`，因此进入 `if not delta` 分支，从解码器中取得累积的输出 token，而非只使用本轮新增的 token。

假设同一请求连续三轮产生以下结果，第三轮因达到长度上限结束，没有命中停止字符串：

| 轮次 | 本轮新增 token | 请求是否结束 | 是否产生 `RequestOutput` |
| --- | --- | --- | --- |
| 第一轮 | `[101]` | 否 | 否，辅助函数返回 `None` |
| 第二轮 | `[102]` | 否 | 否，辅助函数返回 `None` |
| 第三轮 | `[103]` | 是 | 是，生成最终结果 |

最终结果的 `token_ids` 是 `[101, 102, 103]`。前两轮虽然没有用户结果，但生成进度已经保存在解码器状态中。

## 7. 连接本轮的数据流

```text
执行核心返回本轮的新 token
  → 按请求 ID 找到 RequestState
  → 更新该请求的 detokenizer 状态
  → make_request_output() 判断本轮是否输出
      → FINAL_ONLY 且未结束：返回 None，保留生成进度
      → 请求结束：构造最终结果，读取累积输出 token
  → 将本轮可返回的结果加入 request_outputs
```

阅读时要区分本轮新增 token、跨轮解码状态和本轮用户结果列表。这三个层次对应不同的对象，也有不同的保存范围。

## 8. 自查题与参考答案

1. 跨轮生成进度保存在什么对象中？`request_outputs` 会跨调用累积历史 token 吗？
2. `stop_string = detokenizer.update(...)` 中，保存生成进度的对象和赋给 `stop_string` 的返回值分别是什么？
3. 同一请求连续收到 `[101]`、`[102]`、`[103]`，第三轮因长度上限结束，使用 `FINAL_ONLY` 且没有停止字符串裁剪。三轮分别是否产生 `RequestOutput`？最终 token 列表是什么？
4. 同次处理 A、B 两个请求，A 未结束、B 已结束，均使用 `FINAL_ONLY`。A 的辅助函数返回 `None` 后，还会处理 B 吗？本轮用户结果包含谁？

<details>
<summary>参考答案</summary>

1. 保存在对应 `RequestState` 的解码器状态中。`request_outputs` 是本次 `process_outputs()` 新建的用户结果列表，不负责跨轮保存历史 token。
2. 状态变化保存在解码器内部；`stop_string` 接收停止字符串检查结果。
3. 前两轮不产生，第三轮产生。最终 token 列表是 `[101, 102, 103]`，来自解码器中累积的输出 token。
4. 会继续处理 B。A 不产生本轮用户结果，B 产生最终结果，本轮 `request_outputs` 包含 B。辅助函数返回 `None` 不会直接终止外层循环。

</details>

上一轮：[04 引擎输出与用户结果](<04 引擎输出与用户结果.md>)。下一轮：[06 同步多进程客户端与结果队列](<06 同步多进程客户端与结果队列.md>)，回到 `get_output()`，确认由谁接收执行核心的结果，以及调用方怎样取出结果。
