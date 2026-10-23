# 07 EngineCore 进程主循环与输入处理

第八轮承接前端客户端的结果队列，进入执行核心进程，理解它怎样接收输入、持续推进已有工作，再把内部结果交给发送线程。

源码基于 [vLLM 提交 e2d2680417](https://github.com/vllm-project/vllm/tree/e2d2680417432b80986a36d456e3d79173ee1a45)，文中行号对应这一提交。继续按 V1、离线文本生成、单候选 `n=1`、非流式输入，以及 `multiprocess_mode=True`、`asyncio_mode=False` 阅读。统计、Pooling、多模态和数据并行细节暂略。

## 1. 阅读位置与调用关系

主文件是 `vllm/v1/engine/core.py`。阅读顺序是“确认对象 → 看主循环 → 判断是否等待输入 → 执行一步 → 交接输出”。

| 顺序 | 源码位置 | 阅读重点 |
| --- | --- | --- |
| ① | [EngineCoreProc 第 1136 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L1136-L1137) | 确认它继承 `EngineCore`，负责在后台进程运行执行核心 |
| ② | [has_work 与主循环，第 1576–1598 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L1576-L1598) | 区分是否有工作与进程生命周期 |
| ③ | [_process_input_queue 第 1615–1643 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L1615-L1643) | 先看等待循环的条件，再看 `get(block=block)` |
| ④ | [_process_engine_step 第 1645–1653 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L1645-L1653) | 调用执行步骤，将内部结果写入核心进程的队列 |
| ⑤ | [step_fn 绑定，第 225–227 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L225-L227) | 明确下一轮进入哪一个 `step` 实现 |

```text
EngineCoreProc.run_busy_loop()
  → _process_input_queue()
  → _process_engine_step()
      → self.step_fn()
      → self.output_queue.put_nowait(...)
      → self.post_step(model_executed)
  → 回到进程主循环
```

## 2. 请求完成与进程主循环结束是两件事

省略请求计数发布后，`run_busy_loop()` 的主干是：

```python
while self._handle_shutdown():
    self._process_input_queue()
    self._process_engine_step()
```

主循环是否继续由 `_handle_shutdown()` 控制。完成当前请求，并不直接等于退出这个进程主循环；进程还可以回到输入处理阶段，等待后续工作。

这与离线入口中的 `_run_engine()` 要分开理解：

| 循环 | 所在位置 | 关注的结束条件 |
| --- | --- | --- |
| `_run_engine()` 的循环 | 前端离线调用路径 | `has_unfinished_requests()` 是否仍为真 |
| `run_busy_loop()` 的循环 | 执行核心进程 | `_handle_shutdown()` 是否允许继续 |

一次离线生成调用收集完最终结果，与执行核心进程的生命周期结束，属于不同层次。

## 3. has_work() 看的是已有工作，不只看输入队列

```python
def has_work(self) -> bool:
    return (
        self.engines_running
        or self.scheduler.has_requests()
        or bool(self.batch_queue)
    )
```

这个判断没有检查 `input_queue` 是否为空。即使当前没有新的输入消息，只要已有状态使 `has_work()` 为真，执行核心仍可以继续推进工作。

本轮把 `scheduler.has_requests()` 理解为“调度器仍报告有工作”；不把它简化成“输入队列里有新请求”，也不据此断言本轮一定能立即执行模型。其他分支的具体条件留待后续阅读。

`is_running()` 则检查当前是否仍处于 `EngineShutdownState.RUNNING`。它描述关闭状态，和 `has_work()` 各自回答不同的问题。

## 4. block=True 是否等待，要先看外层条件

省略日志、空闲回调和中止队列清理后，输入处理的关键结构如下：

```python
while not self.has_work() and self.is_running():
    block = self.process_input_queue_block
    try:
        req = self.input_queue.get(block=block)
        self._handle_client_request(*req)
    except queue.Empty:
        break
    if not block:
        break

# Handle any more client requests.
while not self.input_queue.empty():
    req = self.input_queue.get_nowait()
    self._handle_client_request(*req)
```

这里先决定是否进入等待循环，再决定取队列元素时是否允许阻塞。不能只看到 `block=True`，就认定执行核心一定停下来等待新输入。

假设 `is_running()` 为真：

| 情况 | 执行过程 |
| --- | --- |
| `has_work()` 为真 | 跳过第一个 `while`，其中的 `get(block=block)` 根本不会执行 |
| `has_work()` 为假，进入循环，`block=True`，队列为空 | 当前线程在 `get()` 中等待输入 |
| `has_work()` 为假，进入循环，`block=True`，队列已有元素 | 直接取得元素，无须等待新元素到达 |
| `has_work()` 为假，进入循环，`block=False`，队列为空 | `get()` 抛出 `queue.Empty`，被捕获后退出这个循环 |

后面的第二个 `while` 使用 `get_nowait()`，处理当前已有的其余输入消息，不以等待下一条新输入为目的。

本轮纠正的关键场景是：

```text
已有请求使 has_work() == True
input_queue 当前为空
process_input_queue_block == True
  → 第一个 while 的条件为 False，跳过等待
  → 第二个 while 也因队列为空而跳过
  → _process_input_queue() 返回
  → 主循环继续调用 _process_engine_step()
```

因此，已有工作可以在没有新消息到达时持续推进。阅读带阻塞操作的代码时，应从外向内检查：能否进入这段代码、是否真的执行队列读取、读取时队列是否为空。

## 5. 执行一步，并把结果交给输出队列

```python
outputs, model_executed = self.step_fn()

for output in outputs.items() if outputs else ():
    self.output_queue.put_nowait(output)

self.post_step(model_executed)
```

在当前阅读路径中，`step_fn()` 返回的 `outputs` 是按客户端分组的内部输出字典。遍历 `.items()` 时，每个 `output` 是一对 `(client_index, EngineCoreOutputs)`，随后被放入核心进程自己的 `output_queue`。

`model_executed` 被交给执行后的钩子 `post_step()`。它的具体计算方式在下一篇进入 `EngineCore.step()` 后再说明。

`step_fn` 在初始化时按以下条件绑定：

```python
self.step_fn = (
    self.step if self.batch_queue is None else self.step_with_batch_queue
)
```

接下来的阅读选择 **`batch_queue is None` 的基础分支**，所以进入 `EngineCore.step()`。这是本次选择的分支，不能推广为所有配置都采用这一分支。

## 6. 核心进程的 output_queue 与前端 outputs_queue

上一轮看到的 `SyncMPClient.outputs_queue`，与这里的 `EngineCoreProc.output_queue` 是不同队列。

```text
执行核心进程
  主循环：step_fn() → output_queue.put_nowait((client_index, outputs))
                              │
                              ▼
                         output_queue
                              │
  发送线程：出队 → 编码 → 通过 socket 发送
                              │
                         跨进程消息
                              │
                              ▼
前端进程
  接收线程：接收 → 解码 → outputs_queue.put_nowait(outputs)
                              │
                              ▼
                         outputs_queue
                              │
  调用线程：SyncMPClient.get_output() → outputs_queue.get()
```

| 队列 | 所属位置 | 本轮普通结果路径中的用途 |
| --- | --- | --- |
| `EngineCoreProc.output_queue` | 执行核心进程内 | 主循环把客户端索引和内部输出交给发送线程 |
| `SyncMPClient.outputs_queue` | 前端进程内 | 接收线程把解码后的结果交给调用线程 |

辅助核对位置：

- [核心进程的队列创建，第 1155–1156 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L1155-L1156)。
- [核心发送线程的创建与启动，第 1239–1248 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L1239-L1248)。
- [发送线程出队，第 2035–2042 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L2035-L2042)，以及[普通客户端结果的编码与发送，第 2057–2061 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core.py#L2057-L2061)。
- [前端队列创建，第 971 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core_client.py#L971)，[接收解码并入队，第 1001–1009 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core_client.py#L1001-L1009)，[调用线程出队，第 1026–1036 行](https://github.com/vllm-project/vllm/blob/e2d2680417432b80986a36d456e3d79173ee1a45/vllm/v1/engine/core_client.py#L1026-L1036)。

两个队列分别用于各自进程内的线程协作，中间通过 socket 传输消息。不能把它们看作同一个跨进程共享队列。

## 7. 自查题与参考答案

下面围绕本轮已确认的等待条件、生命周期和队列归属复习。

**问题一：`has_work()` 为真、输入队列为空，而且 `process_input_queue_block=True`。执行核心是否必须等待一条新消息才能继续计算？为什么？**

<details>
<summary>参考答案</summary>

不必等待。第一个 `while` 要求 `not self.has_work()`，当前条件不成立，因此不会执行其中的阻塞 `get()`。第二个循环也因队列为空而跳过，输入处理返回后，主循环可以继续调用 `_process_engine_step()` 推进已有工作。

</details>

**问题二：一个请求完成，是否意味着 `run_busy_loop()` 立即退出？`_process_engine_step()` 写入的队列是否就是前端 `get_output()` 读取的队列？**

<details>
<summary>参考答案</summary>

请求完成不直接使进程主循环退出，外层循环由 `_handle_shutdown()` 控制。两个队列也不同：核心主循环写入核心进程的 `output_queue`，发送线程从中取出并发送消息；前端接收线程解码后再写入前端的 `outputs_queue`，由 `get_output()` 读取。

</details>

上一轮：[06 同步多进程客户端与结果队列](<06 同步多进程客户端与结果队列.md>)。下一轮：[08 单步调度与模型执行](<08 单步调度与模型执行.md>)，进入 `batch_queue is None` 分支绑定的 `EngineCore.step()`。
