# AstrBot Command2LLM 插件

将已启用插件的普通命令桥接给 LLM。插件同时提供稳定的单命令路由模式，以及复用 AstrBot 原生 Agent 的多轮工具调用模式。

## 工作模式

### 单命令路由（默认）

1. 扫描已启用、未被黑名单排除且当前会话允许使用的插件命令。
2. 将插件唯一名、展示名、插件简介、命令说明、用法和别名作为结构化 JSON 目录交给路由模型。
3. 路由模型一次完成“是否需要命令”和“选择哪个命令”，无匹配时继续 AstrBot 默认 LLM 流程。
4. 插件直接调用选定的注册 handler，并保留命令原有的权限、消息类型和参数过滤条件。

插件目录和用户消息只作为不可信数据传入，第三方插件描述不会被拼接为 system prompt。

### 多轮工具调用

在 Dashboard 开启 `enable_multi_tool_agent` 后，插件复用 AstrBot 原生 Agent：

- 每个普通命令转换为一个独立的 `FunctionTool`，工具名固定对应具体 handler，模型只填写命令参数。
- 可以在一次回复中连续或多次调用不同命令，并根据上一次结果决定下一步。
- 可选择是否同时允许其他插件已经注册的原生 LLM 工具。
- 正常 LLM 触发消息直接进入原生 Agent，不增加一次前置判断请求。
- `listen_mode=global` 下的普通群聊消息会先由 `judge_provider_id` 根据实际能力目录判断是否需要启动 Agent。
- 纯文本命令结果回传给模型生成最终回复；图片、卡片等富媒体会原样发送一次，并向模型返回已发送标记。

Agent 总步数、工具超时、工具 schema 模式、会话历史、人设和结果持久化沿用 AstrBot 全局配置；插件工具还可单独设置调用上限。

## 配置

- `wake_word`：命令唤醒词，默认 `/`。
- `listen_mode`：
  - `global`：所有非命令消息都可参与路由；多轮模式下，非 LLM 触发消息先做能力目录预判。
  - `llm_triggered_only`：只处理原本会触发 AstrBot LLM 的消息，如私聊、@机器人或全局唤醒词。
- `judge_provider_id`：单命令结构化路由和 `global` 多轮预判使用的模型；留空时使用当前会话模型。
- `enable_multi_tool_agent`：启用多轮工具调用，默认关闭。
- `agent_tool_types`：多选允许的插件工具类型：
  - `command`：普通 `@filter.command` 命令，默认启用。
  - `native_tool`：插件原生 `FunctionTool`。
- `multi_tool_limits`：仅在多轮模式中限制本插件管理的工具调用：
  - `max_calls_per_round`：一次 LLM 响应允许的插件工具调用数。
  - `max_call_rounds`：一次回复允许发生插件工具调用的 LLM 轮数。
  - 两项填负数均表示不额外限制，AstrBot 全局限制仍然生效。
- `show_builtin_cmds`：是否把 AstrBot 保留插件的普通命令加入命令目录。
- `plugin_blacklist`：Dashboard 插件选择器。黑名单插件的普通命令和原生工具都不会参与判断或调用。

`agent_tool_types` 和 `plugin_blacklist` 只管理插件能力。AstrBot 内置工具、MCP 工具和 handoff 工具继续遵循 AstrBot 自身配置。

## 控制命令

```text
/ai_enable   # 临时启用自动调用
/ai_disable  # 临时关闭自动调用
/ai_status   # 查看运行模式、工具类型和有效黑名单
```

## 行为与限制

- 插件只执行注册表中真实存在的命令，不执行没有 handler 的文本命令。
- 同名命令会根据插件来源、插件简介、命令说明和用法选择具体 handler。
- 命令别名归并到同一个工具，减少工具数量；执行时统一使用该 handler 的主命令名。
- 会话级插件范围 `event.plugins_name` 会同时约束命令目录、命令工具和原生插件工具。
- 黑名单优先使用插件唯一名，也能识别旧配置中已经保存的展示名。
- 多轮模式要求当前聊天模型支持 function calling/tool use；不支持时可关闭该选项继续使用单命令模式。
- 插件会跳过以 `/`、`#`、`!` 或所配置 `wake_word` 开头的显式命令，让 AstrBot 正常执行它们。

## 兼容性

- 目标运行环境：AstrBot 4.25.3。
- 单命令模式依赖 `llm_generate` 和 `get_current_chat_provider_id`。
- 多轮模式通过 `on_llm_request` 接入 AstrBot 主 Agent，不维护私有 tool-calls 状态机。

## 许可证

本项目遵循 [LICENSE](LICENSE)。
