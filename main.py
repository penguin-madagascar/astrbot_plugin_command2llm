import collections
import copy
import inspect
from dataclasses import dataclass
from typing import Any

from astrbot.api.event import filter
from astrbot.api.star import Context, Star, register
from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.agent.run_context import ContextWrapper

try:
    from astrbot.core.agent.tool import FunctionTool, ToolExecResult, ToolSet
except ImportError:
    from astrbot.core.agent.tool import FunctionTool, ToolSet
    ToolExecResult = Any
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.config import AstrBotConfig
from astrbot.core.star.star_handler import star_handlers_registry
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.filter.command_group import CommandGroupFilter

AGENT_AVAILABLE = True
STAR_AVAILABLE = True
LISTEN_MODE_GLOBAL = "global"
LISTEN_MODE_LLM_TRIGGERED_ONLY = "llm_triggered_only"


@dataclass(frozen=True)
class RegisteredCommand:
    name: str
    description: str
    plugin_name: str
    handler: Any
    command_filter: Any
    event_filters: tuple[Any, ...]


@register("command2llm", "vmoranv", "让大模型能够调用所有插件命令的插件", "1.0.1")
class Command2LLMPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.command_cache = {}
        self.cache_timeout = 300  # 缓存5分钟
        self.enabled = True  # 插件开关
        self.threshold = 0.6  # 命令匹配阈值
        self.last_messages = {}  # 存储用户最后的消息
        self.wake_word = config.get('wake_word', '/')  # 获取唤醒词，默认为/
        self.show_builtin_cmds = bool(config.get("show_builtin_cmds", False))
        self.custom_cmds = config.get("custom_cmds", []) or []
        self.plugin_blacklist = {str(item).strip() for item in (config.get("plugin_blacklist", []) or []) if str(item).strip()}
        self.judge_provider_id = (config.get("judge_provider_id", "") or "").strip()
        self.runtime_supported = all(
            hasattr(self.context, attr)
            for attr in ("tool_loop_agent", "llm_generate", "get_current_chat_provider_id")
        )
        logger.info(f"插件初始化完成，唤醒词设置为: {self.wake_word}, 监听模式: {self._get_listen_mode()}")
        
        

    async def initialize(self):
        """插件初始化方法"""
        if not STAR_AVAILABLE:
            logger.warning("Star模块不可用，插件功能受限")
        if not self.runtime_supported:
            logger.warning("当前 AstrBot 版本缺少 command2llm 所需的 Agent API，插件将保持加载但不会自动调用命令。建议升级到 4.5.7+。")
        logger.info("Command2LLM插件初始化完成")

    @filter.event_message_type(filter.EventMessageType.ALL, priority=-100)
    async def handle_message(self, event, *args, **kwargs):
        """拦截所有消息，判断是否需要调用命令"""
        try:
            # 检查插件是否启用
            if not self.enabled or not self.runtime_supported:
                return

            # 跳过bot自己发送的消息
            if hasattr(event, 'message_obj') and event.message_obj and hasattr(event.message_obj, 'sender') and hasattr(event.message_obj.sender, 'user_id'):
                if hasattr(event.message_obj, 'self_id') and event.message_obj.sender.user_id == event.message_obj.self_id:
                    logger.info(f"跳过bot自己的消息: {event.message_str}")
                    return

            message_str = event.message_str.strip()
            session_id = event.session_id

            if not message_str:
                return

            # 跳过所有命令消息（让命令直接执行，不拦截）
            logger.info(f"检查消息: '{message_str}', 唤醒词: '{self.wake_word}'")
            command_prefixes = {self.wake_word, '/', '#', '!'}
            if any(prefix and message_str.startswith(prefix) for prefix in command_prefixes):
                logger.info(f"跳过命令消息: {message_str}")
                return

            # 跳过本插件的控制命令
            control_commands = ['ai_enable', 'ai_disable', 'ai_status', 'refresh_commands']
            if any(message_str == f'{self.wake_word}{cmd}' or message_str == cmd for cmd in control_commands):
                return

            if self._event_has_result(event):
                logger.info("消息已被其他插件处理，跳过 command2llm")
                return

            if not self._should_process_in_listen_mode(event):
                logger.info("消息未触发 AstrBot LLM，跳过 command2llm")
                return

            # 存储最后一条消息
            self.last_messages[session_id] = message_str

            # 获取当前会话使用的聊天模型ID
            try:
                umo = event.unified_msg_origin
                current_provider_id = await self.context.get_current_chat_provider_id(umo=umo)
            except Exception:
                return  # 无法获取提供商时跳过
            
            if not current_provider_id:
                return  # 没有LLM提供商时跳过

            judge_provider_id = self._resolve_judge_provider_id(current_provider_id)

            # 使用LLM判断是否需要调用命令
            if not await self._should_call_command(event, judge_provider_id):
                logger.info(f"消息不需要调用命令: {message_str}")
                return

            # 使用Agent工具执行命令
            if not AGENT_AVAILABLE:
                logger.error("Agent模块不可用")
                return
                
            registered_commands = self._get_registered_commands()
            commands_list = self._get_all_available_commands(registered_commands)
            logger.info(f"可用命令列表: {commands_list[:10]}...")  # 只显示前10个
            
            # 创建工具集合
            command_tool = ExecuteCommandTool(
                registered_commands,
                self._get_global_config(),
                self.wake_word,
            )
            tools = ToolSet([command_tool])
            
            # 构建系统提示
            system_prompt = f"""你是一个命令执行助手，负责根据用户消息选择合适的命令并执行。

可用命令列表：
{chr(10).join(commands_list)}

重要规则：
1. 根据用户消息选择合适的命令
2. 使用execute_command工具执行命令，参数只需要命令名（不要/前缀）
3. 执行命令后，工具会返回执行结果，请根据结果给用户一个简短的回复
4. 如果用户消息不需要执行任何命令，直接回复用户
5. 每次对话最多只执行一次命令"""

            # 调用Agent处理消息
            try:
                await self.context.tool_loop_agent(
                    event=event,
                    chat_provider_id=current_provider_id,
                    prompt=message_str,
                    system_prompt=system_prompt,
                    tools=tools,
                    max_steps=2,  # 允许两步：执行命令 + 生成回复
                    tool_call_timeout=60
                )

                logger.info("Agent处理完成")
                # 停止事件传播，避免重复处理
                event.stop_event()
                return
            except Exception as e:
                logger.error(f"Agent调用失败: {str(e)}")
                return
                
        except Exception as e:
            logger.error(f"消息处理错误: {str(e)}")

    async def _should_call_command(self, event, provider_id) -> bool:
        """判断是否需要调用命令"""
        try:
            message_str = event.message_str.strip()
            
            # 简单的启发式判断
            call_keywords = [
                '帮我', '请', '能否', '可以', '能不能', '如何', '怎么', '怎样',
                '查看', '搜索', '找', '获取', '设置', '配置', '启动', '停止',
                '天气', '时间', '日期', '新闻', '音乐', '视频', '图片'
            ]
            
            # 如果消息包含调用关键词，则返回True
            for keyword in call_keywords:
                if keyword in message_str:
                    logger.info(f"匹配到关键词: {keyword}")
                    return True
            
            # 使用LLM进行更精确的判断
            try:
                llm_resp = await self.context.llm_generate(
                    chat_provider_id=provider_id,
                    prompt=f"请判断以下消息是否需要调用某个命令或工具来处理：'{message_str}'\n只需要回答'是'或'否'。",
                    system_prompt="你是一个消息分类器，判断用户消息是否需要调用命令或工具。"
                )
                
                result = llm_resp.completion_text.strip()
                logger.info(f"LLM判断结果: {result}")
                return '是' in result
            except Exception:
                return False
            
        except Exception as e:
            logger.error(f"判断命令调用时出错: {str(e)}")
            return False

    def _resolve_judge_provider_id(self, current_provider_id: str) -> str:
        """获取用于判断是否触发命令的提供商 ID"""
        return self.judge_provider_id or current_provider_id

    def _get_listen_mode(self) -> str:
        """获取消息监听模式"""
        listen_mode = str(self.config.get("listen_mode", LISTEN_MODE_GLOBAL) or LISTEN_MODE_GLOBAL).strip()
        if listen_mode not in {LISTEN_MODE_GLOBAL, LISTEN_MODE_LLM_TRIGGERED_ONLY}:
            return LISTEN_MODE_GLOBAL
        return listen_mode

    def _should_process_in_listen_mode(self, event) -> bool:
        """根据监听模式判断是否处理当前消息"""
        if self._get_listen_mode() == LISTEN_MODE_GLOBAL:
            return True
        return self._is_llm_triggered_message(event)

    def _is_llm_triggered_message(self, event) -> bool:
        """判断当前消息是否本来会触发 AstrBot LLM 回复"""
        is_at_or_wake_command = getattr(event, "is_at_or_wake_command", None)
        if is_at_or_wake_command:
            return True

        if self._private_message_triggers_llm(event):
            return True

        if is_at_or_wake_command is False:
            return False

        return self._is_bot_mentioned(event) or self._has_global_wake_prefix(event)

    def _private_message_triggers_llm(self, event) -> bool:
        """兼容旧事件对象：私聊默认触发 LLM，除非全局配置要求唤醒词"""
        if not self._is_private_message(event):
            return False

        platform_settings = self._config_get(self._get_global_config(), "platform_settings", {}) or {}
        needs_wake_prefix = bool(self._config_get(platform_settings, "friend_message_needs_wake_prefix", False))
        return not needs_wake_prefix or self._has_global_wake_prefix(event)

    def _is_private_message(self, event) -> bool:
        get_message_type = getattr(event, "get_message_type", None)
        if callable(get_message_type):
            message_type = get_message_type()
            message_type_text = str(getattr(message_type, "name", message_type)).lower()
            if "group" in message_type_text:
                return False
            if "private" in message_type_text or "friend" in message_type_text:
                return True

        get_group_id = getattr(event, "get_group_id", None)
        if callable(get_group_id):
            return not bool(get_group_id())

        message_obj = getattr(event, "message_obj", None)
        return not bool(getattr(message_obj, "group_id", None))

    def _is_bot_mentioned(self, event) -> bool:
        message_obj = getattr(event, "message_obj", None)
        self_id = str(getattr(message_obj, "self_id", "") or "")
        if not self_id:
            return False

        for component in getattr(message_obj, "message", []) or []:
            if component.__class__.__name__ == "At" and str(getattr(component, "qq", "")) == self_id:
                return True
        return False

    def _has_global_wake_prefix(self, event) -> bool:
        message_str = self._get_event_message_str(event)
        if not message_str:
            return False

        global_config = self._get_global_config()
        provider_settings = self._config_get(global_config, "provider_settings", {}) or {}
        wake_prefixes = []
        wake_prefixes.extend(self._normalize_prefixes(self._config_get(global_config, "wake_prefix", [])))
        wake_prefixes.extend(self._normalize_prefixes(self._config_get(provider_settings, "wake_prefix", "")))
        return any(message_str.startswith(prefix) for prefix in wake_prefixes)

    def _get_event_message_str(self, event) -> str:
        get_message_str = getattr(event, "get_message_str", None)
        if callable(get_message_str):
            message_str = get_message_str()
            if isinstance(message_str, str):
                return message_str.strip()
        return str(getattr(event, "message_str", "") or "").strip()

    def _get_global_config(self):
        get_config = getattr(self.context, "get_config", None)
        if callable(get_config):
            return get_config()
        return {}

    def _config_get(self, config, key, default=None):
        getter = getattr(config, "get", None)
        if callable(getter):
            return getter(key, default)
        return default

    def _normalize_prefixes(self, raw_prefixes) -> list[str]:
        if isinstance(raw_prefixes, str):
            raw_prefixes = [raw_prefixes]
        elif not isinstance(raw_prefixes, (list, tuple, set)):
            return []

        return [str(prefix).strip() for prefix in raw_prefixes if str(prefix).strip()]

    def _event_has_result(self, event) -> bool:
        """检查事件是否已经被其他插件写入结果"""
        try:
            result = event.get_result()
        except Exception:
            return False

        if result is None:
            return False

        chain = getattr(result, "chain", None)
        if chain is None:
            return True

        return len(chain) > 0

    @filter.command("ai_enable")
    async def ai_enable(self, event, *args, **kwargs):
        """启用AI自动调用命令功能"""
        self.enabled = True
        yield event.plain_result("AI自动调用命令功能已启用")

    @filter.command("ai_disable")
    async def ai_disable(self, event, *args, **kwargs):
        """禁用AI自动调用命令功能"""
        self.enabled = False
        yield event.plain_result("AI自动调用命令功能已禁用")

    @filter.command("ai_status")
    async def ai_status(self, event, *args, **kwargs):
        """查看AI功能状态"""
        status = "启用" if self.enabled else "禁用"
        star_status = "可用" if STAR_AVAILABLE else "不可用"
        runtime_status = "支持" if self.runtime_supported else "当前版本不支持"
        yield event.plain_result(f"AI自动调用命令功能当前状态: {status}\nStar模块: {star_status}\n运行时支持: {runtime_status}\n唤醒词: {self.wake_word}\n监听模式: {self._get_listen_mode()}")

    @filter.command("refresh_commands")
    async def refresh_commands(self, event, *args, **kwargs):
        """刷新命令缓存"""
        self.command_cache.clear()
        yield event.plain_result("命令缓存已刷新")

    

    def _get_registered_commands(self) -> list[RegisteredCommand]:
        """从已启用插件的 handler 注册表构建可执行命令索引"""
        registered_commands = []
        seen_commands = set()

        try:
            all_stars_metadata = self.context.get_all_stars()
            all_stars_metadata = [
                star for star in all_stars_metadata
                if getattr(star, "activated", True) and not getattr(star, "disabled", False)
            ]
        except Exception as e:
            logger.error(f"获取插件列表失败: {e}")
            return {}
        
        if not all_stars_metadata:
            logger.warning("没有找到任何插件")
            return []

        for star in all_stars_metadata:
            plugin_name = getattr(star, "name", "未知插件")
            module_path = getattr(star, "module_path", None)

            if self._should_skip_plugin(plugin_name, module_path, star):
                continue

            if not plugin_name or not module_path:
                logger.warning(f"插件 '{plugin_name}' (模块: {module_path}) 的元数据无效或不完整，已跳过。")
                continue

            for handler in star_handlers_registry:
                if (
                    getattr(handler, "handler_module_path", None) != module_path
                    or not getattr(handler, "enabled", True)
                ):
                    continue

                event_filters = tuple(getattr(handler, "event_filters", ()) or ())
                for command_filter in event_filters:
                    if not isinstance(command_filter, (CommandFilter, CommandGroupFilter)):
                        continue

                    for command_name in self._get_complete_command_names(command_filter):
                        command_key = (
                            command_name,
                            getattr(handler, "handler_full_name", id(handler)),
                        )
                        if command_key in seen_commands:
                            continue
                        seen_commands.add(command_key)

                        registered_commands.append(
                            RegisteredCommand(
                                name=command_name,
                                description=getattr(handler, "desc", "") or "",
                                plugin_name=plugin_name,
                                handler=handler.handler,
                                command_filter=command_filter,
                                event_filters=event_filters,
                            )
                        )

        return registered_commands

    def _get_complete_command_names(self, command_filter) -> list[str]:
        """获取命令及其别名对应的完整名称"""
        get_complete_names = getattr(command_filter, "get_complete_command_names", None)
        if callable(get_complete_names):
            names = get_complete_names()
        elif isinstance(command_filter, CommandFilter):
            names = [
                f"{parent} {command}".strip()
                for parent in getattr(command_filter, "parent_command_names", [""])
                for command in [
                    getattr(command_filter, "command_name", ""),
                    *getattr(command_filter, "alias", set()),
                ]
            ]
        else:
            names = [
                getattr(command_filter, "group_name", ""),
                *getattr(command_filter, "alias", set()),
            ]

        return [
            " ".join(str(command_name).split())
            for command_name in names
            if str(command_name).strip()
        ]

    def _get_all_commands_info(
        self,
        registered_commands: list[RegisteredCommand] | None = None,
    ) -> dict:
        """获取所有其他插件及其命令列表, 参考help插件的实现"""
        plugin_commands = collections.defaultdict(list)

        if registered_commands is None:
            registered_commands = self._get_registered_commands()

        for command in registered_commands:
            formatted_command = (
                f"{command.name}#{command.description}"
                if command.description
                else command.name
            )
            if formatted_command not in plugin_commands[command.plugin_name]:
                plugin_commands[command.plugin_name].append(formatted_command)

        custom_commands = self._get_custom_commands_info()
        if custom_commands:
            plugin_commands["自定义命令"].extend(custom_commands)

        return dict(plugin_commands)

    def _should_skip_plugin(self, plugin_name: str, module_path: str, star) -> bool:
        """判断插件是否应该从命令发现中排除"""
        normalized_module_path = (module_path or "").replace("\\", ".")
        star_instance = getattr(star, "star_cls", None)

        if plugin_name in self.plugin_blacklist:
            return True

        if plugin_name in {"command2llm", "astrbot_plugin_command2llm"}:
            return True

        if normalized_module_path == __name__ or star_instance is self:
            return True

        if not getattr(star, "activated", True) or getattr(star, "disabled", False):
            return True

        builtin_flag = getattr(star, "builtin", None)
        reserved_flag = getattr(star, "reserved", False)
        if not self.show_builtin_cmds and (builtin_flag is True or reserved_flag):
            return True

        if not self.show_builtin_cmds and plugin_name in {"astrbot", "astrbot-reminder"}:
            return True

        return False

    def _get_custom_commands_info(self) -> list:
        """解析用户补充的自定义命令"""
        commands = []

        for item in self.custom_cmds:
            raw = str(item).strip()
            if not raw:
                continue

            command_part, separator, description = raw.partition(":")
            if not separator:
                command_part, separator, description = raw.partition("#")

            command_name = command_part.strip()
            description = description.strip()
            if not command_name:
                continue

            commands.append(f"{command_name}#{description}" if description else command_name)

        return commands

    def _get_all_available_commands(
        self,
        registered_commands: list[RegisteredCommand] | None = None,
    ) -> list:
        """获取所有可用命令列表"""
        try:
            commands_info = self._get_all_commands_info(registered_commands)
            commands = []
            for cmd_list in commands_info.values():
                for cmd in cmd_list:
                    # 提取命令名（去掉描述部分）
                    command_name = cmd.split('#')[0].strip()
                    if command_name and command_name not in commands:
                        commands.append(command_name)
            return commands
        except Exception as e:
            logger.error(f"获取命令列表失败: {str(e)}")
            return []

    

    async def terminate(self):
        """插件销毁方法"""
        logger.info("Command2LLM插件已卸载")


class ExecuteCommandTool(FunctionTool[AstrAgentContext]):
    """执行插件命令的工具"""
    
    def __init__(
        self,
        commands: list[RegisteredCommand],
        config,
        wake_word: str = "/",
    ):
        self.commands = sorted(commands, key=lambda item: len(item.name), reverse=True)
        self.config = config
        self.wake_word = wake_word
        self.called = False
        self.executed = False
        self.completed = False
        self.sent_count = 0
        self.last_command = ""
        self.name = "execute_command"
        self.description = "执行指定的插件命令"
        self.parameters = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的命令，包括命令名称和参数，例如：'今天吃什么' 或 '群分析 7'"
                }
            },
            "required": ["command"]
        }

    async def call(self, context: ContextWrapper[AstrAgentContext], **kwargs) -> ToolExecResult:
        self.called = True

        try:
            command = self._normalize_command(kwargs.get("command", ""))
            if not command:
                return "命令不能为空"

            self.last_command = command
            event = context.context.event
            registered_command = self._find_registered_command(command)
            if registered_command is None:
                return f"未找到可同步执行的命令: {command}"

            try:
                command_event = self._create_command_event(event, command)
                filter_error = await self._validate_event_filters(
                    registered_command,
                    command_event,
                )
                if filter_error:
                    return filter_error

                parsed_params = self._get_parsed_params(command_event)
                sent_before = self.sent_count
                sent_result_ids = set()

                async def forward_result(result):
                    return await self._send_result(event, result, sent_result_ids)

                command_event.send = forward_result
                self.executed = True

                logger.info(
                    f"通过注册处理器执行命令: {registered_command.plugin_name}/"
                    f"{registered_command.name}"
                )
                handler_result = registered_command.handler(
                    command_event,
                    **parsed_params,
                )
                await self._consume_handler_result(
                    event,
                    handler_result,
                    sent_result_ids,
                )
                await self._send_event_result(
                    event,
                    command_event,
                    sent_result_ids,
                )

                self.completed = True
                sent_count = self.sent_count - sent_before
                if sent_count:
                    return f"命令 '{command}' 已执行并发送 {sent_count} 条结果"
                return f"命令 '{command}' 已执行完成，但没有产生可发送结果"
            except Exception as e:
                logger.error(f"执行命令时出错: {str(e)}")
                return f"执行命令失败: {str(e)}"

        except Exception as e:
            logger.error(f"工具调用失败: {str(e)}")
            return f"工具调用失败: {str(e)}"

    def _normalize_command(self, command: str) -> str:
        command = str(command or "").strip()
        if self.wake_word and command.startswith(self.wake_word):
            command = command[len(self.wake_word):].strip()
        return " ".join(command.split())

    def _find_registered_command(self, command: str) -> RegisteredCommand | None:
        for registered_command in self.commands:
            if (
                command == registered_command.name
                or command.startswith(f"{registered_command.name} ")
            ):
                return registered_command
        return None

    def _create_command_event(self, event, command: str):
        message_obj = copy.copy(event.message_obj)
        message_obj.message_str = command
        message_obj.message = [Plain(text=command)]
        message_obj.session_id = event.session_id

        event_class = event.__class__
        event_kwargs = {
            "message_str": command,
            "message_obj": message_obj,
            "platform_meta": event.platform_meta,
            "session_id": event.session_id,
        }

        signature = inspect.signature(event_class.__init__)
        if "bot" in signature.parameters and hasattr(event, "bot"):
            event_kwargs["bot"] = event.bot

        command_event = event_class(**event_kwargs)
        command_event.role = getattr(event, "role", "member")
        command_event.is_wake = True
        command_event.is_at_or_wake_command = True

        if hasattr(event, "plugins_name"):
            command_event.plugins_name = event.plugins_name
        if hasattr(event, "_extras"):
            command_event._extras = copy.copy(event._extras)

        return command_event

    async def _validate_event_filters(
        self,
        registered_command: RegisteredCommand,
        command_event,
    ) -> str | None:
        try:
            for event_filter in registered_command.event_filters:
                filter_method = getattr(event_filter, "filter", None)
                if not callable(filter_method):
                    continue

                passed = filter_method(command_event, self.config)
                if inspect.isawaitable(passed):
                    passed = await passed
                if not passed:
                    return (
                        f"当前会话不满足命令 '{registered_command.name}' "
                        "的执行条件"
                    )
        except ValueError as e:
            return f"命令参数错误: {str(e)}"

        return None

    def _get_parsed_params(self, command_event) -> dict:
        get_extra = getattr(command_event, "get_extra", None)
        if callable(get_extra):
            return get_extra("parsed_params") or {}
        return getattr(command_event, "_extras", {}).get("parsed_params", {})

    async def _consume_handler_result(
        self,
        event,
        handler_result,
        sent_result_ids: set[int],
    ) -> None:
        if inspect.isasyncgen(handler_result):
            async for result in handler_result:
                await self._send_result(event, result, sent_result_ids)
        elif inspect.isawaitable(handler_result):
            result = await handler_result
            await self._send_result(event, result, sent_result_ids)
        else:
            await self._send_result(event, handler_result, sent_result_ids)

    async def _send_event_result(
        self,
        event,
        command_event,
        sent_result_ids: set[int],
    ) -> None:
        get_result = getattr(command_event, "get_result", None)
        if not callable(get_result):
            return

        result = get_result()
        if result is not None and id(result) not in sent_result_ids:
            await self._send_result(event, result, sent_result_ids)

    async def _send_result(
        self,
        event,
        result,
        sent_result_ids: set[int],
    ):
        if result is None:
            return None

        if isinstance(result, str):
            result = event.plain_result(result)

        send_result = await event.send(result)
        sent_result_ids.add(id(result))
        self.sent_count += 1
        return send_result
