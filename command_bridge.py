import copy
import hashlib
import inspect
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from astrbot.api.message_components import Plain
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult, ToolSet
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.core.star.filter.command import CommandFilter
from astrbot.core.star.star_handler import star_handlers_registry


TOOL_TYPE_COMMAND = "command"
TOOL_TYPE_NATIVE = "native_tool"
SUPPORTED_TOOL_TYPES = {TOOL_TYPE_COMMAND, TOOL_TYPE_NATIVE}


@dataclass(frozen=True)
class RegisteredCommand:
    name: str
    aliases: tuple[str, ...]
    description: str
    usage: str
    plugin_name: str
    plugin_display_name: str
    plugin_description: str
    module_path: str
    handler_full_name: str
    handler: Any
    command_filter: CommandFilter
    event_filters: tuple[Any, ...]

    def to_catalog_entry(self, command_id: int) -> dict[str, Any]:
        return {
            "command_id": command_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "usage": self.usage,
            "description": self.description,
            "plugin": self.plugin_name,
            "plugin_display_name": self.plugin_display_name,
            "plugin_description": self.plugin_description,
        }


class CommandRegistry:
    """Discover executable commands and resolve their owning plugins."""

    def __init__(
        self,
        context,
        *,
        plugin_blacklist: set[str] | None = None,
        show_builtin_cmds: bool = False,
    ) -> None:
        self.context = context
        self.plugin_blacklist = {
            str(item).strip()
            for item in (plugin_blacklist or set())
            if str(item).strip()
        }
        self.show_builtin_cmds = show_builtin_cmds

    def get_stars(self) -> list[Any]:
        try:
            return list(self.context.get_all_stars() or [])
        except Exception as exc:
            logger.error(f"获取插件列表失败: {exc}")
            return []

    def effective_blacklist(self) -> list[str]:
        effective = []
        stars = self.get_stars()
        for configured_name in sorted(self.plugin_blacklist):
            matched = next(
                (
                    star
                    for star in stars
                    if self._blacklist_entry_matches_star(configured_name, star)
                ),
                None,
            )
            effective.append(
                str(getattr(matched, "name", "") or configured_name)
            )
        return effective

    def discover_commands(self, event=None) -> list[RegisteredCommand]:
        commands: list[RegisteredCommand] = []
        seen_handlers: set[tuple[str, str]] = set()

        for star in self.get_stars():
            if not self._is_command_plugin_allowed(star, event):
                continue

            plugin_name = str(getattr(star, "name", "") or "").strip()
            module_path = str(getattr(star, "module_path", "") or "").strip()
            if not plugin_name or not module_path:
                continue

            for handler in star_handlers_registry:
                if (
                    getattr(handler, "handler_module_path", None) != module_path
                    or not getattr(handler, "enabled", True)
                ):
                    continue

                event_filters = tuple(getattr(handler, "event_filters", ()) or ())
                for command_filter in event_filters:
                    if not isinstance(command_filter, CommandFilter):
                        continue

                    command_names = self._get_complete_command_names(command_filter)
                    if not command_names:
                        continue

                    handler_full_name = str(
                        getattr(handler, "handler_full_name", "") or id(handler)
                    )
                    command_key = (handler_full_name, command_names[0])
                    if command_key in seen_handlers:
                        continue
                    seen_handlers.add(command_key)

                    commands.append(
                        RegisteredCommand(
                            name=command_names[0],
                            aliases=tuple(command_names[1:]),
                            description=self._get_handler_description(handler),
                            usage=self._get_handler_usage(
                                command_names[0],
                                handler.handler,
                            ),
                            plugin_name=plugin_name,
                            plugin_display_name=str(
                                getattr(star, "display_name", "") or plugin_name
                            ),
                            plugin_description=self._get_plugin_description(star),
                            module_path=module_path,
                            handler_full_name=handler_full_name,
                            handler=handler.handler,
                            command_filter=command_filter,
                            event_filters=event_filters,
                        )
                    )

        return commands

    def get_native_capabilities(self, event=None) -> list[dict[str, str]]:
        get_manager = getattr(self.context, "get_llm_tool_manager", None)
        if not callable(get_manager):
            return []

        tool_manager = get_manager()
        get_full_tool_set = getattr(tool_manager, "get_full_tool_set", None)
        if not callable(get_full_tool_set):
            return []

        stars_by_module = self._stars_by_module()
        capabilities = []
        for tool in getattr(get_full_tool_set(), "tools", ()) or ():
            star = stars_by_module.get(getattr(tool, "handler_module_path", None))
            if (
                star is None
                or getattr(star, "reserved", False)
                or not self._is_native_plugin_allowed(star, event)
            ):
                continue
            capabilities.append(
                {
                    "type": TOOL_TYPE_NATIVE,
                    "name": str(getattr(tool, "name", "") or ""),
                    "description": str(getattr(tool, "description", "") or "")[:500],
                    "plugin": str(getattr(star, "name", "") or ""),
                    "plugin_display_name": str(
                        getattr(star, "display_name", "")
                        or getattr(star, "name", "")
                        or ""
                    ),
                    "plugin_description": self._get_plugin_description(star),
                }
            )
        return capabilities

    def filter_request_tools(
        self,
        tool_set: ToolSet | None,
        *,
        include_native_tools: bool,
        limiter: "PluginToolCallLimiter | None" = None,
        event=None,
    ) -> ToolSet:
        filtered = ToolSet()
        stars_by_module = self._stars_by_module()

        for tool in getattr(tool_set, "tools", ()) or ():
            star = stars_by_module.get(getattr(tool, "handler_module_path", None))
            if star is None or getattr(star, "reserved", False):
                filtered.add_tool(tool)
                continue
            if include_native_tools and self._is_native_plugin_allowed(star, event):
                filtered.add_tool(
                    LimitedPluginFunctionTool(tool, limiter)
                    if limiter and limiter.enabled
                    else tool
                )

        return filtered

    def _stars_by_module(self) -> dict[str, Any]:
        return {
            str(getattr(star, "module_path", "")): star
            for star in self.get_stars()
            if getattr(star, "module_path", None)
        }

    def _is_command_plugin_allowed(self, star, event) -> bool:
        if not self._is_active_and_allowed(star, event):
            return False
        if self._is_command2llm(star):
            return False
        if not self.show_builtin_cmds and (
            getattr(star, "reserved", False)
            or getattr(star, "builtin", False)
            or getattr(star, "name", None) in {"astrbot", "astrbot-reminder"}
        ):
            return False
        return True

    def _is_native_plugin_allowed(self, star, event) -> bool:
        if getattr(star, "reserved", False):
            return True
        return self._is_active_and_allowed(star, event) and not self._is_command2llm(star)

    def _is_active_and_allowed(self, star, event) -> bool:
        if not getattr(star, "activated", True) or getattr(star, "disabled", False):
            return False
        if "*" in self.plugin_blacklist:
            return False
        if any(
            self._blacklist_entry_matches_star(configured_name, star)
            for configured_name in self.plugin_blacklist
        ):
            return False

        selected_plugins = getattr(event, "plugins_name", None) if event else None
        if selected_plugins is not None and "*" not in selected_plugins:
            if getattr(star, "reserved", False):
                return True
            return getattr(star, "name", None) in selected_plugins
        return True

    def _is_command2llm(self, star) -> bool:
        identifiers = self._star_identifiers(star)
        return bool({"command2llm", "astrbot_plugin_command2llm"} & identifiers)

    def _star_identifiers(self, star) -> set[str]:
        return {
            str(value).strip()
            for value in (
                getattr(star, "name", ""),
                getattr(star, "display_name", ""),
                getattr(star, "root_dir_name", ""),
            )
            if str(value or "").strip()
        }

    def _blacklist_entry_matches_star(self, configured_name: str, star) -> bool:
        identifiers = self._star_identifiers(star)
        if configured_name in identifiers:
            return True

        configured_key = self._normalize_identifier(configured_name)
        if len(configured_key) < 4:
            return False
        for identifier in identifiers:
            identifier_key = self._normalize_identifier(identifier)
            if len(identifier_key) < 4:
                continue
            if configured_key.startswith(identifier_key) or identifier_key.startswith(
                configured_key
            ):
                return True
        return False

    def _normalize_identifier(self, value: str) -> str:
        normalized = "".join(
            character.lower()
            for character in str(value or "")
            if character.isalnum()
        )
        prefix = "astrbotplugin"
        return normalized[len(prefix) :] if normalized.startswith(prefix) else normalized

    def _get_plugin_description(self, star) -> str:
        description = (
            getattr(star, "short_desc", "")
            or getattr(star, "desc", "")
            or ""
        )
        return " ".join(str(description).split())[:500]

    def _get_handler_description(self, handler) -> str:
        description = getattr(handler, "desc", "") or inspect.getdoc(handler.handler) or ""
        return " ".join(str(description).split())[:500]

    def _get_handler_usage(self, command_name: str, handler) -> str:
        parameters = []
        try:
            signature_parameters = inspect.signature(handler).parameters.values()
        except (TypeError, ValueError):
            return command_name

        for parameter in signature_parameters:
            if parameter.name in {"self", "event"}:
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                continue
            if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
                parameters.append(f"[{parameter.name}...]")
            elif parameter.default is inspect.Parameter.empty:
                parameters.append(f"<{parameter.name}>")
            else:
                parameters.append(f"[{parameter.name}]")
        return " ".join([command_name, *parameters])

    def _get_complete_command_names(self, command_filter: CommandFilter) -> list[str]:
        get_complete_names = getattr(command_filter, "get_complete_command_names", None)
        if callable(get_complete_names):
            names = get_complete_names()
        else:
            names = [
                f"{parent} {command}".strip()
                for command in [
                    getattr(command_filter, "command_name", ""),
                    *getattr(command_filter, "alias", set()),
                ]
                for parent in getattr(command_filter, "parent_command_names", [""])
            ]

        normalized_names = []
        for command_name in names:
            normalized = " ".join(str(command_name).split())
            if normalized and normalized not in normalized_names:
                normalized_names.append(normalized)
        return normalized_names


class ExecuteCommandTool:
    """Execute an exact registered command handler in the source event context."""

    def __init__(
        self,
        commands: list[RegisteredCommand],
        config,
        wake_word: str = "/",
    ) -> None:
        self.commands = sorted(commands, key=lambda item: len(item.name), reverse=True)
        self.config = config
        self.wake_word = wake_word
        self.called = False
        self.executed = False
        self.completed = False
        self.sent_count = 0
        self.last_command = ""

    async def execute(
        self,
        event,
        command: str,
        registered_command: RegisteredCommand | None = None,
        *,
        agent_mode: bool = False,
    ) -> str:
        self.called = True
        command = self._normalize_command(command)
        if not command:
            return "命令不能为空"

        self.last_command = command
        registered_command = registered_command or self._find_registered_command(command)
        if registered_command is None:
            return f"未找到可同步执行的命令: {command}"
        if not self._matches_command(command, registered_command):
            return f"命令路由结果与注册命令不匹配: {command}"

        try:
            command_event = self._create_command_event(event, command)
            filter_error = await self._validate_event_filters(
                registered_command,
                command_event,
            )
            if filter_error:
                return filter_error

            parsed_params = self._get_parsed_params(command_event)
            sent_result_ids: set[int] = set()
            tool_messages: list[str] = []

            async def capture_result(result):
                return await self._capture_result(
                    event,
                    result,
                    sent_result_ids,
                    tool_messages,
                    agent_mode=agent_mode,
                )

            command_event.send = capture_result
            self.executed = True
            logger.info(
                f"通过注册处理器执行命令: {registered_command.plugin_name}/"
                f"{registered_command.name}"
            )
            handler_result = registered_command.handler(
                command_event,
                **parsed_params,
            )
            await self._consume_handler_result(handler_result, capture_result)
            await self._capture_event_result(command_event, capture_result, sent_result_ids)
            self.completed = True

            if agent_mode:
                if tool_messages:
                    return "\n\n".join(tool_messages)
                return f"命令“{command}”执行完成，但没有返回内容。"
            if self.sent_count:
                return f"命令 '{command}' 已执行并发送 {self.sent_count} 条结果"
            return f"命令 '{command}' 已执行完成，但没有产生可发送结果"
        except Exception as exc:
            logger.error(f"执行命令时出错: {exc}")
            return f"执行命令失败: {exc}"

    def _normalize_command(self, command: str) -> str:
        command = str(command or "").strip()
        if self.wake_word and command.startswith(self.wake_word):
            command = command[len(self.wake_word) :].strip()
        return " ".join(command.split())

    def _find_registered_command(self, command: str) -> RegisteredCommand | None:
        for registered_command in self.commands:
            if self._matches_command(command, registered_command):
                return registered_command
        return None

    def _matches_command(
        self,
        command: str,
        registered_command: RegisteredCommand,
    ) -> bool:
        names = (registered_command.name, *registered_command.aliases)
        return any(command == name or command.startswith(f"{name} ") for name in names)

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
                    return f"当前会话不满足命令 '{registered_command.name}' 的执行条件"
        except ValueError as exc:
            return f"命令参数错误: {exc}"
        return None

    def _get_parsed_params(self, command_event) -> dict[str, Any]:
        get_extra = getattr(command_event, "get_extra", None)
        if callable(get_extra):
            return get_extra("parsed_params") or {}
        return getattr(command_event, "_extras", {}).get("parsed_params", {})

    async def _consume_handler_result(self, handler_result, capture_result) -> None:
        if inspect.isasyncgen(handler_result):
            async for result in handler_result:
                await capture_result(result)
        elif inspect.isawaitable(handler_result):
            await capture_result(await handler_result)
        else:
            await capture_result(handler_result)

    async def _capture_event_result(
        self,
        command_event,
        capture_result,
        sent_result_ids: set[int],
    ) -> None:
        get_result = getattr(command_event, "get_result", None)
        if not callable(get_result):
            return
        result = get_result()
        if result is not None and id(result) not in sent_result_ids:
            await capture_result(result)

    async def _capture_result(
        self,
        event,
        result,
        sent_result_ids: set[int],
        tool_messages: list[str],
        *,
        agent_mode: bool,
    ):
        if result is None or id(result) in sent_result_ids:
            return None
        if isinstance(result, str):
            result = event.plain_result(result)

        sent_result_ids.add(id(result))
        text, rich = self._describe_result(result)
        if not agent_mode or rich:
            send_result = await event.send(result)
            self.sent_count += 1
        else:
            send_result = None

        if agent_mode:
            if rich:
                tool_messages.append(
                    "已向用户发送富媒体结果，避免重复转述完整内容。"
                    + (f" 结果摘要：{text}" if text else "")
                )
            elif text:
                tool_messages.append(text)
        return send_result

    def _describe_result(self, result) -> tuple[str, bool]:
        chain = getattr(result, "chain", None)
        if chain is None:
            return str(result), False

        rich = any(not isinstance(component, Plain) for component in chain)
        get_plain_text = getattr(result, "get_plain_text", None)
        if callable(get_plain_text):
            try:
                return str(get_plain_text(with_other_comps_mark=True) or ""), rich
            except TypeError:
                return str(get_plain_text() or ""), rich

        texts = []
        for component in chain:
            if isinstance(component, Plain):
                texts.append(str(getattr(component, "text", "")))
            else:
                texts.append(f"[{component.__class__.__name__}]")
        return " ".join(filter(None, texts)), rich


class PluginToolCallLimiter:
    """Enforce per-response plugin tool call and tool-round limits."""

    def __init__(self, max_calls_per_round: int, max_call_rounds: int) -> None:
        self.max_calls_per_round = max_calls_per_round
        self.max_call_rounds = max_call_rounds
        self._round_marker: int | None = None
        self._round_count = 0
        self._calls_in_round = 0

    @property
    def enabled(self) -> bool:
        return self.max_calls_per_round >= 0 or self.max_call_rounds >= 0

    def begin_call(
        self,
        context: ContextWrapper[AstrAgentContext],
    ) -> tuple[bool, str, bool]:
        marker = len(context.messages)
        if marker != self._round_marker:
            if self.max_call_rounds >= 0 and self._round_count >= self.max_call_rounds:
                return (
                    False,
                    "Command2LLM 已达到插件工具最大调用轮数，"
                    "本次调用未执行。请停止调用插件工具并生成最终回复。",
                    False,
                )
            self._round_marker = marker
            self._round_count += 1
            self._calls_in_round = 0

        if (
            self.max_calls_per_round >= 0
            and self._calls_in_round >= self.max_calls_per_round
        ):
            return (
                False,
                "Command2LLM 已达到本轮插件工具调用上限，"
                "本次调用未执行。请在下一轮再调用必要工具，或直接生成最终回复。",
                False,
            )

        self._calls_in_round += 1
        final_round = (
            self.max_call_rounds >= 0
            and self._round_count >= self.max_call_rounds
        )
        return True, "", final_round


class LimitedPluginFunctionTool(FunctionTool[AstrAgentContext]):
    """Keep a plugin FunctionTool's behavior while adding call limits."""

    def __init__(
        self,
        original: FunctionTool[AstrAgentContext],
        limiter: PluginToolCallLimiter,
    ) -> None:
        self.original = original
        self.limiter = limiter
        self.name = original.name
        self.description = original.description
        self.parameters = original.parameters
        self.handler = None
        self.handler_module_path = original.handler_module_path
        self.active = original.active
        self.is_background_task = original.is_background_task

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ):
        allowed, error, final_round = self.limiter.begin_call(context)
        if not allowed:
            yield error
            return

        result = self._invoke_original(context, kwargs)
        if inspect.isasyncgen(result):
            async for item in result:
                yield item
        elif inspect.isawaitable(result):
            yield await result
        else:
            yield result

        if final_round:
            yield (
                "Command2LLM 提示：已到达插件工具最大调用轮数，"
                "请基于已有结果生成最终回复，不要继续调用插件工具。"
            )

    def _invoke_original(
        self,
        context: ContextWrapper[AstrAgentContext],
        kwargs: dict[str, Any],
    ):
        if self.original.handler:
            return self.original.handler(context.context.event, **kwargs)

        if self._overrides_call(self.original):
            return self.original.call(context, **kwargs)

        run = getattr(self.original, "run", None)
        if callable(run):
            return run(context.context.event, **kwargs)
        raise ValueError(f"工具 {self.original.name} 没有可执行处理器")

    @staticmethod
    def _overrides_call(tool: FunctionTool) -> bool:
        return any(
            "call" in tool_type.__dict__
            and tool_type.__dict__["call"] is not FunctionTool.call
            for tool_type in type(tool).mro()
        )


class PluginCommandFunctionTool(FunctionTool[AstrAgentContext]):
    """Expose one registered command as one native AstrBot function tool."""

    def __init__(self, context, command: RegisteredCommand, wake_word: str = "/"):
        self.context = context
        self.command = command
        self.wake_word = wake_word
        identity = f"{command.plugin_name}:{command.handler_full_name}:{command.name}"
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:12]
        self.name = f"plugin_command_{digest}"
        self.description = self._build_description(command)
        self.parameters = {
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "string",
                    "description": (
                        "只填写命令名后面的参数，不要包含命令名或命令前缀；"
                        "无参数时省略或传空字符串。"
                    ),
                }
            },
            "additionalProperties": False,
        }
        self.handler = None
        self.handler_module_path = command.module_path
        self.active = True
        self.is_background_task = False

    async def call(
        self,
        context: ContextWrapper[AstrAgentContext],
        **kwargs: Any,
    ) -> ToolExecResult:
        event = context.context.event
        config = self._get_session_config(event)
        executor = ExecuteCommandTool([self.command], config, self.wake_word)
        arguments = " ".join(str(kwargs.get("arguments", "") or "").split())
        command_line = " ".join(f"{self.command.name} {arguments}".split())
        return await executor.execute(
            event,
            command_line,
            self.command,
            agent_mode=True,
        )

    def _get_session_config(self, event):
        get_config = getattr(self.context, "get_config", None)
        if not callable(get_config):
            return {}
        try:
            return get_config(umo=event.unified_msg_origin)
        except TypeError:
            return get_config()

    def _build_description(self, command: RegisteredCommand) -> str:
        parts = [
            f"调用插件“{command.plugin_display_name}”({command.plugin_name})的命令“{command.name}”。",
        ]
        if command.description:
            parts.append(f"命令用途：{command.description}。")
        elif command.plugin_description:
            parts.append(f"插件用途：{command.plugin_description}。")
        parts.append(f"用法：{command.usage}。")
        if command.aliases:
            parts.append(f"别名：{', '.join(command.aliases)}。")
        return "".join(parts)[:1000]
