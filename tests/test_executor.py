import asyncio
import copy
import unittest

from load_plugin import CommandFilter, load_plugin_module


plugin = load_plugin_module()


class ParsingFilter(CommandFilter):
    def filter(self, event, config):
        _, _, tags = event.message_str.partition(" ")
        event._extras["parsed_params"] = {"tags": tags}
        return True


class Message:
    def __init__(self):
        self.message_str = ""
        self.message = []
        self.session_id = "session"
        self.sender = type("Sender", (), {"user_id": "user"})()
        self.self_id = "bot"


class Event:
    sent = None

    def __init__(self, message_str, message_obj, platform_meta, session_id):
        self.message_str = message_str
        self.message_obj = message_obj
        self.platform_meta = platform_meta
        self.session_id = session_id
        self._extras = {}
        self.role = "member"
        self.sent = []
        self.unified_msg_origin = "umo"
        self.is_at_or_wake_command = True
        self.stopped = False

    async def send(self, result):
        self.sent.append(result)

    def plain_result(self, text):
        return plugin.Plain(text=text)

    def get_extra(self, key):
        return self._extras.get(key)

    def get_result(self):
        return None

    def stop_event(self):
        self.stopped = True


def make_event():
    return Event(
        message_str="爱弥斯 pixiv",
        message_obj=copy.copy(Message()),
        platform_meta={"name": "test"},
        session_id="session",
    )


def make_command(plugin_name, handler):
    command_filter = ParsingFilter()
    return plugin.RegisteredCommand(
        name="pixiv",
        description="按标签搜索插画",
        usage="pixiv [tags]",
        plugin_name=plugin_name,
        handler=handler,
        command_filter=command_filter,
        event_filters=(command_filter,),
    )


class ExecuteCommandTests(unittest.TestCase):
    def test_message_flow_routes_and_executes_without_agent_tool_call(self):
        class Response:
            def __init__(self, text):
                self.completion_text = text

        class Context:
            def __init__(self):
                self.responses = iter(
                    [
                        Response("是"),
                        Response('{"command_id": 0, "arguments": "爱弥斯"}'),
                    ]
                )

            async def get_current_chat_provider_id(self, **kwargs):
                return "provider"

            async def llm_generate(self, **kwargs):
                return next(self.responses)

            def get_config(self):
                return {}

        async def handler(event, tags=""):
            yield f"image:{tags}"

        context = Context()
        command_plugin = plugin.Command2LLMPlugin(
            context,
            {"wake_word": "/", "listen_mode": "llm_triggered_only"},
        )
        command = make_command("pixiv-search-plugin", handler)
        command_plugin._get_registered_commands = lambda: [command]
        event = make_event()

        asyncio.run(command_plugin.handle_message(event))

        self.assertEqual([item.text for item in event.sent], ["image:爱弥斯"])
        self.assertTrue(event.stopped)

    def test_router_returns_the_exact_selected_handler(self):
        class Response:
            completion_text = '{"command_id": 1, "arguments": "爱弥斯"}'

        class Context:
            def __init__(self):
                self.prompt = ""

            async def llm_generate(self, **kwargs):
                self.prompt = kwargs["prompt"]
                return Response()

        async def handler(event, tags=""):
            yield tags

        context = Context()
        command_plugin = object.__new__(plugin.Command2LLMPlugin)
        command_plugin.context = context
        first = make_command("random-image-plugin", handler)
        second = make_command("pixiv-search-plugin", handler)

        selection = asyncio.run(
            command_plugin._select_command(
                "爱弥斯 pixiv",
                "provider",
                [first, second],
            )
        )

        self.assertIs(selection[0], second)
        self.assertEqual(selection[1], "pixiv 爱弥斯")
        self.assertIn('"plugin":"pixiv-search-plugin"', context.prompt)
        self.assertIn('"usage":"pixiv [tags]"', context.prompt)

    def test_executes_selected_handler_when_command_names_conflict(self):
        calls = []

        async def wrong_handler(event, tags=""):
            calls.append(("wrong", tags))
            yield "wrong"

        async def selected_handler(event, tags=""):
            calls.append(("selected", tags))
            yield f"selected:{tags}"

        wrong = make_command("random-image-plugin", wrong_handler)
        selected = make_command("pixiv-search-plugin", selected_handler)
        executor = plugin.ExecuteCommandTool([wrong, selected], config={})
        event = make_event()

        result = asyncio.run(
            executor.execute(event, "pixiv 爱弥斯", selected)
        )

        self.assertEqual(calls, [("selected", "爱弥斯")])
        self.assertEqual([item.text for item in event.sent], ["selected:爱弥斯"])
        self.assertTrue(executor.completed)
        self.assertEqual(executor.sent_count, 1)
        self.assertIn("已执行并发送 1 条结果", result)


if __name__ == "__main__":
    unittest.main()
