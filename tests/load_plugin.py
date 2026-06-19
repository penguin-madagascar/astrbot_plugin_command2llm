import importlib.util
import sys
import types
from pathlib import Path


class CommandFilter:
    pass


class CommandGroupFilter:
    pass


class Plain:
    def __init__(self, text):
        self.text = text


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def load_plugin_module():
    root = Path(__file__).resolve().parents[1]
    package_name = "command2llm_test_package"

    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package

    class FilterApi:
        class EventMessageType:
            ALL = "all"

        @staticmethod
        def event_message_type(*args, **kwargs):
            return lambda function: function

        @staticmethod
        def command(*args, **kwargs):
            return lambda function: function

    class Star:
        def __init__(self, context):
            self.context = context

    class Logger:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    def register(*args, **kwargs):
        return lambda cls: cls

    astrbot = _module("astrbot")
    api = _module("astrbot.api", logger=Logger())
    event = _module("astrbot.api.event", filter=FilterApi())
    star = _module(
        "astrbot.api.star",
        Context=object,
        Star=Star,
        register=register,
    )
    components = _module("astrbot.api.message_components", Plain=Plain)
    core = _module("astrbot.core")
    config = _module("astrbot.core.config", AstrBotConfig=dict)
    core_star = _module("astrbot.core.star")
    registry = _module(
        "astrbot.core.star.star_handler",
        star_handlers_registry=[],
    )
    filter_package = _module("astrbot.core.star.filter")
    command = _module(
        "astrbot.core.star.filter.command",
        CommandFilter=CommandFilter,
    )
    command_group = _module(
        "astrbot.core.star.filter.command_group",
        CommandGroupFilter=CommandGroupFilter,
    )

    astrbot.api = api
    astrbot.core = core
    api.event = event
    api.star = star
    api.message_components = components
    core.config = config
    core.star = core_star
    core_star.star_handler = registry
    core_star.filter = filter_package
    filter_package.command = command
    filter_package.command_group = command_group

    module_name = f"{package_name}.main"
    spec = importlib.util.spec_from_file_location(module_name, root / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
