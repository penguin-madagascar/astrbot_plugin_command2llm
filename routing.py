import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    command_id: int
    arguments: str


def parse_route_decision(response_text: str, command_count: int) -> RouteDecision | None:
    """Parse the command router's JSON response."""
    text = str(response_text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("路由模型未返回 JSON 对象")

    payload = json.loads(text[start:end + 1])
    command_id = payload.get("command_id")
    if command_id is None:
        return None

    if isinstance(command_id, str) and command_id.isdigit():
        command_id = int(command_id)
    if isinstance(command_id, bool) or not isinstance(command_id, int):
        raise ValueError("command_id 必须是整数或 null")
    if not 0 <= command_id < command_count:
        raise ValueError(f"command_id 超出范围: {command_id}")

    arguments = payload.get("arguments", "")
    if not isinstance(arguments, str):
        raise ValueError("arguments 必须是字符串")

    return RouteDecision(
        command_id=command_id,
        arguments=" ".join(arguments.split()),
    )


def build_command_line(command_name: str, arguments: str) -> str:
    return " ".join(f"{command_name} {arguments}".split())
