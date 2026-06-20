import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RouteDecision:
    command_id: int
    arguments: str


def _parse_json_object(response_text: str) -> dict:
    text = str(response_text or "").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("模型未返回 JSON 对象")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("模型返回值必须是 JSON 对象")
    return payload


def parse_route_decision(response_text: str, command_count: int) -> RouteDecision | None:
    """Parse the command router's JSON response."""
    payload = _parse_json_object(response_text)
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


def parse_agent_trigger_decision(response_text: str) -> bool:
    """Parse whether a non-LLM-triggered message should start the agent."""
    payload = _parse_json_object(response_text)
    use_agent = payload.get("use_agent")
    if not isinstance(use_agent, bool):
        raise ValueError("use_agent 必须是布尔值")
    return use_agent


def build_command_line(command_name: str, arguments: str) -> str:
    return " ".join(f"{command_name} {arguments}".split())
