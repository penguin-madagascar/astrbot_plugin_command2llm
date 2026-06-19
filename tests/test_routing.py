import unittest

from routing import build_command_line, parse_route_decision


class RouteDecisionTests(unittest.TestCase):
    def test_parses_plain_json(self):
        decision = parse_route_decision(
            '{"command_id": 3, "arguments": "爱弥斯"}',
            command_count=4,
        )

        self.assertEqual(decision.command_id, 3)
        self.assertEqual(decision.arguments, "爱弥斯")

    def test_parses_fenced_json(self):
        decision = parse_route_decision(
            '```json\n{"command_id": "1", "arguments": "  北京   明天 "}\n```',
            command_count=2,
        )

        self.assertEqual(decision.command_id, 1)
        self.assertEqual(decision.arguments, "北京 明天")

    def test_null_command_means_no_route(self):
        decision = parse_route_decision(
            '{"command_id": null, "arguments": ""}',
            command_count=2,
        )

        self.assertIsNone(decision)

    def test_rejects_out_of_range_command(self):
        with self.assertRaisesRegex(ValueError, "超出范围"):
            parse_route_decision(
                '{"command_id": 2, "arguments": ""}',
                command_count=2,
            )

    def test_builds_canonical_command_line(self):
        self.assertEqual(build_command_line("pixiv", "爱弥斯"), "pixiv 爱弥斯")


if __name__ == "__main__":
    unittest.main()
