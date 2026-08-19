import unittest

from analysis.pipeline.extract_events import normalize_claude, normalize_codex, normalize_plain
from analysis.pipeline.parse_claude import parse_claude_records
from analysis.pipeline.parse_codex import parse_codex_records
from analysis.pipeline.parse_trace import parse_plain_trace


class ParserTests(unittest.TestCase):
    def test_claude_warning_and_tool_use(self):
        text = 'WARNING: cuda\n{"type":"assistant","message":{"content":[{"type":"text","text":"plan"},{"type":"tool_use","name":"Bash","input":{"command":"python train.py"}}]}}\n{"type":"result","result":"done"}\n'
        records, stats = parse_claude_records(text)
        events = normalize_claude(records)
        self.assertEqual(stats["json_lines"], 2)
        self.assertEqual(events[1]["command"], "python train.py")
        self.assertEqual(events[-1]["event_type"], "result")

    def test_codex_items(self):
        text = '{"type":"item.completed","item":{"type":"reasoning","text":"think"}}\n{"type":"item.completed","item":{"type":"command_execution","command":"python evaluate.py","aggregated_output":"ok","exit_code":0,"status":"completed"}}\n'
        records, _ = parse_codex_records(text)
        events = normalize_codex(records)
        self.assertEqual(events[0]["role"], "assistant")
        self.assertEqual(events[1]["event_type"], "command_execution")
        self.assertEqual(events[1]["exit_code"], 0)

    def test_timestamp_prefix_and_plain_trace(self):
        records, _ = parse_claude_records('[12:00:00] {"type":"assistant","message":{"content":"hello"}}')
        self.assertEqual(len(records), 1)
        trace, _ = parse_plain_trace('Session start\nAssistant — turn 1\nTool call — Bash\npython train.py')
        events = normalize_plain(trace)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["parser_confidence"], "low")

    def test_empty_and_malformed(self):
        records, stats = parse_claude_records('{not-json}\n')
        self.assertEqual(records, [])
        self.assertEqual(stats["malformed_json_lines"], 1)

    def test_large_event_text_is_capped(self):
        records, _ = parse_claude_records('{"type":"assistant","message":{"content":"' + ('x' * 25000) + '"}}')
        event = normalize_claude(records)[0]
        self.assertLessEqual(len(event["text"]), 4020)
        self.assertEqual(event["truncated_fields"], ["text"])


if __name__ == "__main__":
    unittest.main()
