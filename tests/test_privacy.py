from __future__ import annotations

import json
import unittest

from src.openai_report import summarize_with_openai
from src.report import assert_aggregate_payload_safe


class _FakeResponse:
    output_text = json.dumps(
        {
            "title": "Aggregate report",
            "executive_summary": "No clinical conclusion is supported.",
            "key_findings": ["History was competitive."],
            "evidence_strength": "Exploratory.",
            "data_quality_flags": ["Small cohort."],
            "limitations": ["Participant count is limited."],
            "responsible_use": "Research use only.",
        }
    )
    _request_id = "req_test"
    status = "completed"


class _FakeResponses:
    def __init__(self) -> None:
        self.arguments = None

    def create(self, **kwargs):
        self.arguments = kwargs
        return _FakeResponse()


class _FakeClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _FakeIncompleteResponse:
    output_text = '{"title":"truncated'
    _request_id = "req_incomplete"
    status = "incomplete"
    incomplete_details = {"reason": "max_output_tokens"}


class _FakeMalformedResponse:
    output_text = '{"title":"truncated'
    _request_id = "req_malformed"
    status = "completed"


class _FakeSingleResponseClient:
    def __init__(self, response) -> None:
        self.responses = _FakeResponses()
        self.responses.create = lambda **kwargs: response


class PrivacyTests(unittest.TestCase):
    def test_aggregate_payload_rejects_participant_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "Participant-level field"):
            assert_aggregate_payload_safe({"participant_id": "P01"})

    def test_openai_summary_disables_storage_and_uses_aggregate_payload(self) -> None:
        client = _FakeClient()
        payload = {"benchmark": "CycleBench", "scores": [{"track": "history_only", "mae": 3.0}]}
        result = summarize_with_openai(payload, client=client)
        self.assertEqual(result["request_id"], "req_test")
        self.assertIs(client.responses.arguments["store"], False)
        self.assertEqual(json.loads(client.responses.arguments["input"]), payload)
        self.assertNotIn("participant_id", client.responses.arguments["input"])

    def test_openai_summary_reports_incomplete_response_before_json_parsing(self) -> None:
        payload = {"benchmark": "CycleBench"}
        client = _FakeSingleResponseClient(_FakeIncompleteResponse())
        with self.assertRaisesRegex(RuntimeError, "status=incomplete, reason=max_output_tokens"):
            summarize_with_openai(payload, client=client)

    def test_openai_summary_reports_malformed_structured_output(self) -> None:
        payload = {"benchmark": "CycleBench"}
        client = _FakeSingleResponseClient(_FakeMalformedResponse())
        with self.assertRaisesRegex(RuntimeError, "malformed structured output.*req_malformed"):
            summarize_with_openai(payload, client=client)


if __name__ == "__main__":
    unittest.main()
