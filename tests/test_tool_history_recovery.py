"""Regressions for malformed native arguments poisoning the next chat template."""

import copy
import json

import pytest
from test_engine import FakeModels, answer, call, run, task
from locaii.engine import Engine, clean_messages
from locaii.storage import canonical


class StrictTemplateModels(FakeModels):
    """Mirror the server's strict argument parsing before token measurement."""

    def count_tokens(self, messages, tools=None):
        for message in messages:
            for tool in message.get("tool_calls", []):
                value = json.loads(tool["function"]["arguments"])
                assert isinstance(value, dict)
                json.dumps(value, allow_nan=False)
        return super().count_tokens(messages, tools)


@pytest.mark.parametrize(
    "raw",
    ['{"path":"large.py","content":"unfinished', "[1,2]", "null", '{"a":NaN}', ""],
)
def test_prompt_normalization_is_strict_bounded_and_preserves_source(raw):
    broken = call("write_file", {}, "broken")
    broken["function"]["arguments"] = raw
    messages = [
        answer("Previous attempt", [broken])["message"],
        {
            "role": "tool",
            "tool_call_id": "broken",
            "content": '{"error":"Invalid JSON; not executed"}',
        },
        {"role": "user", "content": "Continue"},
    ]
    original = copy.deepcopy(messages)
    cleaned = clean_messages(messages)
    marker = json.loads(cleaned[0]["tool_calls"][0]["function"]["arguments"])
    assert marker["_locaii_invalid_historical_arguments"] is True
    assert "Do not execute" in marker["notice"]
    assert len(marker["untrusted_excerpt"]) <= 400
    assert cleaned[1:] == messages[1:]
    assert messages == original
    assert clean_messages(cleaned) == cleaned


def test_mixed_valid_and_broken_calls_retain_receipts_and_pairing():
    broken = call("write_file", {}, "bad")
    broken["function"]["arguments"] = '{"content":"' + ("x" * 12000)
    messages = [
        answer("", [call("read_text_file", {"path": "input.txt"}, "ok"), broken])[
            "message"
        ],
        {"role": "tool", "tool_call_id": "ok", "content": "Observed source evidence"},
        {
            "role": "tool",
            "tool_call_id": "bad",
            "content": "Invalid arguments; nothing executed",
        },
    ]
    result = clean_messages(messages)
    assert json.loads(result[0]["tool_calls"][0]["function"]["arguments"]) == {
        "path": "input.txt"
    }
    assert len(canonical(result)) < 2000
    assert result[1:] == messages[1:]


def test_live_truncation_recovers_before_next_template_render(task):
    store, identifier, workspace = task
    broken = call("write_file", {}, "bad")
    broken["function"]["arguments"] = '{"path":"wrong.py","content":"unterminated'

    def repair(messages, kwargs):
        assert "smaller tool call" in canonical(messages)
        assert "_locaii_invalid_historical_arguments" in canonical(messages)
        return answer(
            "",
            [
                call(
                    "write_file",
                    {"path": "derived.txt", "content": "RECOVERED"},
                    "fixed",
                )
            ],
            "tool_calls",
        )

    models = StrictTemplateModels(
        [
            answer("Writing a long file", [broken], "length"),
            repair,
            answer(
                "",
                [call("read_text_file", {"path": "derived.txt"}, "read")],
                "tool_calls",
            ),
            answer("Verified RECOVERED."),
        ]
    )
    result = run(
        Engine(store, models), identifier, "Write and verify a small derived file."
    )
    assert result["status"] == "completed"
    assert (workspace / "derived.txt").read_text() == "RECOVERED"
    assert not (workspace / "wrong.py").exists()
    saved = store.messages(identifier)
    assert (
        saved[1]["tool_calls"][0]["function"]["arguments"]
        == broken["function"]["arguments"]
    )
    assert any(e["type"] == "tool_rejected" for e in store.events(identifier))
    assert not any(
        e["type"] == "tool_dispatched" and e["call_id"] == "bad"
        for e in store.events(identifier)
    )


def test_continue_recovers_saved_invalid_history_without_rewriting_it(task):
    store, identifier, workspace = task
    store.append(identifier, "user", "Read the synthetic input and report the code.")
    bad = call("write_file", {}, "bad")
    bad["function"]["arguments"] = '{"path":"forbidden.py","content":"unfinished'
    store.append(identifier, answer("Old broken attempt", [bad])["message"])
    store.append(
        identifier,
        "tool",
        '{"error":"Invalid JSON. Not executed."}',
        tool_call_id="bad",
    )
    original = copy.deepcopy(store.messages(identifier))
    (workspace / "input.txt").write_text("READ_ONLY_RECOVERY")
    models = StrictTemplateModels(
        [
            answer(
                "",
                [call("read_text_file", {"path": "input.txt"}, "read")],
                "tool_calls",
            ),
            answer("READ_ONLY_RECOVERY"),
        ]
    )
    result = run(
        Engine(store, models),
        identifier,
        "Continue. Read input.txt only; do not write.",
    )
    assert result["status"] == "completed" and result["text"] == "READ_ONLY_RECOVERY"
    assert store.messages(identifier)[: len(original)] == original
    assert not (workspace / "forbidden.py").exists()


def test_checkpoint_does_not_describe_rejected_write_as_completed():
    from locaii.engine import observation_ledger

    messages = [
        answer(
            "", [call("write_file", {"path": "never.txt", "content": "old"}, "skipped")]
        )["message"],
        {
            "role": "tool",
            "tool_call_id": "skipped",
            "content": '{"status":"not_executed","not_executed":true}',
        },
    ]
    assert "wrote" not in observation_ledger(messages)
