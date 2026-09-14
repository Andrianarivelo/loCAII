"""Context failures expose useful backend causes without replaying or leaking data."""

import copy
import json
from unittest.mock import Mock

import pytest
import requests

from locaii.models import ModelError, ModelManager


@pytest.fixture
def manager(tmp_path):
    """Represent an already-running owned runtime without launching any process."""
    instance = ModelManager(tmp_path / "models.json", tmp_path / "state")
    instance._process = Mock(pid=123)
    instance._process.poll.return_value = None
    instance._model = instance.list_models()[0]
    instance._phase = "ready"
    yield instance
    instance._session.close()


def response(status, value, *, plain=False):
    """Construct a real requests response so HTTPError carries its backend body."""
    result = requests.Response()
    result.status_code = status
    result.url = "http://127.0.0.1:8787/synthetic-endpoint"
    result._content = (value if plain else json.dumps(value)).encode()
    result.encoding = "utf-8"
    return result


def completed_stream():
    """Return one valid synthetic chat stream used only for payload comparison."""
    result = Mock(ok=True)
    result.__enter__ = Mock(return_value=result)
    result.__exit__ = Mock(return_value=False)
    result.iter_lines.return_value = [
        b'data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}',
        b"data: [DONE]",
    ]
    return result


def test_template_json_failure_reports_cause_without_raw_arguments(manager):
    raw_arguments = (
        '{"path":"private-file.txt","content":"' + "SENSITIVE_PAYLOAD" * 1500
    )
    error = {
        "error": {
            "code": 500,
            "type": "server_error",
            "message": "[json.exception.parse_error.101] parse error at line 1, column 11164: syntax error while parsing object - unexpected end of input; last read: "
            + raw_arguments,
        }
    }
    manager._session.post = Mock(return_value=response(500, error))
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_broken",
                    "type": "function",
                    "function": {"name": "write_file", "arguments": raw_arguments},
                }
            ],
        }
    ]
    original = copy.deepcopy(messages)
    with pytest.raises(ModelError) as raised:
        manager.count_tokens(messages)
    message = str(raised.value)
    assert "/apply-template HTTP 500" in message
    assert "json.exception.parse_error.101" in message
    assert "column 11164" in message
    assert "unexpected end of input" in message
    assert "saved history" in message
    assert "no approximate token count" in message
    assert "SENSITIVE_PAYLOAD" not in message
    assert "private-file" not in message
    assert len(message) < 700
    assert manager._session.post.call_count == 1
    assert messages == original


def test_tokenizer_error_identifies_stage_and_does_not_retry(manager):
    manager._session.post = Mock(
        side_effect=[
            response(200, {"prompt": "synthetic prompt"}),
            response(503, {"error": {"message": "tokenizer unavailable"}}),
        ]
    )
    with pytest.raises(ModelError) as raised:
        manager.count_tokens([{"role": "user", "content": "test"}])
    assert "/tokenize HTTP 503" in str(raised.value)
    assert "tokenizer unavailable" in str(raised.value)
    assert "incomplete tool call" not in str(raised.value)
    assert manager._session.post.call_count == 2


def test_plain_backend_error_is_bounded_and_credentials_are_removed(manager):
    detail = "context overflow api_key=SECRET_VALUE " + "details " * 2000
    manager._session.post = Mock(return_value=response(500, detail, plain=True))
    with pytest.raises(ModelError) as raised:
        manager.count_tokens([])
    message = str(raised.value)
    assert "context overflow" in message
    assert "SECRET_VALUE" not in message
    assert len(message) < 550
    assert manager._session.post.call_count == 1


@pytest.mark.parametrize(
    "rendered,tokenized",
    [
        ({"prompt": []}, None),
        ({"prompt": "valid"}, {"tokens": "not-a-token-list"}),
        ({"wrong": "missing prompt"}, None),
    ],
)
def test_malformed_success_body_never_produces_approximate_count(
    manager, rendered, tokenized
):
    replies = [response(200, rendered)]
    if tokenized is not None:
        replies.append(response(200, tokenized))
    manager._session.post = Mock(side_effect=replies)
    with pytest.raises(ModelError, match="no approximate token count"):
        manager.count_tokens([])
    assert manager._session.post.call_count == len(replies)


def test_count_and_chat_submit_identical_prepared_history_and_tools(manager):
    messages = [
        {"role": "user", "content": "Read the synthetic note."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_a",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"note.txt"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "name": "read_file",
            "content": "SYNTHETIC_RECEIPT",
        },
    ]
    original = copy.deepcopy(messages)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        }
    ]
    manager._session.post = Mock(
        side_effect=[
            response(200, {"prompt": "rendered native tool pairs"}),
            response(200, {"tokens": [1, 2, 3]}),
            completed_stream(),
        ]
    )
    assert manager.count_tokens(messages, tools) == 3
    assert manager.chat(messages, tools)["message"]["content"] == "done"
    count_body = manager._session.post.call_args_list[0].kwargs["json"]
    chat_body = manager._session.post.call_args_list[2].kwargs["json"]
    assert count_body["messages"] == chat_body["messages"] == original
    assert count_body["tools"] == chat_body["tools"] == tools
    assert count_body["chat_template_kwargs"] == chat_body["chat_template_kwargs"]
    assert messages == original


def test_chat_backend_error_uses_same_sanitized_parser_detail(manager):
    failure = response(
        500,
        {
            "error": {
                "message": "invalid JSON while parsing tool arguments: "
                + '{"private":"SHOULD_NOT_APPEAR"' * 1000
            }
        },
    )
    manager._session.post = Mock(return_value=failure)
    with pytest.raises(ModelError) as raised:
        manager.chat([])
    assert "Inference HTTP 500" in str(raised.value)
    assert "Malformed JSON" in str(raised.value)
    assert "SHOULD_NOT_APPEAR" not in str(raised.value)
    assert manager.status()["state"] == "ready"
    assert manager._session.post.call_count == 1
