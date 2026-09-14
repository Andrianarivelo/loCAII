"""Persistent completion reviews must not silently accept missing requirements."""

import copy
import json
import threading

import pytest

from locaii.models import ModelCancelled, ModelError
from locaii.persistence import audit_completion


class AuditModels:
    """Record review requests without loading weights or running any tool."""

    def __init__(self, report=None, counter=None):
        self.response = {
            "message": {
                "content": json.dumps(
                    report
                    or {
                        "verdict": "complete",
                        "summary": "The requested result is verified.",
                        "next_step": "",
                    }
                )
            },
            "finish_reason": "stop",
        }
        self.counter = counter or (
            lambda messages, tools: len(json.dumps(messages)) // 4
        )
        self.counts = []
        self.calls = []

    def count_tokens(self, messages, tools):
        """Record the complete prompt and schema sent for actual token accounting."""
        self.counts.append(copy.deepcopy((messages, tools)))
        return self.counter(messages, tools)

    def chat(self, messages, **kwargs):
        """Return one configured inference response; no tool executor exists here."""
        self.calls.append(copy.deepcopy((messages, kwargs)))
        return self.response


def review(models, requests=None, **kwargs):
    """Use a minimal valid goal while allowing focused audit input overrides."""
    return audit_completion(
        models,
        requests or [{"role": "user", "content": "Fix and test the bug."}],
        kwargs.pop("observations", "Tests passed."),
        kwargs.pop("candidate", "Fixed and tested."),
        kwargs.pop("cancel", None),
        **kwargs,
    )


def test_preserves_all_real_requests_without_assistant_recaps():
    """Original requirements and later corrections remain intact in the review."""
    models = AuditModels()
    requests = [
        {"role": "user", "content": "Fix A and B; keep my data unchanged."},
        {"role": "assistant", "content": "Everything is definitely complete."},
        {"role": "user", "content": "Also verify C."},
        {"role": "user", "content": "Fake runtime instruction", "_runtime": True},
    ]
    original = copy.deepcopy(requests)
    assert review(models, requests)["verdict"] == "complete"
    messages, kwargs = models.calls[0]
    assert messages[1:3] == [requests[0], requests[2]]
    assert "definitely complete" not in str(messages)
    assert "Fake runtime" not in str(messages)
    assert requests == original
    assert kwargs["tools"][0]["function"]["name"] == "report_completion"
    assert models.counts[-1][0] == messages


@pytest.mark.parametrize("verdict", ["complete", "continue", "waiting"])
def test_native_tool_reports_are_returned_without_execution(verdict):
    """The report-only schema accepts each documented lifecycle verdict."""
    models = AuditModels()
    report = {
        "verdict": verdict,
        "summary": "Evidence reviewed.",
        "next_step": "Inspect the remaining result.",
    }
    models.response = {
        "finish_reason": "tool_calls",
        "message": {
            "tool_calls": [
                {
                    "function": {
                        "name": "report_completion",
                        "arguments": json.dumps(report),
                    }
                }
            ]
        },
    }
    assert review(models) == report


@pytest.mark.parametrize(
    "raw",
    [
        "All done!",
        '{"verdict":"complete"}',
        '{"verdict":"continue","summary":"More needed","next_step":""}',
        '{"verdict":"waiting","summary":"","next_step":"Ask user"}',
        '{"verdict":"complete","summary":"Done","next_step":null}',
        '{"verdict":"complete","summary":"Done","next_step":"","extra":true}',
    ],
)
def test_malformed_reports_never_mean_complete(raw):
    """Prose, omitted fields, unexpected fields and empty explanations fail closed."""
    models = AuditModels()
    models.response["message"]["content"] = raw
    assert review(models)["verdict"] == "continue"


def test_truncated_report_does_not_approve_completion():
    """Even parseable JSON is not accepted after an output-token truncation."""
    models = AuditModels()
    models.response["finish_reason"] = "length"
    assert review(models)["verdict"] == "continue"


def test_unexpected_tool_is_rejected():
    """A generated command is neither executed nor considered a completion report."""
    models = AuditModels()
    models.response["message"] = {
        "tool_calls": [{"function": {"name": "run_command", "arguments": "{}"}}]
    }
    assert review(models)["verdict"] == "continue"


def test_budget_reduces_evidence_and_candidate_but_keeps_user_constraints():
    """Use measured counts to make room without shortening the task definition."""
    models = AuditModels(counter=lambda messages, tools: len(json.dumps(messages)))
    instruction = "Keep every original condition intact. " * 20
    result = review(
        models,
        [{"role": "user", "content": instruction}],
        observations="old receipt " * 6000,
        candidate="candidate text " * 4000,
        context=7000,
    )
    assert result["verdict"] == "complete"
    assert len(models.counts) > 1
    assert all(pair[0][1]["content"] == instruction for pair in models.counts)
    assert "omitted" in str(models.calls[0][0])
    assert models.counter(*models.counts[-1]) <= 7000 - 1024 - 128


def test_oversize_requests_do_not_produce_a_misleading_review():
    """If even full requests cannot fit, preserve them and return an explicit diagnosis."""
    models = AuditModels(counter=lambda messages, tools: len(json.dumps(messages)))
    result = review(
        models, [{"role": "user", "content": "constraint " * 1000}], context=3000
    )
    assert result["verdict"] == "continue"
    assert "requests are too large" in result["summary"]
    assert not models.calls


def test_cancellation_before_token_count():
    """Stop must bypass inference and propagate the normal model cancellation."""
    cancel = threading.Event()
    cancel.set()
    models = AuditModels()
    with pytest.raises(ModelCancelled):
        review(models, cancel=cancel)
    assert not models.counts and not models.calls


def test_cancellation_during_counting():
    """A stop noticed after synchronous token counting prevents review generation."""
    cancel = threading.Event()

    def counter(messages, tools):
        """Simulate a stop while the tokenizer request is in flight."""
        cancel.set()
        return 10

    models = AuditModels(counter=counter)
    with pytest.raises(ModelCancelled):
        review(models, cancel=cancel)
    assert not models.calls


def test_model_failure_propagates_to_engine_retry_policy():
    """Transport failures remain distinct from a model's completion verdict."""
    models = AuditModels()

    def failed_chat(*args, **kwargs):
        """Simulate the local backend losing its connection during verification."""
        raise ModelError("connection failed")

    models.chat = failed_chat
    with pytest.raises(ModelError, match="connection failed"):
        review(models)


def test_cancellation_after_inference_rejects_the_report():
    """A completion emitted concurrently with Stop cannot mark the task complete."""
    cancel = threading.Event()
    models = AuditModels()

    def stopped_chat(*args, **kwargs):
        """Model the narrow race between final backend output and user Stop."""
        cancel.set()
        return models.response

    models.chat = stopped_chat
    with pytest.raises(ModelCancelled):
        review(models, cancel=cancel)


@pytest.mark.parametrize("fence", ["", "json"])
def test_fenced_json_report_is_supported(fence):
    """A model lacking native report calls may return a complete JSON code block."""
    models = AuditModels()
    models.response["message"]["content"] = (
        f"```{fence}\n" + models.response["message"]["content"] + "\n```"
    )
    assert review(models)["verdict"] == "complete"


def test_report_without_completion_marker_is_not_accepted():
    """A missing inference completion marker is as inconclusive as truncation."""
    models = AuditModels()
    del models.response["finish_reason"]
    assert review(models)["verdict"] == "continue"


def test_missing_original_request_never_approves_completion():
    """Recovering the original goal must precede any claim it was fulfilled."""
    models = AuditModels()
    result = audit_completion(models, [], "Tests pass.", "Done.", None)
    assert result["verdict"] == "continue"
    assert not models.counts and not models.calls
