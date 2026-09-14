"""Current observations must defeat stale claims and repeated text prefixes."""

import json

from locaii.engine import Engine, collapse_receipts, observation_ledger
from test_engine import FakeModels, answer, call, run, task


def history_pair(identifier, body):
    """Create a real native tool-call pair for the same inspected path."""
    return [
        answer("", [call("read_text_file", {"path": "metadata.csv"}, identifier)])[
            "message"
        ],
        {
            "role": "tool",
            "name": "read_text_file",
            "tool_call_id": identifier,
            "content": json.dumps(body),
        },
    ]


def test_shorter_new_read_supersedes_old_large_receipt():
    """A shrunken input cannot be hidden by preferring the longest prior result."""
    old = {
        "path": "metadata.csv",
        "size_bytes": 51816693,
        "content": "old data " * 2000,
        "next_offset_chars": 6000,
    }
    new = {
        "path": "metadata.csv",
        "size_bytes": 5000,
        "content": "header only",
        "next_offset_chars": None,
    }
    history = history_pair("before", old) + history_pair("after", new)
    collapsed = collapse_receipts(history)
    assert json.loads(collapsed[1]["content"])["superseded"]
    assert json.loads(collapsed[3]["content"]) == new
    assert old["content"] not in str(collapsed)


def test_ledger_retains_current_file_size_not_old_claim():
    old = {
        "path": "metadata.csv",
        "size_bytes": 51816693,
        "content": "x" * 6000,
        "next_offset_chars": 6000,
    }
    new = {
        "path": "metadata.csv",
        "size_bytes": 5000,
        "content": "header",
        "next_offset_chars": None,
    }
    ledger = observation_ledger(history_pair("old", old) + history_pair("new", new))
    assert "5000 bytes" in ledger
    assert "51816693" not in ledger
    assert "more from 6000" not in ledger


def test_stalled_csv_is_actually_inspected_without_writing(task):
    """Recovery changes read strategy and records zero observed data rows."""
    store, identifier, workspace = task
    file = workspace / "metadata.csv"
    file.write_text("mouse_1_x,mouse_2_x," + "h" * 4980)
    before = file.read_bytes()
    stamp = file.stat().st_mtime_ns
    responses = [
        answer(
            "The file is large; I will inspect data.",
            [call("read_text_file", {"path": "metadata.csv"}, f"r{i}")],
        )
        for i in range(1)
    ]

    def finish(messages, _options):
        receipts = [
            json.loads(m["content"])
            for m in messages
            if m.get("role") == "tool" and m.get("name") == "inspect_csv"
        ]
        assert receipts[-1]["header_only"] is True
        assert receipts[-1]["total_data_rows"] == 0
        assert receipts[-1]["size_bytes"] == len(before)
        return answer(
            "The current file contains a header but no data rows. Coordinate comparisons cannot be verified from it."
        )

    models = FakeModels(responses + [finish])
    result = run(Engine(store, models), identifier, "Inspect the coordinate data.")
    assert result["status"] == "completed"
    assert "no data rows" in result["text"]
    assert file.read_bytes() == before and file.stat().st_mtime_ns == stamp
    records = store.messages(identifier)
    assert len([m for m in records if m.get("name") == "inspect_csv"]) >= 1
    assert any(
        m.get("content") == "The file is large; I will inspect data." for m in records
    )


def test_engine_shell_source_writes_follow_task_permission(task):
    """Exercise enabled source editing and disabled writes through the app payload."""
    import threading
    from locaii.tools import LocalTools

    store, identifier, workspace = task
    source = workspace / "metadata.csv"
    source.write_text("x,y\n1,2\n")
    local = LocalTools(
        store,
        identifier,
        {"workspace_writes": True, "allow_terminal": True},
        threading.Event(),
        lambda *args: None,
    )
    edited = local.execute("run_command", {"command": "printf updated > metadata.csv"})
    assert edited["status"] == "completed", edited
    assert source.read_text() == "updated"
    local.options["workspace_writes"] = False
    refused = local.execute("run_command", {"command": "printf denied > metadata.csv"})
    assert refused["status"] == "failed", refused
    assert source.read_text() == "updated"
    local.options["workspace_writes"] = True
    allowed = local.execute(
        "run_command", {"command": 'cp metadata.csv "$LOCAII_OUTPUT_DIR/derived.csv"'}
    )
    assert allowed["status"] == "completed", allowed
    assert (
        local.command_output_root() / "outputs" / "derived.csv"
    ).read_text() == source.read_text()


def test_changing_columns_on_header_only_file_is_not_progress(task):
    """Different selectors cannot indefinitely hide the same zero-row evidence."""
    store, identifier, workspace = task
    (workspace / "metadata.csv").write_text("a,b,c,d,e,f,g,h,i\n")
    requests = [
        answer(
            "",
            [
                call(
                    "inspect_csv",
                    {"path": "metadata.csv", "column_indices": [i]},
                    f"c{i}",
                )
            ],
        )
        for i in range(9)
    ]
    result = run(
        Engine(store, FakeModels(requests)), identifier, "Inspect actual data rows"
    )
    assert result["status"] == "needs_attention"
    assert "zero data rows" in result["text"]
    assert store.get_session(identifier)["status"] == "needs_attention"
    assert any(e["type"] == "input_diagnosis" for e in store.events(identifier))
    assert (
        len(
            [
                e
                for e in store.events(identifier)
                if e.get("reason") == "repeated_observations"
            ]
        )
        == 0
    )


def test_recovery_keeps_instructions_and_latest_evidence_without_false_recaps(task):
    """A retry may discard model prose, never the user's constraints or durable history."""
    store, identifier, _ = task
    store.append(
        identifier, "user", "Compare mouse_1_x and mouse_2_x. Never change originals."
    )
    store.append(
        identifier, "assistant", "The file is 51 MB and all rows were normalized."
    )
    for message in history_pair(
        "current",
        {"path": "metadata.csv", "size_bytes": 5000, "next_offset_chars": None},
    ):
        store.append(identifier, message)
    before = store.messages(identifier)
    engine = Engine(store, FakeModels())
    retry = engine._ground_recovery_history(
        identifier,
        engine._context(identifier),
        {"role": "user", "_runtime": True, "content": "Use observed results."},
    )
    assert "Never change originals." in str(retry)
    assert "5000" in str(retry)
    assert "51 MB" not in str(retry)
    assert "all rows were normalized" not in str(retry)
    assert store.messages(identifier) == before


def test_first_text_read_provides_requested_columns_beyond_preview(task):
    """Actual named measurements, not a prefix of generic headers, reach the model."""
    import csv

    store, identifier, workspace = task
    source = workspace / "metadata.csv"
    with source.open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow([f"field_{i}" for i in range(800)] + ["mouse_1_x", "mouse_2_x"])
        writer.writerow(["0"] * 800 + ["10", "20"])

    def finish(messages, _options):
        receipts = [
            json.loads(m["content"]) for m in messages if m.get("name") == "inspect_csv"
        ]
        assert receipts[-1]["rows"][0]["values"] == ["10", "20"]
        return answer("The observed coordinates differ: 10 and 20.")

    models = FakeModels(
        [answer("", [call("read_text_file", {"path": "metadata.csv"}, "read")]), finish]
    )
    result = run(
        Engine(store, models),
        identifier,
        "Compare mouse_1_x and mouse_2_x in metadata.csv. Never modify it.",
    )
    assert result["status"] == "completed"
    assert (
        len(
            [
                p
                for p in models.prompts
                if not p[0]["content"].startswith("Write concise Markdown task memory")
            ]
        )
        == 2
    )


def test_distinct_csv_pages_and_columns_survive_prompt_recovery(task):
    """A comparison needs both pages or selected columns, not just the last receipt."""
    store, identifier, _ = task
    store.append(
        identifier, "user", "Compare the observed rows without changing inputs."
    )
    history = []
    for index, args in enumerate(
        [
            {"path": "table.csv", "offset_rows": 0, "column_names": ["left"]},
            {"path": "table.csv", "offset_rows": 3, "column_names": ["left"]},
            {"path": "table.csv", "offset_rows": 3, "column_names": ["right"]},
            {"path": "table.csv", "offset_rows": 3, "column_names": ["right"]},
        ]
    ):
        pair = [
            answer("", [call("inspect_csv", args, str(index))])["message"],
            {
                "role": "tool",
                "name": "inspect_csv",
                "tool_call_id": str(index),
                "content": json.dumps({"path": "table.csv", "rows": [index]}),
            },
        ]
        history.extend(pair)
        for message in pair:
            store.append(identifier, message)
    collapsed = collapse_receipts(history.copy())
    actual = [
        json.loads(m["content"])["rows"]
        for m in collapsed
        if m.get("role") == "tool" and not json.loads(m["content"]).get("superseded")
    ]
    assert actual == [[0], [1], [3]]
    engine = Engine(store, FakeModels())
    grounded = engine._ground_recovery_history(
        identifier,
        engine._context(identifier),
        {"role": "user", "content": "Use the receipts."},
    )
    actual = [
        json.loads(m["content"])["rows"] for m in grounded if m.get("role") == "tool"
    ]
    assert actual == [[0], [1], [3]]


def test_empty_template_diagnosis_does_not_claim_corruption(task):
    """No rows is not evidence that a valid empty schema needs restoring."""
    store, identifier, workspace = task
    (workspace / "template.csv").write_text("a,b,c\n")
    requests = [
        answer("", [call("read_text_file", {"path": "template.csv"}, f"r{i}")])
        for i in range(3)
    ]
    result = run(
        Engine(store, FakeModels(requests)),
        identifier,
        "Describe the column names in this empty template.",
    )
    assert result["status"] == "needs_attention"
    assert "Column names can be inspected" in result["text"]
    assert "does not prove" in result["text"]
    assert (
        "Requested measurements or comparisons cannot be verified" not in result["text"]
    )


def test_repaired_table_is_rechecked_before_empty_input_diagnosis(task, monkeypatch):
    """An external repair must invalidate the earlier zero-row observation."""
    from locaii.tools import LocalTools

    store, identifier, workspace = task
    file = workspace / "metadata.csv"
    file.write_text("x,y\n")
    original = LocalTools.execute
    inspected = 0

    def execute(local, name, args):
        nonlocal inspected
        if name == "inspect_csv":
            inspected += 1
            if inspected == 2:
                file.write_text("x,y\n10,20\n")
        return original(local, name, args)

    monkeypatch.setattr(LocalTools, "execute", execute)
    requests = [
        answer("", [call("read_text_file", {"path": "metadata.csv"}, f"r{i}")])
        for i in range(3)
    ]

    def finish(messages, _options):
        reports = [
            json.loads(m["content"]) for m in messages if m.get("name") == "inspect_csv"
        ]
        assert reports[-1]["rows"][0]["values"] == ["10", "20"]
        return answer("The updated input now has a data row: 10,20.")

    result = run(
        Engine(store, FakeModels(requests + [finish])), identifier, "Inspect data rows."
    )
    assert result["status"] == "completed"
    assert not any(e["type"] == "input_diagnosis" for e in store.events(identifier))
