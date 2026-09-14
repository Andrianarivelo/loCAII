"""CSV inspection preserves original bytes and distinguishes evidence from limits."""

import csv
import io
import json
import threading

import pytest

from locaii.storage import Store
from locaii.tools import LocalTools, MAX_READ
import locaii.tools as tool_module


@pytest.fixture
def tools(tmp_path):
    """Use an isolated workspace and disable terminal access for every test."""
    store = Store(tmp_path / "state")
    task = store.create_session("CSV inspection", tmp_path / "workspace")
    return LocalTools(
        store,
        task["id"],
        {"read_all_files": False},
        threading.Event(),
        lambda *args: None,
    )


def write_csv(tools, rows, name="sample.csv"):
    """Create only synthetic fixtures with normal CSV quoting."""
    buffer = io.StringIO(newline="")
    csv.writer(buffer).writerows(rows)
    path = tools.workspace / name
    path.write_text(buffer.getvalue(), encoding="utf-8", newline="")
    return path


def inspect(tools, **args):
    """Exercise public schema validation and dispatch, not just the helper."""
    return tools.execute("inspect_csv", {"path": "sample.csv", **args})


def test_wide_header_returns_actual_selected_data_and_preserves_source(tools):
    path = write_csv(tools, [[f"column_{i}" for i in range(1000)], list(range(1000))])
    before = path.read_bytes(), path.stat().st_mtime_ns
    result = inspect(tools, column_indices=[0, 49, 999])
    assert result["column_count"] == 1000
    assert result["size_bytes"] == len(before[0])
    assert result["rows"][0]["values"] == ["0", "49", "999"]
    assert result["columns_preview_truncated"] is True
    assert result["eof_reached"] is True
    assert result["total_data_rows"] == 1
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_header_only_5000_bytes_has_no_fabricated_data_or_truncation_claim(tools):
    (tools.workspace / "sample.csv").write_text("a" * 4998 + ",b")
    result = inspect(tools)
    assert result["size_bytes"] == 5000
    assert result["header_only"] is True
    assert result["total_data_rows"] == 0
    assert result["rows"] == []
    assert "does not establish" in result["note"]


def test_duplicate_names_and_repeated_header_records_keep_positions(tools):
    write_csv(tools, [["x", "x", "y"], ["1", "2", "3"], ["x", "x", "y"]])
    result = inspect(tools, column_indices=[1, 0])
    assert result["duplicate_header_names"] == [
        {"name": "x", "indices": [0, 1], "occurrences": 2}
    ]
    assert result["rows"][0]["values"] == ["2", "1"]
    assert result["rows"][1]["matches_header"] is True
    assert result["total_data_rows"] == 2
    assert result["repeated_header_rows_observed"] == 1


def test_unicode_bom_and_quoted_newline_are_parsed_as_one_row(tools):
    path = write_csv(
        tools, [["animal", "note"], ["猫", "é,quoted\nline"], ["β", 'a "quote"']]
    )
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())
    result = inspect(tools)
    assert result["selected_columns"][0]["name"] == "animal"
    assert result["rows"][0]["values"] == ["猫", "é,quoted\nline"]
    assert result["rows"][1]["values"] == ["β", 'a "quote"']
    assert result["total_data_rows"] == 2


def test_pagination_reports_observed_lower_bound_until_real_eof(tools):
    write_csv(tools, [["value"], *[[i] for i in range(4)]])
    first = inspect(tools, max_rows=2)
    assert first["rows_examined"] == 3
    assert first["total_data_rows"] is None
    assert first["eof_reached"] is False
    assert first["row_count_is_lower_bound"] is True
    second = inspect(tools, offset_rows=first["next_offset_rows"], max_rows=2)
    assert [r["values"] for r in second["rows"]] == [["2"], ["3"]]
    assert second["total_data_rows"] == 4
    assert second["next_offset_rows"] is None


@pytest.mark.parametrize("payload", [b'a,b\n"unfinished', b"a,b\n\xff,2"])
def test_malformed_or_non_utf8_input_never_claims_eof(tools, payload):
    (tools.workspace / "sample.csv").write_bytes(payload)
    result = inspect(tools)
    assert result["limitations"]
    assert result["eof_reached"] is False
    assert result["header_only"] is None
    assert result["total_data_rows"] is None


@pytest.mark.parametrize("limit", ["CSV_MAX_BYTES", "CSV_MAX_RECORD_BYTES"])
def test_long_quoted_record_hits_byte_bounds_without_false_rows(
    tools, monkeypatch, limit
):
    monkeypatch.setattr(tool_module, limit, 32)
    (tools.workspace / "sample.csv").write_text('a,b\n"' + "long\n" * 100 + '",2\n')
    result = inspect(tools)
    assert result["limitations"]
    assert result["rows"] == []
    assert result["total_data_rows"] is None
    assert result["eof_reached"] is False
    assert result["bytes_read"] <= 37


def test_header_column_bound_is_reported_without_data_claim(tools, monkeypatch):
    monkeypatch.setattr(tool_module, "CSV_MAX_HEADER_COLUMNS", 3)
    write_csv(tools, [["a", "b", "c", "d"], [1, 2, 3, 4]])
    result = inspect(tools)
    assert result["column_count"] == 4
    assert result["limitations"]
    assert result["header_only"] is None
    assert result["eof_reached"] is False


@pytest.mark.parametrize("limit", ["CSV_MAX_SCAN_ROWS", "CSV_MAX_CELLS"])
def test_large_offset_stops_at_scan_budget(tools, monkeypatch, limit):
    monkeypatch.setattr(tool_module, limit, 4)
    write_csv(tools, [["a"], *[[i] for i in range(20)]])
    result = inspect(tools, offset_rows=100000000)
    assert result["rows_examined"] <= 4
    assert result["limitations"]
    assert result["total_data_rows"] is None


def test_output_stays_bounded_even_with_escaped_control_characters(tools):
    rows = [[("\x01" * 1000) + str(i) for i in range(20)]] * 21
    write_csv(tools, rows)
    result = inspect(tools, max_rows=20, max_columns=20)
    assert len(json.dumps(result, ensure_ascii=False)) <= MAX_READ
    assert result["text_truncated"] is True


def test_permissions_schema_and_invalid_column_are_enforced(tools, tmp_path):
    outside = tmp_path / "outside.csv"
    outside.write_text("a\n1\n")
    assert inspect(tools, path=str(outside))["success"] is False
    write_csv(tools, [["a"], [1]])
    assert inspect(tools, column_indices=[1])["success"] is False
    assert inspect(tools, max_rows=21)["success"] is False


def test_empty_and_tsv_files_are_distinguished(tools):
    (tools.workspace / "sample.csv").write_text("")
    empty = inspect(tools)
    assert empty["eof_reached"] and empty["total_data_rows"] == 0
    assert empty["header_only"] is False
    (tools.workspace / "sample.tsv").write_text("a\tb\n1\t2\n")
    result = inspect(tools, path="sample.tsv")
    assert result["rows"][0]["values"] == ["1", "2"]


def test_cancellation_is_checked_during_record_consumption(tools):
    write_csv(tools, [["a"], *[[i] for i in range(100)]])

    class CancelAfterChecks:
        """Trigger during scanning, after public execute's initial check."""

        count = 0

        def is_set(self):
            self.count += 1
            return self.count > 8

    tools.cancel = CancelAfterChecks()
    with pytest.raises(InterruptedError):
        inspect(tools, offset_rows=99)


def test_exact_name_selection_finds_columns_beyond_header_preview(tools):
    header = [f"column_{i}" for i in range(100)] + ["mouse_1_center_x"]
    write_csv(tools, [header, list(range(101))])
    result = inspect(tools, column_names=["mouse_1_center_x"])
    assert result["selected_columns"] == [{"index": 100, "name": "mouse_1_center_x"}]
    assert result["rows"][0]["values"] == ["100"]
    assert result["unmatched_column_names"] == []
    assert result["selection_truncated"] is False


def test_name_selection_preserves_duplicate_positions_in_file_order(tools):
    write_csv(tools, [["x", "y", "x"], [1, 2, 3]])
    result = inspect(tools, column_names=["x", "x"])
    assert result["selected_columns"] == [
        {"index": 0, "name": "x"},
        {"index": 2, "name": "x"},
    ]
    assert result["rows"][0]["values"] == ["1", "3"]


def test_missing_names_are_reported_without_substituting_other_columns(tools):
    write_csv(tools, [["x", "y"], [1, 2]])
    result = inspect(tools, column_names=["mouse_2_center_x"])
    assert result["unmatched_column_names"] == ["mouse_2_center_x"]
    assert result["selected_columns"] == []
    assert result["rows"][0]["values"] == []
    assert result["total_data_rows"] == 1


def test_name_selection_cap_is_explicit_for_many_duplicate_columns(tools):
    write_csv(tools, [["x"] * 25, list(range(25))])
    result = inspect(tools, column_names=["x"])
    assert [column["index"] for column in result["selected_columns"]] == list(range(20))
    assert result["selection_truncated"] is True
    assert result["unmatched_column_names"] == []


def test_index_and_name_selectors_are_mutually_exclusive(tools):
    write_csv(tools, [["x"], [1]])
    result = inspect(tools, column_names=["x"], column_indices=[0])
    assert result["success"] is False
    assert "not both" in result["error"]
