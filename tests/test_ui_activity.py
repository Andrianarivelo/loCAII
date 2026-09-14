"""Readable, literal diagnostics in a narrow inspector without changing receipts."""

import json

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from test_ui import create_window


def rendered_lines(browser):
    """Inspect actual laid-out text lines rather than only the scrollbar policy."""
    block = browser.document().begin()
    while block.isValid():
        layout = block.layout()
        for index in range(layout.lineCount()):
            yield layout.lineAt(index)
        block = block.next()


def test_tool_json_long_token_wraps_without_losing_literal_text(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.activity.setFixedWidth(270)
    token = "synthetic_" + "A" * 3600
    payload = {
        "name": "read_text_file",
        "status": "failed",
        "result": {
            "error": "Unable to parse synthetic request",
            "token": token,
            "literal": "<invalid>&tool",
        },
    }
    original = json.loads(json.dumps(payload))
    window.handle_event("tool", payload)
    app.processEvents()
    plain = window.activity.toPlainText()
    assert token in plain
    assert "<invalid>&tool" in plain
    assert payload == original
    assert window.activity.horizontalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    assert window.activity.horizontalScrollBar().maximum() == 0
    lines = list(rendered_lines(window.activity))
    assert len(lines) > 30
    assert (
        max(line.naturalTextWidth() for line in lines)
        <= window.activity.viewport().width()
    )
    window.close()


def test_long_error_has_short_alert_and_full_wrapped_activity(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.activity.setFixedWidth(270)
    error = (
        "Request failed: synthetic apply-template diagnostic\n"
        + '{"invalid": "'
        + "X" * 4000
        + '"}\nOriginal history retained.'
    )
    window.show_error(error)
    app.processEvents()
    assert len(window.error_label.text()) < 260
    assert "Full details in Activity" in window.error_label.text()
    assert error in window.activity.toPlainText()
    assert window.error_label.textFormat() == Qt.PlainText
    assert window.activity.horizontalScrollBar().maximum() == 0
    assert (
        max(line.naturalTextWidth() for line in rendered_lines(window.activity))
        <= window.activity.viewport().width()
    )
    assert window.error_seen
    window.close()


def test_diagnostic_append_preserves_reader_scroll_and_selection(tmp_path):
    app, window, engine = create_window(tmp_path)
    window.activity.setFixedWidth(270)
    for index in range(40):
        window.append_activity(
            "Synthetic receipt", f"Receipt {index}\nVerified fixture details " * 4
        )
    app.processEvents()
    cursor = window.activity.textCursor()
    cursor.setPosition(0)
    cursor.movePosition(QTextCursor.NextWord, QTextCursor.KeepAnchor)
    window.activity.setTextCursor(cursor)
    window.activity.verticalScrollBar().setValue(0)
    selected = window.activity.textCursor().selectedText()
    window.append_activity(
        "New receipt", "New output while the reader inspects earlier events."
    )
    app.processEvents()
    assert window.activity.verticalScrollBar().value() == 0
    assert window.activity.textCursor().selectedText() == selected
    window.close()
