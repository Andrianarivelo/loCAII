"""Qt desktop workbench. Backend work runs in cancellable worker threads.

Widgets depend on service contracts rather than inference implementations. A new
model or workflow therefore appears without introducing provider logic into Qt.
"""

from __future__ import annotations

import copy
import html
import json
import os
import secrets
import threading
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt, QThread, Signal, QTimer, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFileSystemModel,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

ACCENT = "#426c5b"
STYLE = """
QWidget { font-family: 'Inter', 'Noto Sans', sans-serif; font-size: 13px; color: #27332e; }
QMainWindow, QWidget#canvas { background: #f8f9f6; }
QWidget#sidebar { background: #eef1ea; border-right: 1px solid #dfe5dc; }
QLabel#brand { font-size: 29px; font-weight: 750; letter-spacing: -1px; color: #294b3b; }
QLabel#eyebrow { color: #7a877b; font-size: 10px; font-weight: 700; letter-spacing: 1.5px; }
QLabel#title { font-size: 26px; font-weight: 700; letter-spacing: -0.6px; }
QLabel#subtitle, QLabel#muted { color: #7b847e; }
QLabel#status { color: #426c5b; background: #e8efe7; border-radius: 10px; padding: 7px 12px; }
QLabel#warning { color: #a34530; background: #fbede7; border-radius: 8px; padding: 8px; }
QPushButton { background: #fff; border: 1px solid #dbe2d9; border-radius: 8px; padding: 9px 14px; font-weight: 600; }
QPushButton:hover { background: #e7eee5; border-color: #bbcbbd; }
QPushButton:disabled { color: #9da79e; background: #f1f3ee; }
QPushButton#primary { background: #426c5b; border-color: #426c5b; color: white; }
QPushButton#primary:hover { background: #335847; }
QPushButton#nav { text-align: left; background: transparent; border: 0; padding: 12px 15px; color: #627064; }
QPushButton#nav:checked { background: #dce7dc; color: #284e3b; }
QPushButton#danger { color: #a34530; }
QLineEdit, QTextEdit, QTextBrowser, QComboBox, QSpinBox, QDoubleSpinBox { background: white; border: 1px solid #dfe5dc; border-radius: 7px; padding: 7px; selection-background-color: #cfdfce; }
QTextBrowser#message { border: 0; background: transparent; padding: 0; }
QTextBrowser#details { background: #f7f9f4; font-family: 'JetBrains Mono', 'DejaVu Sans Mono'; font-size: 11px; }
QTextEdit#composer { border: 0; background: transparent; padding: 3px; font-size: 14px; }
QFrame#composerFrame { background: white; border: 1px solid #dbe2d7; border-radius: 14px; }
QFrame#card { background: white; border: 1px solid #e1e7dd; border-radius: 12px; }
QFrame#userBubble { background: #eaf0e6; border: 1px solid #dfe8d8; border-radius: 12px; }
QFrame#assistantBubble { background: transparent; border: none; }
QListWidget { background: transparent; border: 0; outline: 0; }
QListWidget::item { padding: 11px 10px; margin: 2px 0; border-radius: 7px; }
QListWidget::item:selected { color: #294d3b; background: #dce7dc; }
QListWidget::item:hover { background: #e4ebe0; }
QTabWidget::pane { border: 1px solid #e0e6dc; border-radius: 8px; background: #fff; }
QTabBar::tab { padding: 11px 12px; color: #79867a; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #355c46; border-bottom: 2px solid #426c5b; }
QTreeView { border: 0; background: #fff; alternate-background-color: #f8faf6; }
QHeaderView::section { background: #f4f7ef; border: 0; padding: 7px; color: #778276; }
QScrollArea { background: transparent; border: 0; }
QScrollBar:vertical { width: 8px; background: transparent; }
QScrollBar::handle:vertical { background: #d3ddd0; border-radius: 4px; min-height: 35px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QCheckBox { spacing: 8px; }
QSplitter::handle { background: #e5eae1; width: 1px; }
QToolTip { background: #294b3b; color: #fff; border: none; padding: 6px; }
"""


def label(text: str, name: str = "", wrap: bool = False) -> QLabel:
    """Create a consistently styled, optionally wrapping label."""
    widget = QLabel(text)
    widget.setObjectName(name)
    widget.setWordWrap(wrap)
    return widget


def button(
    text: str, callback: Callable | None = None, primary: bool = False
) -> QPushButton:
    """Create a button and connect its explicit action."""
    widget = QPushButton(text)
    if primary:
        widget.setObjectName("primary")
    if callback:
        widget.clicked.connect(callback)
    return widget


def card() -> tuple[QFrame, QVBoxLayout]:
    """Return a card and its content layout."""
    frame = QFrame()
    frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(22, 20, 22, 20)
    layout.setSpacing(14)
    return frame, layout


class Worker(QThread):
    """Deliver service events through queued Qt signals; cancellation is cooperative."""

    event = Signal(str, object)
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, action: Callable, parent: QWidget):
        super().__init__(parent)
        self.action = action
        self.cancel = threading.Event()

    def run(self) -> None:
        try:
            self.result.emit(
                self.action(
                    self.cancel, lambda name, payload: self.event.emit(name, payload)
                )
            )
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)


class Composer(QTextEdit):
    """Enter sends, Shift+Enter adds a newline; dropped local files are attachments."""

    submitted = Signal()
    files_dropped = Signal(list)

    def keyPressEvent(self, event) -> None:
        if (
            event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and not event.modifiers() & Qt.ShiftModifier
        ):
            self.submitted.emit()
            return
        super().keyPressEvent(event)

    def canInsertFromMimeData(self, source) -> bool:
        return source.hasUrls() or super().canInsertFromMimeData(source)

    def insertFromMimeData(self, source) -> None:
        if source.hasUrls():
            paths = [url.toLocalFile() for url in source.urls() if url.isLocalFile()]
            if paths:
                self.files_dropped.emit(paths)
                return
        super().insertFromMimeData(source)


class Message(QFrame):
    """A selectable Markdown message that grows with its rendered document."""

    def __init__(self, role: str, content: str):
        super().__init__()
        self.setObjectName("userBubble" if role == "user" else "assistantBubble")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)
        layout.addWidget(
            label("YOU" if role == "user" else "loCAII  /  LOCAL ASSISTANT", "eyebrow")
        )
        self.browser = QTextBrowser()
        self.browser.setObjectName("message")
        self.browser.setOpenExternalLinks(True)
        self.browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.browser.document().documentLayout().documentSizeChanged.connect(self.fit)
        layout.addWidget(self.browser)
        self.content = ""
        self.set_content(content)

    def set_content(self, content: str) -> None:
        self.content = content
        self.browser.setMarkdown(content)
        QTimer.singleShot(0, self.fit)

    def fit(self, *_args) -> None:
        width = max(150, self.browser.viewport().width())
        self.browser.document().setTextWidth(width)
        self.browser.setFixedHeight(
            max(32, min(900, int(self.browser.document().size().height()) + 8))
        )
        self.browser.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded
            if self.browser.document().size().height() > 890
            else Qt.ScrollBarAlwaysOff
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        QTimer.singleShot(0, self.fit)


class MainWindow(QMainWindow):
    """The application shell. Only worker threads call potentially slow backends."""

    def __init__(self, store, engine, models, media):
        super().__init__()
        self.store, self.engine, self.models, self.media = store, engine, models, media
        self.session_id: str | None = None
        self.workspace = str(Path.home())
        self.attachments: list[str] = []
        self.worker: Worker | None = None
        self._active_chat_session_id: str | None = None
        self._active_chat_options: dict | None = None
        self._active_chat_succeeded = False
        self._steering_restart_generation = 0
        self._visible_steering_ids: set[str] = set()
        self._steering_bubbles: dict[str, Message] = {}
        self._stream_prefix_before_update = ""
        self.stream_message: Message | None = None
        self.error_seen = False
        self._pending_close = False
        self._delegation_hub = None
        self._delegation_job_id: str | None = None
        self._delegation_session_id: str | None = None
        self._supervisor_instructions = ""
        self._supervisor_scope = None
        self._model_rows: list[dict] = []
        self._media_files: list[str] = []
        self.setWindowTitle("loCAII · Local intelligence, your workspace")
        self.resize(1480, 980)
        self.setMinimumSize(1080, 720)
        self.setStyleSheet(STYLE)
        icon = Path(__file__).resolve().parents[2] / "assets" / "locaii.svg"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))
        root = QWidget()
        root.setObjectName("canvas")
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._build_sidebar(outer)
        self.pages = QStackedWidget()
        outer.addWidget(self.pages, 1)
        self._build_chat()
        self._build_create()
        self._build_models()
        self._build_supervision()
        self.reload_models()
        self.reload_sessions()
        self.refresh_workflows()
        self.reload_media_jobs(recover_pending=True)
        self._load_preferences()
        if self.task_list.count():
            self.task_list.setCurrentRow(0)
        else:
            self.show_welcome()
        self.statusBar().showMessage(
            "Ready · Files stay local unless you enable a remote supervisor"
        )

    def _build_sidebar(self, outer: QHBoxLayout) -> None:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(228)
        outer.addWidget(side)
        layout = QVBoxLayout(side)
        layout.setContentsMargins(20, 25, 18, 20)
        layout.setSpacing(8)
        layout.addWidget(label("◈  loCAII", "brand"))
        layout.addWidget(label("YOUR LOCAL AI WORKSPACE", "eyebrow"))
        layout.addSpacing(27)
        self.nav_buttons = []
        for index, text in enumerate(
            ["◌   Workspace", "✧   Create", "▦   Models", "◎   Supervision"]
        ):
            nav = button(text, lambda checked=False, i=index: self.navigate(i))
            nav.setObjectName("nav")
            nav.setCheckable(True)
            nav.setChecked(index == 0)
            layout.addWidget(nav)
            self.nav_buttons.append(nav)
        layout.addSpacing(22)
        row = QHBoxLayout()
        row.addWidget(label("TASKS", "eyebrow"))
        row.addStretch()
        new = button("+", self.new_task)
        new.setToolTip("Create a new task")
        new.setFixedSize(32, 30)
        new.setStyleSheet("padding: 0; font-size: 20px;")
        row.addWidget(new)
        layout.addLayout(row)
        self.task_list = QListWidget()
        self.task_list.currentItemChanged.connect(self.select_task)
        layout.addWidget(self.task_list, 1)
        privacy, privacy_layout = card()
        privacy_layout.setContentsMargins(12, 13, 12, 13)
        privacy_layout.addWidget(label("●  LOCAL FIRST", "eyebrow"))
        privacy_layout.addWidget(
            label(
                "Your models. Your files.\nAn inspectable trail of work.", "muted", True
            )
        )
        layout.addWidget(privacy)
        layout.addWidget(label("loCAII  0.1  /  Desktop", "muted"))

    def _page(
        self, title: str, subtitle: str, scrollable: bool = False
    ) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        page.setObjectName("canvas")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 26, 30, 20)
        layout.setSpacing(16)
        layout.addWidget(label(title, "title"))
        layout.addWidget(label(subtitle, "subtitle", True))
        if scrollable:
            scroller = QScrollArea()
            scroller.setWidgetResizable(True)
            scroller.setWidget(page)
            self.pages.addWidget(scroller)
        else:
            self.pages.addWidget(page)
        return page, layout

    def _build_chat(self) -> None:
        page, layout = self._page(
            "A little intelligence. A lot of possibility.",
            "Think, make, and finish work with models running on your computer.",
        )
        self.chat_heading = layout.itemAt(0).widget()
        top = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(270)
        self.model_combo.setToolTip(
            "The local model used for this task. It loads on the first message."
        )
        top.addWidget(self.model_combo)
        self.workspace_button = button("⌁  Workspace", self.choose_workspace)
        top.addWidget(self.workspace_button)
        top.addStretch()
        self.run_status = label("● Ready", "status")
        top.addWidget(self.run_status)
        layout.addLayout(top)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, 1)
        chat = QWidget()
        body = QVBoxLayout(chat)
        body.setContentsMargins(0, 0, 18, 0)
        body.setSpacing(12)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.messages_widget = QWidget()
        self.messages_widget.setObjectName("canvas")
        self.message_layout = QVBoxLayout(self.messages_widget)
        self.message_layout.setContentsMargins(0, 12, 0, 12)
        self.message_layout.setSpacing(14)
        self.message_layout.addStretch()
        self.scroll.setWidget(self.messages_widget)
        body.addWidget(self.scroll, 1)
        self.error_label = label("", "warning", True)
        self.error_label.hide()
        body.addWidget(self.error_label)
        self.steering_status = label("", "muted", True)
        self.steering_status.hide()
        body.addWidget(self.steering_status)
        composer = QFrame()
        composer.setObjectName("composerFrame")
        compose_layout = QVBoxLayout(composer)
        compose_layout.setContentsMargins(15, 12, 15, 12)
        self.attachment_label = label("", "muted", True)
        self.attachment_label.hide()
        compose_layout.addWidget(self.attachment_label)
        self.composer = Composer()
        self.composer.setObjectName("composer")
        self.composer.setPlaceholderText(
            "Ask anything, or give your local assistant a task…"
        )
        self.composer.setFixedHeight(76)
        self.composer.submitted.connect(self.send)
        self.composer.files_dropped.connect(self.add_attachments)
        compose_layout.addWidget(self.composer)
        controls = QHBoxLayout()
        controls.addWidget(button("＋ Files", self.pick_attachments))
        self.clear_files_button = button("Clear files", self.clear_attachments)
        self.clear_files_button.hide()
        controls.addWidget(self.clear_files_button)
        self.continue_button = button("Continue", self.resume)
        controls.addWidget(self.continue_button)
        controls.addStretch()
        self.stop_button = button("Stop", self.stop)
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        controls.addWidget(self.stop_button)
        self.send_button = button("Send  ↑", self.send, True)
        controls.addWidget(self.send_button)
        compose_layout.addLayout(controls)
        body.addWidget(composer)
        body.addWidget(
            label(
                "Enter to send · Shift + Enter for a new line · Drop files to attach",
                "muted",
            )
        )
        splitter.addWidget(chat)
        self.inspect_tabs = QTabWidget()
        self.inspect_tabs.setMinimumWidth(270)
        self.activity = QTextBrowser()
        self.activity.setObjectName("details")
        self.activity.document().setMaximumBlockCount(3000)
        self.activity.setPlaceholderText(
            "Tool calls, progress, and verification appear here."
        )
        self.inspect_tabs.addTab(self.activity, "Activity")
        self.memory_view = QTextBrowser()
        self.memory_view.setOpenExternalLinks(True)
        self.inspect_tabs.addTab(self.memory_view, "Memory")
        file_page = QWidget()
        file_layout = QVBoxLayout(file_page)
        file_layout.setContentsMargins(6, 8, 6, 8)
        file_layout.addWidget(button("Open workspace folder", self.open_workspace))
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(self.workspace)
        self.file_tree = QTreeView()
        self.file_tree.setModel(self.fs_model)
        self.file_tree.setRootIndex(self.fs_model.index(self.workspace))
        self.file_tree.setColumnWidth(0, 190)
        for column in (1, 2, 3):
            self.file_tree.hideColumn(column)
        self.file_tree.doubleClicked.connect(self.preview_file)
        file_layout.addWidget(self.file_tree)
        file_layout.addWidget(
            label("Double-click a file for a read-only preview.", "muted", True)
        )
        self.inspect_tabs.addTab(file_page, "Files")
        splitter.addWidget(self.inspect_tabs)
        splitter.setSizes([850, 300])

    def _build_create(self) -> None:
        _, layout = self._page(
            "From an idea to an image.",
            "Generate images and video using your local ComfyUI workflows.",
            scrollable=True,
        )
        row = QHBoxLayout()
        settings, form_layout = card()
        settings.setMaximumWidth(410)
        form = QFormLayout()
        self.media_kind = QComboBox()
        self.media_kind.addItems(["image", "video"])
        self.media_kind.currentTextChanged.connect(self.refresh_workflows)
        self.workflow_combo = QComboBox()
        self._workflow_rows = []
        self.workflow_combo.currentIndexChanged.connect(self.workflow_selected)
        self.media_prompt = QTextEdit()
        self.media_prompt.setPlaceholderText(
            "Describe the scene, light, movement, and feeling…"
        )
        self.media_prompt.setFixedHeight(120)
        self.media_negative = QLineEdit()
        self.media_negative.setPlaceholderText("Optional details to avoid")
        self.media_fps = QSpinBox()
        self.media_fps.setRange(1, 60)
        self.media_fps.setValue(16)
        self.media_input = QLineEdit()
        self.media_input.setPlaceholderText("Optional source image path")
        image_row = QWidget()
        self.media_image_row = image_row
        image_controls = QHBoxLayout(image_row)
        image_controls.setContentsMargins(0, 0, 0, 0)
        image_controls.addWidget(self.media_input)
        self.media_input_button = button("…", self.pick_media_input)
        image_controls.addWidget(self.media_input_button)
        self.media_width = QSpinBox()
        self.media_width.setRange(128, 2048)
        self.media_width.setSingleStep(64)
        self.media_width.setValue(768)
        self.media_height = QSpinBox()
        self.media_height.setRange(128, 2048)
        self.media_height.setSingleStep(64)
        self.media_height.setValue(768)
        self.media_steps = QSpinBox()
        self.media_steps.setRange(1, 100)
        self.media_steps.setValue(20)
        self.media_frames = QSpinBox()
        self.media_frames.setRange(9, 129)
        self.media_frames.setSingleStep(8)
        self.media_frames.setValue(49)
        self.media_seed = QSpinBox()
        self.media_seed.setRange(-1, 2147483647)
        self.media_seed.setValue(-1)
        self.media_seed.setToolTip("-1 chooses a random seed")
        for name, widget in [
            ("Type", self.media_kind),
            ("Workflow", self.workflow_combo),
            ("Prompt", self.media_prompt),
            ("Avoid", self.media_negative),
            ("Source image", image_row),
            ("Width", self.media_width),
            ("Height", self.media_height),
            ("Steps", self.media_steps),
            ("Video frames", self.media_frames),
            ("Video FPS", self.media_fps),
            ("Seed", self.media_seed),
        ]:
            form.addRow(name, widget)
        self.media_form = form
        form_layout.addLayout(form)
        self.media_preset_note = label("", "muted", True)
        form_layout.addWidget(self.media_preset_note)
        form_layout.addWidget(
            label(
                "Rendering releases the chat model to make room on the GPU. Workflow support determines which controls apply.",
                "muted",
                True,
            )
        )
        self.generate_button = button("Generate locally", self.generate, True)
        form_layout.addWidget(self.generate_button)
        self.media_stop = button("Stop generation", self.stop)
        self.media_stop.setEnabled(False)
        form_layout.addWidget(self.media_stop)
        form_layout.addWidget(button("Import workflow JSON", self.import_workflow))
        form_layout.addWidget(button("Render engine settings", self.media_settings))
        form_layout.addWidget(
            button(
                "Open ComfyUI",
                lambda: QDesktopServices.openUrl(
                    QUrl(
                        getattr(self.media, "config", {}).get(
                            "url", "http://127.0.0.1:8188"
                        )
                    )
                ),
            )
        )
        form_layout.addStretch()
        row.addWidget(settings)
        preview, preview_layout = card()
        self.media_preview = QLabel(
            "✧\n\nYour next creation starts here.\nWrite a prompt and choose a workflow."
        )
        self.media_preview.setAlignment(Qt.AlignCenter)
        self.media_preview.setStyleSheet(
            "font-size: 18px; color: #81917f; background: #f3f6ef; border-radius: 10px;"
        )
        self.media_preview.setMinimumSize(320, 260)
        preview_layout.addWidget(self.media_preview, 1)
        self.media_status = label("Local render engine · ComfyUI", "muted", True)
        preview_layout.addWidget(self.media_status)
        self.media_results = QListWidget()
        self.media_results.setMaximumHeight(120)
        self.media_results.itemDoubleClicked.connect(
            lambda item: self.open_media_file(item.data(Qt.UserRole))
        )
        preview_layout.addWidget(self.media_results)
        output_controls = QHBoxLayout()
        output_controls.addWidget(
            button("Open selected output", self.open_selected_media)
        )
        output_controls.addWidget(button("Refresh outputs", self.recover_media_jobs))
        preview_layout.addLayout(output_controls)
        row.addWidget(preview, 1)
        layout.addLayout(row, 1)

    def _build_models(self) -> None:
        _, layout = self._page(
            "Your model library.",
            "Add GGUF weights, select a memory profile, and run independently of LM Studio.",
            scrollable=True,
        )
        row = QHBoxLayout()
        row.addWidget(button("＋ Import GGUF", self.import_model, True))
        row.addWidget(button("Refresh library", self.reload_models))
        self.load_model_button = button("Load selected model", self.load_model)
        row.addWidget(self.load_model_button)
        self.unload_model_button = button("Unload model", self.unload_model)
        row.addWidget(self.unload_model_button)
        self.cpu_preset_button = button("CPU-only preset", self.cpu_preset)
        row.addWidget(self.cpu_preset_button)
        row.addStretch()
        layout.addLayout(row)
        self.models_list = QListWidget()
        self.models_list.currentRowChanged.connect(self.model_selected)
        layout.addWidget(self.models_list, 1)
        profile, profile_layout = card()
        profile_layout.addWidget(label("MODEL PROFILE", "eyebrow"))
        form = QFormLayout()
        self.model_name = QLineEdit()
        self.model_context = QSpinBox()
        self.model_context.setRange(2048, 262144)
        self.model_context.setSingleStep(2048)
        self.model_context.setValue(32768)
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(-1, 999)
        self.gpu_layers.setValue(-1)
        self.gpu_layers.setToolTip("-1 offloads all available layers; 0 uses CPU only.")
        self.model_device = QLineEdit()
        self.model_device.setPlaceholderText(
            "Automatic GPU selection; enter none for CPU only"
        )
        self.mmproj = QLineEdit()
        self.mmproj.setPlaceholderText("Optional vision projector GGUF path")
        self.model_vision = QCheckBox(
            "Enable image understanding (requires matching projector)"
        )
        self.model_thinking = QCheckBox(
            "Enable reasoning mode (uses additional generation tokens)"
        )
        for name, widget in [
            ("Display name", self.model_name),
            ("Context tokens", self.model_context),
            ("GPU layers", self.gpu_layers),
            ("Device", self.model_device),
            ("Vision projector", self.mmproj),
            ("Vision", self.model_vision),
            ("Reasoning", self.model_thinking),
        ]:
            form.addRow(name, widget)
        profile_layout.addLayout(form)
        profile_layout.addWidget(
            label(
                "32K is the conservative starting point for this machine. Compaction supports long tasks without allocating an ever-larger GPU cache. Changes apply on the next model load.",
                "muted",
                True,
            )
        )
        profile_layout.addWidget(button("Save selected profile", self.save_model))
        self.model_status = label("No model loaded by this app.", "muted", True)
        profile_layout.addWidget(self.model_status)
        layout.addWidget(profile)

    def _build_supervision(self) -> None:
        _, layout = self._page(
            "Big-picture thinking. Local execution.",
            "A supervisor plans and reviews milestones while your local model does the detailed work.",
            scrollable=True,
        )
        connection, connection_layout = card()
        connection_layout.addWidget(label("YOUR SUPERVISOR", "eyebrow"))
        connection_layout.addWidget(
            label(
                "Open an assistant and give it your local model as a teammate.",
                "",
                True,
            )
        )
        connection_layout.addWidget(
            label(
                "The supervisor plans, delegates simple to moderate tasks to the selected local "
                "model, and checks its results. You can watch each delegated task in Workspace.",
                "muted",
                True,
            )
        )
        connection_row = QHBoxLayout()
        self.supervisor_cli = QComboBox()
        self.supervisor_cli.addItem("Codex CLI", "codex-cli")
        self.supervisor_cli.addItem("Claude CLI", "claude-cli")
        connection_row.addWidget(self.supervisor_cli)
        self.open_supervisor_button = button(
            "Open && connect supervisor", self.open_supervisor, True
        )
        connection_row.addWidget(self.open_supervisor_button)
        self.copy_supervisor_button = button(
            "Copy launch command", self.copy_supervisor_connection
        )
        connection_row.addWidget(self.copy_supervisor_button)
        self.disconnect_supervisor_button = button(
            "Disconnect", self.disconnect_supervisor
        )
        self.disconnect_supervisor_button.setEnabled(False)
        connection_row.addWidget(self.disconnect_supervisor_button)
        connection_layout.addLayout(connection_row)
        connection_layout.addWidget(
            label(
                "Opening or copying a connection allows the external assistant to receive "
                "tasks and results from the selected workspace. Local work uses the file and "
                "terminal permissions below. Your CLI account handles its own usage charges.",
                "muted",
                True,
            )
        )
        self.supervisor_connection_status = label("Disconnected", "status", True)
        connection_layout.addWidget(self.supervisor_connection_status)
        self.supervisor_scope_label = label(
            "Uses the selected model and workspace. Opens your assistant in a terminal.",
            "muted",
            True,
        )
        connection_layout.addWidget(self.supervisor_scope_label)
        self.supervisor_launch_error = label("", "warning", True)
        self.supervisor_launch_error.hide()
        connection_layout.addWidget(self.supervisor_launch_error)
        layout.addWidget(connection)

        self.advanced_supervision_toggle = button("Advanced review settings")
        self.advanced_supervision_toggle.setCheckable(True)
        layout.addWidget(self.advanced_supervision_toggle)
        self.advanced_supervision = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_supervision)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        self.advanced_supervision_toggle.toggled.connect(
            self.advanced_supervision.setVisible
        )
        self.advanced_supervision.setVisible(False)
        layout.addWidget(self.advanced_supervision)
        box, content = card()
        form = QFormLayout()
        form.setVerticalSpacing(15)
        self.supervisor_provider = QComboBox()
        for caption, value in [
            ("Off · local task execution", "off"),
            ("Local model · self-review", "local"),
            ("Manual review packets", "manual"),
            ("OpenAI API", "openai"),
            ("Anthropic API", "anthropic"),
            ("Codex CLI", "codex-cli"),
            ("Claude CLI", "claude-cli"),
        ]:
            self.supervisor_provider.addItem(caption, value)
        self.supervisor_provider.currentIndexChanged.connect(self.supervisor_selected)
        self.supervisor_model = QLineEdit()
        self.supervisor_model.setPlaceholderText(
            "Provider model ID (optional for local/manual)"
        )
        self.supervisor_scope = QComboBox()
        self.supervisor_scope.addItem("Tool work only (recommended)", "tasks")
        self.supervisor_scope.addItem("Every message, including conversation", "all")
        self.supervisor_scope.setToolTip(
            "Tool work only saves supervisor usage for tasks that use local tools. "
            "Every message also requests planning and review for ordinary conversation."
        )
        self.supervisor_every = QSpinBox()
        self.supervisor_every.setRange(1, 100)
        self.supervisor_every.setValue(6)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(1, 500)
        self.max_iterations.setValue(60)
        self.token_budget = QSpinBox()
        self.token_budget.setRange(256, 1000000)
        self.token_budget.setSingleStep(1000)
        self.token_budget.setValue(60000)
        self.remote_enabled = QCheckBox(
            "Allow objective, file excerpts, and evidence to reach the selected external supervisor"
        )
        self.remote_enabled.setToolTip(
            "Remote API and CLI modes need this explicit authorization. Local/manual modes do not send data automatically."
        )
        self.allow_terminal = QCheckBox(
            "Allow full-user shell access (can change files outside the workspace)"
        )
        self.allow_terminal.setToolTip(
            "Terminal commands run with your account permissions. File-tool backups and raw-data protections cannot constrain arbitrary shell commands. Enable only for tasks you trust."
        )
        self.workspace_writes = QCheckBox(
            "Allow writes inside the selected workspace, with backups"
        )
        self.workspace_writes.setChecked(True)
        self.read_all_files = QCheckBox(
            "Allow reading local files outside the workspace"
        )
        self.read_all_files.setChecked(True)
        self.raw_protection = QCheckBox(
            "Protect raw research data and metadata from automatic edits"
        )
        self.raw_protection.setChecked(True)
        self.raw_protection.setEnabled(False)
        for name, widget in [
            ("Supervisor", self.supervisor_provider),
            ("Supervisor model", self.supervisor_model),
            ("When to review", self.supervisor_scope),
            ("Review every N steps", self.supervisor_every),
            ("Maximum local iterations", self.max_iterations),
            ("Supervisor token allowance", self.token_budget),
            ("External access", self.remote_enabled),
            ("Terminal", self.allow_terminal),
            ("Workspace", self.workspace_writes),
            ("Read access", self.read_all_files),
            ("Data integrity", self.raw_protection),
        ]:
            form.addRow(name, widget)
        content.addLayout(form)
        self.supervisor_allowance_hint = label(
            "CLI reviews include substantial agent context. Increase the allowance to at least "
            "24,000 tokens before running automatic CLI reviews, or choose manual review.",
            "warning",
            True,
        )
        self.supervisor_allowance_hint.hide()
        content.addWidget(self.supervisor_allowance_hint)
        self.token_budget.valueChanged.connect(self.supervisor_selected)
        content.addWidget(
            label(
                "API modes use OPENAI_API_KEY or ANTHROPIC_API_KEY from your environment. CLI modes use your authenticated CLI. API reviews use compact packets. CLI reviews include extra agent context; the token allowance is an estimate, not a billing cap. No credentials are extracted from another app.",
                "muted",
                True,
            )
        )
        content.addWidget(button("Save preferences", self.save_preferences, True))
        advanced_layout.addWidget(box)
        packets, packets_layout = card()
        packets_layout.addWidget(label("BRING YOUR EXISTING ASSISTANT", "eyebrow"))
        packets_layout.addWidget(
            label(
                "Export the active task's compact review packet for Codex or Claude. Import its structured review to guide the next local iteration.",
                "muted",
                True,
            )
        )
        controls = QHBoxLayout()
        controls.addWidget(button("Export review packet", self.export_packet))
        controls.addWidget(button("Import supervisor review", self.import_packet))
        controls.addStretch()
        packets_layout.addLayout(controls)
        self.review_view = QTextBrowser()
        self.review_view.setMaximumHeight(120)
        self.review_view.setPlaceholderText("Reviews and supervisor usage appear here.")
        packets_layout.addWidget(self.review_view)
        self.supervisor_usage_label = label(
            "No supervisor usage recorded for this task.", "muted", True
        )
        packets_layout.addWidget(self.supervisor_usage_label)
        self.token_budget.valueChanged.connect(self.update_supervisor_totals)
        advanced_layout.addWidget(packets)
        layout.addStretch()
        self.delegation_timer = QTimer(self)
        self.delegation_timer.setInterval(400)
        self.delegation_timer.timeout.connect(self.poll_delegation)

    def _supervisor_connection(self):
        """Freeze the selected workspace, model, and local permissions per connection."""
        from .delegation import DelegationHub
        from .supervisor_launch import SupervisorLauncher

        model_id = self.model_combo.currentData()
        model = next(
            (row for row in self.models.list_models() if row.get("id") == model_id),
            None,
        )
        if not model:
            raise ValueError(
                "Select a local model in Models before connecting a supervisor."
            )
        if self._delegation_hub is None:
            self._delegation_hub = DelegationHub(self._preferences_path().parent)
        if self._supervisor_scope is None:
            options = {
                **self.options(),
                "remote_enabled": False,
                "supervisor_provider": "off",
            }
            connection_path = self._delegation_hub.connect(
                self.workspace, options, model
            )
            self._supervisor_scope = (connection_path, self.workspace, model)
        else:
            connection_path, _, model = self._supervisor_scope
        self.supervisor_scope_label.setText(
            f"Local model: {model.get('name', model['id'])} · Workspace: {self._supervisor_scope[1]}\n"
            "Disconnect before changing the connection's workspace, model, or permissions."
        )
        launcher = SupervisorLauncher(self._preferences_path().parent)
        self.disconnect_supervisor_button.setEnabled(True)
        self.delegation_timer.start()
        return launcher, connection_path, model

    def open_supervisor(self) -> None:
        """Launch a dedicated CLI with a connection scoped to this app and workspace."""
        existing_connection = self._supervisor_scope is not None
        self.supervisor_launch_error.hide()
        try:
            launcher, connection_path, model = self._supervisor_connection()
            result = launcher.open(
                self.supervisor_cli.currentData(),
                connection_path,
                self._supervisor_scope[1],
                model,
            )
            self._supervisor_instructions = str(result.get("instructions", ""))
            self._save_supervisor_choice()
            self.supervisor_connection_status.setText(
                "Ready. Waiting for the supervisor to connect…"
            )
        except Exception as exc:
            if not existing_connection:
                self.disconnect_supervisor()
            self.supervisor_launch_error.setText(
                "Could not open supervisor: " + str(exc)
            )
            self.supervisor_launch_error.show()

    def copy_supervisor_connection(self) -> None:
        """Prepare scoped instructions for an existing assistant without launching it."""
        existing_connection = self._supervisor_scope is not None
        self.supervisor_launch_error.hide()
        try:
            launcher, connection_path, model = self._supervisor_connection()
            result = launcher.prepare(
                self.supervisor_cli.currentData(),
                connection_path,
                self._supervisor_scope[1],
                model,
            )
            self._supervisor_instructions = str(result["instructions"])
            QApplication.clipboard().setText(self._supervisor_instructions)
            self._save_supervisor_choice()
            self.supervisor_connection_status.setText(
                "Connection instructions copied. Waiting for the supervisor…"
            )
        except Exception as exc:
            if not existing_connection:
                self.disconnect_supervisor()
            self.supervisor_launch_error.setText(
                "Could not prepare connection: " + str(exc)
            )
            self.supervisor_launch_error.show()

    def disconnect_supervisor(self) -> None:
        """Revoke new delegations while allowing an already running task to finish."""
        if self._delegation_hub is not None:
            self._delegation_hub.disconnect()
        self.delegation_timer.stop()
        self._supervisor_scope = None
        self.disconnect_supervisor_button.setEnabled(False)
        self.supervisor_connection_status.setText(
            "Disconnected. The running local task will finish; use Stop to interrupt it."
            if self._delegation_job_id
            else "Disconnected"
        )

    def poll_delegation(self) -> None:
        """Drain one local job only when chat, inference, and rendering are idle."""
        hub = self._delegation_hub
        if hub is None or self._pending_close:
            return
        identity = None
        try:
            state = hub.status()
            contacted = bool(state.get("connected") and state.get("last_client_seen"))
            self.supervisor_connection_status.setText(
                (
                    "Supervisor contacted · connection available"
                    if contacted
                    else "Ready. Waiting for supervisor"
                )
                + (f" · Local task running" if self._delegation_job_id else "")
                + f" · {state.get('queued', 0)} queued"
            )
            if self._delegation_job_id:
                if self.worker and hub.cancel_requested(self._delegation_job_id):
                    self.worker.cancel.set()
                return
            if self.worker:
                return
            job = hub.take_next()
            if not job:
                return
            identity = str(job["id"])
            session_id = job.get("session_id")
            if not session_id:
                session = self.store.create_session(
                    "Delegated · " + str(job.get("title") or job["task"])[:65],
                    job["workspace"],
                )
                session_id = str(session["id"])
            hub.bind_session(identity, session_id)
            self._delegation_job_id = identity
            self._delegation_session_id = session_id
            self.reload_sessions()
            if (
                not self.session_id
                and not self.composer.toPlainText().strip()
                and not self.attachments
            ):
                for row in range(self.task_list.count()):
                    item = self.task_list.item(row)
                    if item.data(Qt.UserRole) == session_id:
                        self.task_list.setCurrentItem(item)
                        break
            self.statusBar().showMessage(
                "Supervisor delegated a local task. It appears in your task list."
            )
            options = {
                **job["options"],
                "supervisor_provider": "off",
                "remote_enabled": False,
            }

            def run_delegated(cancel, emit):
                """Persist bridge receipts independently of the currently displayed task."""

                def relay(event, payload):
                    hub.record_event(identity, event, payload)
                    emit(
                        "delegated_event",
                        {"session_id": session_id, "event": event, "payload": payload},
                    )

                try:
                    result = self.engine.run(
                        session_id, job["task"], [], options, cancel, relay
                    )
                    hub.complete(identity, result)
                    return result
                except Exception as exc:
                    hub.fail(identity, str(exc))
                    raise

            self.start_work(
                run_delegated,
                self.delegation_finished,
                session_id=session_id,
                options=options,
            )
        except Exception as exc:
            if identity is not None and not self.worker:
                hub.fail(identity, str(exc))
                self._delegation_job_id = None
                self._delegation_session_id = None
            self.delegation_timer.stop()
            self.supervisor_connection_status.setText(
                "Connection needs attention: " + str(exc)
            )

    def delegation_finished(self, result: Any) -> None:
        """Keep the user's selected task and unsent draft untouched by background work."""
        if self.session_id == self._delegation_session_id:
            self.chat_finished(result)

    def update_supervisor_totals(self, *_args) -> None:
        """Account for the full task history, not merely the visible event tail."""
        if (
            not hasattr(self, "supervisor_usage_label")
            or not self.session_id
            or not hasattr(self.store, "supervisor_tokens")
        ):
            return
        total = self.store.supervisor_tokens(self.session_id)
        remote = self.store.supervisor_tokens(self.session_id, remote_only=True)
        local = max(0, total - remote)
        allowance = self.token_budget.value()
        self.supervisor_usage_label.setText(
            f"Task allowance accounting: {remote:,} external tokens / {allowance:,} allowance; {local:,} local review tokens. Counts may include estimates. Local reviews do not consume the external allowance."
        )

    def supervisor_selected(self, *_args) -> None:
        """Explain insufficient CLI headroom without overriding the user's allowance."""
        if hasattr(self, "supervisor_allowance_hint"):
            self.supervisor_allowance_hint.setVisible(
                self.supervisor_provider.currentData() in {"codex-cli", "claude-cli"}
                and self.token_budget.value() < 24000
            )

    def navigate(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        for i, nav in enumerate(self.nav_buttons):
            nav.setChecked(i == index)

    def reload_sessions(self) -> None:
        selected = self.session_id
        self.task_list.blockSignals(True)
        self.task_list.clear()
        try:
            for session in self.store.sessions():
                item = QListWidgetItem(session.get("title") or "Untitled task")
                item.setData(Qt.UserRole, str(session["id"]))
                item.setToolTip(
                    f"{session.get('workspace', '')}\n{session.get('status', 'ready')}"
                )
                self.task_list.addItem(item)
                if str(session["id"]) == selected:
                    self.task_list.setCurrentItem(item)
        finally:
            self.task_list.blockSignals(False)

    def new_task(self) -> None:
        if self.worker:
            self.statusBar().showMessage(
                "Stop the current task before creating another."
            )
            return
        self.session_id = None
        self.task_list.setCurrentRow(-1)
        self.navigate(0)
        self.clear_messages()
        self.show_welcome()
        self.activity.clear()
        self.memory_view.clear()
        self.error_label.hide()
        self.review_view.clear()
        self.supervisor_usage_label.setText(
            "No supervisor usage recorded for this task."
        )
        self.composer.clear()
        self.clear_attachments()
        self.composer.setFocus()
        self.chat_heading.setText("A little intelligence. A lot of possibility.")

    def select_task(self, item: QListWidgetItem | None, _previous=None) -> None:
        if not item:
            return
        selected = item.data(Qt.UserRole)
        if self.worker and not self._delegation_job_id and selected != self.session_id:
            self.reload_sessions()
            return
        self.session_id = selected
        try:
            session = self.store.get_session(selected)
            self.set_workspace(session.get("workspace") or str(Path.home()))
            self.chat_heading.setText((session.get("title") or "Workspace")[:65])
            self.clear_messages()
            for message in self.store.messages(selected):
                if (
                    message.get("role") in ("user", "assistant")
                    and message.get("content")
                    and not message.get("_runtime")
                ):
                    self.add_message(
                        message["role"], self.message_text(message["content"])
                    )
            self.memory_view.setMarkdown(
                self.store.memory(selected)
                or "No checkpoint yet. Memory is saved as the task progresses."
            )
            self.activity.clear()
            self.review_view.clear()
            self.error_label.hide()
            if hasattr(self.store, "events"):
                for saved in self.store.events(selected, limit=80):
                    kind = saved.get("type", "event")
                    if kind in {
                        "tool",
                        "review",
                        "plan",
                        "error",
                        "compaction",
                        "usage",
                        "supervisor_usage",
                        "review_skipped",
                        "steering_applied",
                    }:
                        self.handle_event(kind, saved)
            self.update_supervisor_totals()
            self.refresh_pending_steering(show_messages=True)
            self.update_send_mode()
            self.navigate(0)
        except Exception as exc:
            self.show_error(str(exc))

    @staticmethod
    def message_text(content: Any) -> str:
        """Display multimodal transcripts without exposing encoded image payloads."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    parts.append(str(part.get("text", "")))
                elif isinstance(part, dict) and part.get("type") == "image_url":
                    parts.append("*[Attached image]*")
            return "\n\n".join(parts)
        return str(content)

    def clear_messages(self) -> None:
        self.stream_message = None
        self._stream_prefix_before_update = ""
        self._visible_steering_ids.clear()
        self._steering_bubbles.clear()
        self.steering_status.hide()
        while self.message_layout.count() > 1:
            item = self.message_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def show_welcome(self) -> None:
        welcome, layout = card()
        layout.setContentsMargins(27, 26, 27, 26)
        layout.addWidget(label("A WORKSPACE THAT WORKS WITH YOU", "eyebrow"))
        layout.addWidget(
            label(
                "Start with a question.\nBuild toward something useful.", "title", True
            )
        )
        layout.addWidget(
            label(
                "Choose a model, set a workspace, and describe the result you want. Your assistant can use tools, save progress, and pick up where it left off.",
                "muted",
                True,
            )
        )
        for text in [
            "Explore a folder and explain what is inside",
            "Build a small Python tool and test it",
            "Plan a long task with checkable milestones",
        ]:
            suggestion = button(
                "↗  " + text,
                lambda checked=False, t=text: self.composer.setPlainText(t),
            )
            layout.addWidget(suggestion)
        self.message_layout.insertWidget(0, welcome)

    def add_message(self, role: str, text: str) -> Message:
        message = Message(role, text)
        self.message_layout.insertWidget(self.message_layout.count() - 1, message)
        QTimer.singleShot(30, self.scroll_bottom)
        return message

    def scroll_bottom(self) -> None:
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        )

    def choose_workspace(self) -> None:
        if self.worker:
            return
        path = QFileDialog.getExistingDirectory(
            self, "Choose a task workspace", self.workspace
        )
        if path:
            if self.session_id:
                self.new_task()
            self.set_workspace(path)

    def set_workspace(self, path: str) -> None:
        self.workspace = path
        self.workspace_button.setText("⌁  " + (Path(path).name or path))
        self.workspace_button.setToolTip(path)
        self.fs_model.setRootPath(path)
        self.file_tree.setRootIndex(self.fs_model.index(path))

    def open_workspace(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.workspace))

    def preview_file(self, index) -> None:
        path = Path(self.fs_model.filePath(index))
        if not path.is_file():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Read-only preview · {path.name}")
        dialog.resize(850, 650)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label(str(path), "muted", True))
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            view = QLabel()
            view.setAlignment(Qt.AlignCenter)
            view.setPixmap(
                QPixmap(str(path)).scaled(
                    800, 530, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        else:
            view = QTextBrowser()
            try:
                with path.open("rb") as handle:
                    data = handle.read(200_001)
                text = data[:200_000].decode("utf-8")
                if "\x00" in text:
                    raise ValueError("Binary format")
                view.setPlainText(
                    text
                    + ("\n\n[Preview limited to 200 KB]" if len(data) > 200_000 else "")
                )
            except (OSError, UnicodeError, ValueError):
                view.setPlainText(
                    "This format cannot be previewed as text. Open it with its associated application, or attach it to a task."
                )
        layout.addWidget(view, 1)
        controls = QHBoxLayout()
        controls.addWidget(
            button(
                "Attach to task",
                lambda: (self.add_attachments([str(path)]), dialog.accept()),
            )
        )
        controls.addWidget(
            button(
                "Open externally",
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))),
            )
        )
        controls.addWidget(button("Close", dialog.accept))
        layout.addLayout(controls)
        dialog.exec()

    def pick_attachments(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Attach files", self.workspace)
        self.add_attachments(paths)

    def add_attachments(self, paths: list[str]) -> None:
        for path in paths:
            if Path(path).is_file() and path not in self.attachments:
                self.attachments.append(path)
        self.attachment_label.setText(
            "Attached: " + " · ".join(Path(path).name for path in self.attachments)
        )
        self.attachment_label.setVisible(bool(self.attachments))
        self.clear_files_button.setVisible(bool(self.attachments))

    def clear_attachments(self) -> None:
        self.attachments.clear()
        self.attachment_label.hide()
        self.clear_files_button.hide()

    def options(self) -> dict:
        return {
            "model_id": self.model_combo.currentData(),
            "workspace_writes": self.workspace_writes.isChecked(),
            "allow_terminal": self.allow_terminal.isChecked(),
            "read_all_files": self.read_all_files.isChecked(),
            "supervisor_provider": self.supervisor_provider.currentData(),
            "supervisor_model": self.supervisor_model.text().strip(),
            "supervisor_every": self.supervisor_every.value(),
            "supervisor_scope": self.supervisor_scope.currentData(),
            "max_iterations": self.max_iterations.value(),
            "supervisor_token_budget": self.token_budget.value(),
            "remote_enabled": self.remote_enabled.isChecked(),
        }

    def send(self) -> None:
        if self.worker:
            self.queue_live_update()
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        if not self.model_combo.currentData():
            self.show_error("Import or select a GGUF model in Models first.")
            self.navigate(2)
            return
        options = self.options()
        if (
            options["supervisor_provider"]
            in {"openai", "anthropic", "codex-cli", "claude-cli"}
            and not options["remote_enabled"]
        ):
            self.show_error(
                "Enable external supervisor access in Supervision, or choose a local/off supervisor."
            )
            self.navigate(3)
            return
        if not self.session_id:
            try:
                session = self.store.create_session(text[:65], self.workspace)
                self.session_id = str(session["id"])
                self.clear_messages()
                self.chat_heading.setText(text[:65])
                self.reload_sessions()
            except Exception as exc:
                self.show_error(str(exc))
                return
        attachments = list(self.attachments)
        session_id = self.session_id
        self.add_message(
            "user",
            text
            + (
                "\n\n*Attached: " + ", ".join(Path(p).name for p in attachments) + "*"
                if attachments
                else ""
            ),
        )
        self.stream_message = None
        self.composer.clear()
        self.clear_attachments()
        self.error_label.hide()
        self.start_work(
            lambda cancel, emit: self.engine.run(
                session_id, text, attachments, options, cancel, emit
            ),
            self.chat_finished,
            session_id=session_id,
            options=options,
        )

    def update_send_mode(self) -> None:
        """Allow steering only for the displayed, actively executing chat task."""
        can_update = bool(
            self.worker
            and self._active_chat_session_id
            and self._active_chat_session_id == self.session_id
        )
        self.send_button.setText("Send update" if can_update else "Send  ↑")
        self.send_button.setEnabled(not self.worker or can_update)

    def refresh_pending_steering(self, show_messages: bool = False) -> list:
        """Keep durable, unapplied updates visible after interruption or selection."""
        if not self.session_id or not hasattr(self.store, "pending_steering"):
            return []
        pending = self.store.pending_steering(self.session_id)
        if show_messages:
            for item in pending:
                identity = str(item["id"])
                if identity not in self._visible_steering_ids:
                    self.add_message("user", str(item["text"]) + "\n\n*Queued update*")
                    self._visible_steering_ids.add(identity)
        if pending:
            tail = (
                "Waiting for a safe boundary."
                if self.worker
                else "Use Continue to apply them."
            )
            self.steering_status.setText(f"{len(pending)} update(s) queued. {tail}")
            self.steering_status.show()
        return pending

    def queue_live_update(self) -> None:
        """Save steering before acknowledging it, without disturbing the running worker."""
        if (
            not self._active_chat_session_id
            or self.session_id != self._active_chat_session_id
        ):
            self.statusBar().showMessage(
                "Select the running chat task to send an update. Other operations cannot receive chat updates."
            )
            return
        if self.worker.cancel.is_set():
            self.statusBar().showMessage(
                "This task is stopping. Your draft is preserved; use Continue after it stops."
            )
            return
        if self.attachments:
            self.steering_status.setText(
                "Updates during a run accept text only. Clear attachments to send this update, or wait for the task to finish. Your draft and files are preserved."
            )
            self.steering_status.show()
            return
        text = self.composer.toPlainText().strip()
        if not text:
            return
        try:
            queued = self.store.queue_steering(self._active_chat_session_id, text)
        except Exception as exc:
            self.steering_status.setText(
                "Could not save this update. Your draft is preserved: " + str(exc)
            )
            self.steering_status.show()
            return
        if self.stream_message:
            self._stream_prefix_before_update += self.stream_message.content
        self.stream_message = None
        self._steering_bubbles[str(queued["id"])] = self.add_message(
            "user", text + "\n\n*Queued update*"
        )
        self._visible_steering_ids.add(str(queued["id"]))
        self.composer.clear()
        self.refresh_pending_steering()
        self.statusBar().showMessage(
            "Update saved. It will enter the model's context at the next safe boundary."
        )

    def resume(self) -> None:
        """Continue the selected task without overwriting an unsent composer draft."""
        if self.session_id and not self.worker:
            session_id, options = self.session_id, self.options()
            if not options.get("model_id"):
                self.show_error("Import or select a GGUF model in Models first.")
                self.navigate(2)
                return
            if (
                options["supervisor_provider"]
                in {"openai", "anthropic", "codex-cli", "claude-cli"}
                and not options["remote_enabled"]
            ):
                self.show_error(
                    "Enable external supervisor access in Supervision, or choose a local/off supervisor."
                )
                self.navigate(3)
                return
            self.add_message("user", "Continue")
            self.stream_message = None
            self._stream_prefix_before_update = ""
            self.start_work(
                lambda cancel, emit: self.engine.run(
                    session_id, "Continue", [], options, cancel, emit
                ),
                self.chat_finished,
                session_id=session_id,
                options=options,
            )

    def _resume_late_steering(
        self, session_id: str, options: dict, generation: int
    ) -> None:
        """Close the completion race only for successful normal chat, after QThread exit."""
        if (
            self.worker
            or self._pending_close
            or self.session_id != session_id
            or generation != self._steering_restart_generation
            or not self.store.pending_steering(session_id)
        ):
            return
        self.stream_message = None
        self._stream_prefix_before_update = ""
        self.start_work(
            lambda cancel, emit: self.engine.run(
                session_id, "", [], options, cancel, emit
            ),
            self.chat_finished,
            session_id=session_id,
            options=options,
        )

    def start_work(
        self,
        action: Callable,
        finished: Callable | None = None,
        *,
        session_id: str | None = None,
        options: dict | None = None,
    ) -> None:
        """Serialize jobs and explicitly distinguish steerable chats from other work."""
        if self.worker:
            self.show_error(
                "Another operation is still running. Stop it before starting this one."
            )
            return
        self._steering_restart_generation += 1
        self._active_chat_session_id = session_id
        self._active_chat_options = copy.deepcopy(options) if session_id else None
        self._active_chat_succeeded = False
        self._stream_prefix_before_update = ""
        self.error_seen = False
        self.error_label.hide()
        self.worker = Worker(action, self)
        self.worker.event.connect(self.handle_event)
        if finished:
            self.worker.result.connect(finished)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.work_finished)
        self.update_send_mode()
        self.continue_button.setEnabled(False)
        self.generate_button.setEnabled(False)
        self.workspace_button.setEnabled(False)
        self.load_model_button.setEnabled(False)
        self.unload_model_button.setEnabled(False)
        self.cpu_preset_button.setEnabled(False)
        self.task_list.setEnabled(bool(self._delegation_job_id))
        self.stop_button.setEnabled(True)
        self.media_stop.setEnabled(True)
        self.run_status.setText("◌ Working")
        self.worker.start()

    def stop(self) -> None:
        self._steering_restart_generation += 1
        if self.worker:
            self.worker.cancel.set()
            self.run_status.setText("◌ Stopping…")
            self.statusBar().showMessage(
                "Stop requested. Waiting for the current operation to checkpoint and release resources."
            )

    def work_finished(self) -> None:
        old = self.worker
        stopped = bool(old and old.cancel.is_set())
        session_id = self._active_chat_session_id
        options = self._active_chat_options
        successful = self._active_chat_succeeded and not self.error_seen and not stopped
        delegated = bool(self._delegation_job_id)
        generation = self._steering_restart_generation
        self.worker = None
        self._active_chat_session_id = None
        self._active_chat_options = None
        self._active_chat_succeeded = False
        self._delegation_job_id = None
        self._delegation_session_id = None
        if old:
            old.deleteLater()
        self.update_send_mode()
        self.continue_button.setEnabled(True)
        self.generate_button.setEnabled(True)
        self.workspace_button.setEnabled(True)
        self.load_model_button.setEnabled(True)
        self.unload_model_button.setEnabled(True)
        self.cpu_preset_button.setEnabled(True)
        self.task_list.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.media_stop.setEnabled(False)
        self.run_status.setText(
            "! Needs attention"
            if self.error_seen
            else "◌ Stopped" if stopped else "● Ready"
        )
        self.reload_sessions()
        pending = self.refresh_pending_steering()
        if self._pending_close:
            self.close()
        elif successful and session_id and not delegated and pending:
            QTimer.singleShot(
                0,
                lambda: self._resume_late_steering(
                    session_id, options or {}, generation
                ),
            )

    def chat_finished(self, result: Any) -> None:
        if isinstance(result, dict):
            self._active_chat_succeeded = result.get("status") == "completed"
            text = result.get("content") or result.get("text") or result.get("response")
            if (
                isinstance(text, str)
                and self._stream_prefix_before_update
                and text.startswith(self._stream_prefix_before_update)
            ):
                text = text[len(self._stream_prefix_before_update) :]
            if text and (
                not self.stream_message or self.stream_message.content != text
            ):
                if self.stream_message:
                    self.stream_message.set_content(str(text))
                else:
                    self.stream_message = self.add_message("assistant", str(text))
            if result.get("status") in {"error", "failed"}:
                self.show_error(
                    str(
                        result.get("error")
                        or result.get("message")
                        or "Task needs attention. Inspect the activity log."
                    )
                )
        if self.session_id:
            try:
                self.memory_view.setMarkdown(self.store.memory(self.session_id))
            except Exception as exc:
                self.activity.append(
                    "Memory display could not refresh: " + html.escape(str(exc))
                )

    def handle_event(self, event: str, payload: Any) -> None:
        """Render truthful progress; errors never become blank success messages."""
        payload = payload if isinstance(payload, dict) else {"text": str(payload)}
        if event == "delegated_event":
            if payload.get("session_id") == self.session_id:
                self.handle_event(payload["event"], payload["payload"])
            return
        if event == "delta":
            text = str(payload.get("text", ""))
            if not self.stream_message:
                self.stream_message = self.add_message("assistant", text)
            else:
                self.stream_message.set_content(self.stream_message.content + text)
            self.scroll_bottom()
        elif event in {"status", "progress", "model_status"}:
            text = str(
                payload.get("text")
                or payload.get("message")
                or (
                    str(payload.get("state", "")) + " " + str(payload.get("model", ""))
                ).strip()
                or "Working…"
            )
            self.run_status.setText("◌ " + text[:70])
            self.media_status.setText(text)
            self.statusBar().showMessage(text)
            if event != "progress":
                self.activity.append(
                    "<p style='color:#6b826c'>" + html.escape(text) + "</p>"
                )
            if event == "model_status":
                self.model_status.setText(text)
        elif event == "error":
            self.show_error(str(payload.get("text") or payload.get("error") or payload))
        elif event == "memory":
            self.memory_view.setMarkdown(str(payload.get("text", "")))
            self.memory_view.setToolTip(str(payload.get("path", "")))
        elif event == "tool":
            self._stream_prefix_before_update = ""
            name = str(payload.get("name", "tool"))
            state = str(payload.get("status", ""))
            detail = payload.get("result", payload.get("arguments", {}))
            rendered = (
                json.dumps(detail, ensure_ascii=False, indent=2, default=str)
                if not isinstance(detail, str)
                else detail
            )
            if len(rendered) > 9000:
                rendered = (
                    rendered[:9000]
                    + "\n[Display excerpt. Full receipt remains in task history.]"
                )
            self.activity.append(
                f"<p><b>{html.escape(name)}</b> · {html.escape(state)}</p><pre>{html.escape(rendered)}</pre>"
            )
            self.stream_message = None
        elif event == "steering_applied":
            for identity in payload.get("ids", []):
                bubble = self._steering_bubbles.get(str(identity))
                if bubble:
                    bubble.set_content(
                        bubble.content.removesuffix("\n\n*Queued update*")
                        + "\n\n*Added to model context*"
                    )
            self._stream_prefix_before_update = ""
            self.stream_message = None
            text = "Update added to the model's next prompt. Continuing with the updated request."
            self.steering_status.setText(text)
            self.steering_status.show()
            self.activity.append("<p>" + html.escape(text) + "</p>")
            self.statusBar().showMessage(text)
        elif event == "review_skipped":
            # A skipped external review is neither local failure nor verified success.
            # Keep the answer and its streaming widget intact as local work continues.
            text = str(
                payload.get("text")
                or "Automatic review skipped; local work can continue."
            )
            self.activity.append("<p>Review skipped: " + html.escape(text) + "</p>")
            self.statusBar().showMessage(text)
        elif event in {"review", "usage", "plan", "supervisor_usage"}:
            detail = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
            self.activity.append(
                f"<b>{event.title()}</b><pre>{html.escape(detail)}</pre>"
            )
            if event in {"review", "supervisor_usage"}:
                self.review_view.append(
                    f"<b>{event.replace('_', ' ').title()}</b><pre>{html.escape(detail)}</pre>"
                )
            if event == "review" and self.session_id and hasattr(self.store, "events"):
                usages = [
                    entry
                    for entry in self.store.events(self.session_id, limit=30)
                    if entry.get("type") == "supervisor_usage"
                ]
                if usages:
                    usage = usages[-1]
                    scope = (
                        "Local model"
                        if usage.get("provider") == "local"
                        else "External supervisor"
                    )
                    measure = "Estimated" if usage.get("estimated") else "Reported"
                    caption = f"{scope} · {usage.get('provider')} · {measure} tokens: {usage.get('charged_tokens', 'unavailable')}"
                    self.review_view.append(
                        "<b>"
                        + html.escape(caption)
                        + "</b><pre>"
                        + html.escape(json.dumps(usage.get("usage", {}), indent=2))
                        + "</pre>"
                    )
            if event in {"review", "supervisor_usage"}:
                self.update_supervisor_totals()
        elif event == "done":
            if payload.get("status") in {"error", "failed"}:
                self.show_error(
                    str(
                        payload.get("error")
                        or payload.get("text")
                        or "Task failed. Inspect Activity."
                    )
                )
        else:
            self.activity.append(
                f"<b>{html.escape(event)}</b><pre>{html.escape(json.dumps(payload, default=str))}</pre>"
            )

    def show_error(self, text: str) -> None:
        self.error_seen = True
        self.error_label.setText(text)
        self.error_label.show()
        self.run_status.setText("! Needs attention")
        self.model_status.setText(text)
        self.media_status.setText(text)
        self.statusBar().showMessage(text)
        self.activity.append(
            "<p style='color:#a34530'><b>Needs attention</b><br>"
            + html.escape(text)
            + "</p>"
        )

    def reload_models(self) -> None:
        if self.worker:
            return
        selected = self.model_combo.currentData()
        try:
            self._model_rows = list(self.models.list_models())
            self.model_combo.clear()
            self.models_list.clear()
            for model in self._model_rows:
                name = model.get("name") or Path(model["path"]).stem
                self.model_combo.addItem(name, model["id"])
                self.models_list.addItem(
                    f"{name}\n{int(model.get('context', 32768)):,} context · GGUF · {model['path']}"
                )
            index = self.model_combo.findData(selected)
            if index >= 0:
                self.model_combo.setCurrentIndex(index)
            if self.models_list.count():
                self.models_list.setCurrentRow(max(0, index))
        except Exception as exc:
            self.show_error(str(exc))

    def model_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._model_rows):
            return
        model = self._model_rows[row]
        self.model_name.setText(model.get("name", ""))
        self.model_context.setValue(int(model.get("context", 32768)))
        self.gpu_layers.setValue(int(model.get("gpu_layers", -1)))
        self.model_device.setText(model.get("device") or "")
        self.mmproj.setText(model.get("mmproj") or "")
        self.model_vision.setChecked(bool(model.get("vision", False)))
        self.model_thinking.setChecked(bool(model.get("thinking", False)))

    def import_model(self) -> None:
        if self.worker:
            self.statusBar().showMessage(
                "Wait for the current operation before changing the model library."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import local model weights", str(Path.home()), "GGUF models (*.gguf)"
        )
        if path:
            try:
                self.models.register(
                    path, Path(path).stem, context=32768, gpu_layers=-1
                )
                self.reload_models()
            except Exception as exc:
                self.show_error(str(exc))

    def save_model(self) -> None:
        row = self.models_list.currentRow()
        if row < 0 or self.worker:
            return
        model = self._model_rows[row]
        try:
            self.models.update(
                model["id"],
                name=self.model_name.text().strip() or model["name"],
                context=self.model_context.value(),
                gpu_layers=self.gpu_layers.value(),
                device=self.model_device.text().strip() or None,
                mmproj=self.mmproj.text().strip() or None,
                vision=self.model_vision.isChecked(),
                thinking=self.model_thinking.isChecked(),
            )
            self.reload_models()
            self.model_status.setText(
                "Profile saved. Reload this model to apply changes."
            )
        except Exception as exc:
            self.show_error(str(exc))

    def cpu_preset(self) -> None:
        """Apply an explicit CPU profile without loading it or disturbing another GPU task."""
        row = self.models_list.currentRow()
        if self.worker or row < 0:
            return
        model = self._model_rows[row]
        try:
            self.models.update(
                model["id"], gpu_layers=0, device="none", context=8192, vision=False
            )
            self.reload_models()
            self.models_list.setCurrentRow(row)
            self.model_status.setText(
                "CPU-only preset saved: 8K context, no GPU layers, vision off. Load the model to apply it. CPU inference is slower."
            )
        except Exception as exc:
            self.show_error(str(exc))

    def load_model(self) -> None:
        row = self.models_list.currentRow()
        if row < 0:
            return
        model = self._model_rows[row]
        self.model_combo.setCurrentIndex(self.model_combo.findData(model["id"]))
        self.start_work(
            lambda cancel, emit: self.models.start(model["id"], cancel, emit),
            lambda result: self.model_status.setText("Model loaded and ready."),
        )

    def unload_model(self) -> None:
        self.start_work(
            lambda cancel, emit: self.models.stop(),
            lambda result: self.model_status.setText(
                "Model unloaded. GPU memory released."
            ),
        )

    def refresh_workflows(self, *_args) -> None:
        if not hasattr(self, "workflow_combo"):
            return
        self._workflow_rows = []
        self.workflow_combo.clear()
        try:
            for item in self.media.list_workflows():
                if item.get("kind", "image") == self.media_kind.currentText():
                    self._workflow_rows.append(item)
                    self.workflow_combo.addItem(
                        item.get("name", item.get("id", "Workflow")),
                        item.get("id", item.get("name")),
                    )
            if not self.workflow_combo.count():
                self.workflow_combo.addItem("No workflow registered", None)
            self.workflow_selected(self.workflow_combo.currentIndex())
        except Exception as exc:
            if hasattr(self, "media_status"):
                self.media_status.setText(str(exc))

    def workflow_selected(self, index: int) -> None:
        """Only expose controls the selected workflow actually accepts."""
        if not hasattr(self, "media_seed") or not 0 <= index < len(self._workflow_rows):
            return
        workflow = self._workflow_rows[index]
        self.media_preset_note.setText(
            "Verified Wan preset: 832 × 480, 33 frames, 20 steps. "
            "Very small resolutions and clips can produce distorted results."
            if workflow.get("name") == "Wan 2.2 5B"
            else ""
        )
        mapping = workflow.get("mapping", {})
        for key, widget in [
            ("negative_prompt", self.media_negative),
            ("width", self.media_width),
            ("height", self.media_height),
            ("steps", self.media_steps),
            ("frames", self.media_frames),
            ("fps", self.media_fps),
            ("seed", self.media_seed),
        ]:
            enabled = (
                key in mapping
                if mapping
                else (
                    key not in {"frames", "fps"}
                    or self.media_kind.currentText() == "video"
                )
            )
            widget.setEnabled(enabled)
            widget.setToolTip(
                "" if enabled else "This workflow does not expose this control."
            )
        image_input = bool(
            workflow.get("supports_image_input") or workflow.get("image_input_node")
        )
        self.media_input.setEnabled(image_input)
        self.media_input_button.setEnabled(image_input)
        self.media_form.setRowVisible(self.media_image_row, image_input)
        self.media_form.setRowVisible(
            self.media_frames, self.media_kind.currentText() == "video"
        )
        self.media_form.setRowVisible(
            self.media_fps, self.media_kind.currentText() == "video"
        )
        self.media_input.setToolTip(
            "Optional starting image for this video."
            if image_input
            else "This workflow does not accept source images."
        )
        try:
            graph = json.loads(Path(workflow["path"]).read_text())
            for key, widget in [
                ("width", self.media_width),
                ("height", self.media_height),
                ("steps", self.media_steps),
                ("frames", self.media_frames),
                ("fps", self.media_fps),
            ]:
                if key in mapping:
                    node, field = mapping[key]
                    widget.setValue(int(graph[node]["inputs"][field]))
        except (KeyError, TypeError, ValueError, OSError):
            pass

    def import_workflow(self) -> None:
        if self.worker:
            self.statusBar().showMessage(
                "Wait for the current operation before importing a workflow."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import a ComfyUI API workflow",
            self.workspace,
            "JSON workflow (*.json)",
        )
        if path:
            try:
                self.media.register_workflow(
                    Path(path).stem, path, self.media_kind.currentText()
                )
                self.refresh_workflows()
                self.media_status.setText(
                    "Workflow imported. Use an API-format workflow with editable prompt/input nodes."
                )
            except Exception as exc:
                self.show_error(str(exc))

    def media_settings(self) -> None:
        """Edit explicit local engine paths without restarting or touching other apps."""
        if self.worker:
            self.statusBar().showMessage(
                "Wait for the active operation before changing engine settings."
            )
            return
        if not hasattr(self.media, "config"):
            self.media_status.setText(
                "This media provider does not expose local engine settings."
            )
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Local render engine settings")
        dialog.resize(670, 350)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        fields = {}
        for key, caption in [
            ("comfy_dir", "ComfyUI directory"),
            ("python", "Python executable"),
            ("url", "Loopback server URL"),
            ("output_dir", "Generated output directory"),
        ]:
            field = QLineEdit(str(self.media.config.get(key, "")))
            fields[key] = field
            form.addRow(caption, field)
        layout.addLayout(form)
        notice = label(
            "Use an existing ComfyUI installation and its Python environment. Changes apply to the next render.",
            "muted",
            True,
        )
        layout.addWidget(notice)

        def save() -> None:
            from urllib.parse import urlparse

            values = {key: field.text().strip() for key, field in fields.items()}
            parsed = urlparse(values["url"])
            if parsed.scheme != "http" or parsed.hostname not in {
                "localhost",
                "127.0.0.1",
            }:
                notice.setText("Use a local HTTP URL, such as http://127.0.0.1:8188.")
                return
            if not (Path(values["comfy_dir"]) / "main.py").is_file() or not os.access(
                values["python"], os.X_OK
            ):
                notice.setText(
                    "Choose a ComfyUI directory containing main.py and an executable Python interpreter."
                )
                return
            if not Path(values["output_dir"]).is_absolute():
                notice.setText("Choose an absolute output directory path.")
                return
            try:
                path = Path(self.media.config_path)
                existing = json.loads(path.read_text()) if path.exists() else {}
                existing["media"] = {**self.media.config, **values}
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(existing, indent=2))
                temporary.replace(path)
                self.media.config.update(values)
                self.media_status.setText("Local render engine settings saved.")
                dialog.accept()
            except (OSError, ValueError) as exc:
                notice.setText(str(exc))

        controls = QHBoxLayout()
        controls.addWidget(button("Cancel", dialog.reject))
        controls.addWidget(button("Save settings", save, True))
        layout.addLayout(controls)
        dialog.exec()

    def pick_media_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a source image",
            self.workspace,
            "Images (*.png *.jpg *.jpeg *.webp)",
        )
        if path:
            self.media_input.setText(path)

    def generate(self) -> None:
        prompt = self.media_prompt.toPlainText().strip()
        if not prompt:
            self.media_status.setText("Write a prompt first.")
            return
        if not self.workflow_combo.currentData():
            self.media_status.setText("Import a local ComfyUI API workflow first.")
            return
        params = {
            "workflow": self.workflow_combo.currentData(),
            "negative_prompt": self.media_negative.text().strip(),
            "fps": self.media_fps.value(),
            "image_path": (
                (self.media_input.text().strip() or None)
                if self.media_input.isEnabled()
                else None
            ),
            "width": self.media_width.value(),
            "height": self.media_height.value(),
            "steps": self.media_steps.value(),
            "frames": self.media_frames.value(),
            "seed": (
                self.media_seed.value()
                if self.media_seed.value() >= 0
                else secrets.randbelow(2147483648)
            ),
        }
        kind = self.media_kind.currentText()
        self.start_work(
            lambda cancel, emit: self.media.generate(
                kind, prompt, params, cancel, emit
            ),
            self.media_finished,
        )

    def media_finished(self, result: Any) -> None:
        files = result.get("files", []) if isinstance(result, dict) else []
        self._media_files = [
            str(item.get("path")) if isinstance(item, dict) else str(item)
            for item in files
        ]
        self.media_results.clear()
        for path in self._media_files:
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            self.media_results.addItem(item)
        if not files:
            self.show_error(
                "The render returned no output files. Inspect the activity log."
            )
            return
        self.media_results.setCurrentRow(0)
        first = self._media_files[0]
        if Path(first).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            self.media_preview.setPixmap(
                QPixmap(first).scaled(
                    self.media_preview.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        else:
            self.media_preview.setText(
                "▶\n\nVideo is ready.\nOpen the output to play it."
            )
        self.media_status.setText(
            f"Saved {len(files)} output file(s). Double-click a result to open it."
        )

    def reload_media_jobs(self, recover_pending: bool = False) -> None:
        """Render disk receipts immediately; defer uncertain engine checks to a worker."""
        if self.worker:
            return
        try:
            rows = self.media.jobs()
            self.render_media_jobs(rows)
            uncertain = {"starting", "running", "queued", "unknown", "interrupted"}
            if recover_pending and any(row.get("status") in uncertain for row in rows):
                QTimer.singleShot(0, self.recover_media_jobs)
        except Exception as exc:
            self.media_status.setText("Could not restore render history: " + str(exc))

    def recover_media_jobs(self) -> None:
        """Reconcile previous renders without blocking Qt or resubmitting a generation."""
        if self.worker:
            self.statusBar().showMessage(
                "Wait for the active operation before checking render history."
            )
            return
        if not hasattr(self.media, "recover_jobs"):
            self.reload_media_jobs()
            return
        self.media_status.setText(
            "Checking previous render jobs with the local engine…"
        )
        self.start_work(
            lambda cancel, emit: self.media.recover_jobs(), self.render_media_jobs
        )

    def render_media_jobs(self, rows: list[dict]) -> None:
        """Show both output links and honest interrupted/unknown recovery states."""
        self.media_results.clear()
        counts: dict[str, int] = {}
        for job in rows[:50]:
            state = str(job.get("status", "unknown"))
            counts[state] = counts.get(state, 0) + 1
            files = job.get("files", [])
            for path in files:
                item = QListWidgetItem(Path(str(path)).name + " · " + state)
                item.setData(Qt.UserRole, str(path))
                item.setToolTip(str(path))
                self.media_results.addItem(item)
            if not files:
                item = QListWidgetItem(
                    str(job.get("kind", "Render")).title() + " · " + state
                )
                item.setData(Qt.UserRole, None)
                detail = str(
                    job.get("recovery_error")
                    or job.get("error")
                    or "No output file has been confirmed."
                )
                item.setToolTip(detail)
                self.media_results.addItem(item)
        if self.media_results.count():
            self.media_results.setCurrentRow(0)
        if counts:
            summary = "Render history: " + ", ".join(
                f"{count} {state}" for state, count in counts.items()
            )
            if counts.get("unknown") or counts.get("interrupted"):
                summary += ". Unconfirmed jobs were not automatically restarted. Hover over a row for details."
            self.media_status.setText(summary)
        else:
            self.media_status.setText(
                "No previous render jobs. Choose a workflow to begin."
            )

    def open_selected_media(self) -> None:
        item = self.media_results.currentItem()
        if item:
            self.open_media_file(item.data(Qt.UserRole))

    def open_media_file(self, path: str) -> None:
        """Use Qt Multimedia for video, with the OS player as a transparent fallback."""
        if not path:
            item = self.media_results.currentItem()
            if item:
                self.media_status.setText(item.toolTip())
            return
        if Path(path).suffix.lower() not in {".mp4", ".webm", ".mov", ".mkv", ".avi"}:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
            return
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget

            dialog = QDialog(self)
            dialog.setWindowTitle(Path(path).name)
            dialog.resize(920, 650)
            layout = QVBoxLayout(dialog)
            video = QVideoWidget()
            layout.addWidget(video, 1)
            player = QMediaPlayer(dialog)
            audio = QAudioOutput(dialog)
            player.setAudioOutput(audio)
            player.setVideoOutput(video)
            player.setSource(QUrl.fromLocalFile(path))
            row = QHBoxLayout()
            row.addWidget(button("Play", player.play))
            row.addWidget(button("Pause", player.pause))
            row.addWidget(
                button(
                    "Open externally",
                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(path)),
                )
            )
            layout.addLayout(row)
            player.errorOccurred.connect(
                lambda error, message: self.statusBar().showMessage(
                    "Video playback: " + message
                )
            )
            player.play()
            dialog.exec()
            player.stop()
        except ImportError:
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _preferences_path(self) -> Path:
        root = Path(
            getattr(
                self.store,
                "root",
                os.environ.get(
                    "LOCAII_HOME", str(Path(__file__).resolve().parents[2] / ".local")
                ),
            )
        )
        root.mkdir(parents=True, exist_ok=True)
        return root / "ui-settings.json"

    def _load_preferences(self) -> None:
        try:
            data = json.loads(self._preferences_path().read_text())
            for key, widget in [
                ("workspace_writes", self.workspace_writes),
                ("allow_terminal", self.allow_terminal),
                ("read_all_files", self.read_all_files),
                ("remote_enabled", self.remote_enabled),
            ]:
                if key in data:
                    widget.setChecked(bool(data[key]))
            for key, widget in [
                ("supervisor_every", self.supervisor_every),
                ("max_iterations", self.max_iterations),
                ("supervisor_token_budget", self.token_budget),
            ]:
                if key in data:
                    widget.setValue(int(data[key]))
            self.supervisor_model.setText(data.get("supervisor_model", ""))
            for key, widget in [
                ("supervisor_provider", self.supervisor_provider),
                ("supervisor_scope", self.supervisor_scope),
                ("supervisor_cli", self.supervisor_cli),
                ("model_id", self.model_combo),
            ]:
                index = widget.findData(data.get(key))
                if index >= 0:
                    widget.setCurrentIndex(index)
            if Path(data.get("workspace", "")).is_dir():
                self.set_workspace(data["workspace"])
        except (FileNotFoundError, ValueError, KeyError, OSError):
            pass

    def _save_supervisor_choice(self) -> None:
        """Remember the launch choice without persisting unrelated permission changes."""
        try:
            path = self._preferences_path()
            data = json.loads(path.read_text()) if path.exists() else {}
            data["supervisor_cli"] = self.supervisor_cli.currentData()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(data, indent=2))
            temporary.replace(path)
        except (OSError, ValueError, TypeError) as exc:
            self.statusBar().showMessage(
                "Supervisor opened, but its preference could not be saved: " + str(exc)
            )

    def save_preferences(self) -> None:
        try:
            path = self._preferences_path()
            temporary = path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(
                    {
                        **self.options(),
                        "workspace": self.workspace,
                        "supervisor_cli": self.supervisor_cli.currentData(),
                    },
                    indent=2,
                )
            )
            temporary.replace(path)
            self.statusBar().showMessage(
                "Preferences saved. They apply to the next run."
            )
        except OSError as exc:
            self.show_error(str(exc))

    def export_packet(self) -> None:
        if not self.session_id:
            self.statusBar().showMessage(
                "Start or select a task before exporting a review packet."
            )
            return
        self.start_work(
            lambda cancel, emit: self.engine.export_supervision(self.session_id),
            self.packet_exported,
        )

    def packet_exported(self, path: Any) -> None:
        path = str(path)
        self.review_view.setPlainText(
            "Review packet saved:\n"
            + path
            + "\n\nShare this file with your assistant, then import its structured review."
        )
        QApplication.clipboard().setText(path)
        self.statusBar().showMessage("Review packet path copied to clipboard.")

    def import_packet(self) -> None:
        if not self.session_id:
            self.statusBar().showMessage("Select the task that this review belongs to.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import structured supervisor review",
            self.workspace,
            "JSON review (*.json)",
        )
        if path:
            self.start_work(
                lambda cancel, emit: self.engine.import_supervision(
                    self.session_id, path
                ),
                lambda result: self.review_view.setPlainText(
                    json.dumps(result, ensure_ascii=False, indent=2, default=str)
                ),
            )

    def closeEvent(self, event) -> None:
        """Do not destroy a running QThread; request a cooperative stop before exit."""
        self._steering_restart_generation += 1
        if self.worker:
            self._pending_close = True
            self.stop()
            self.statusBar().showMessage("Finishing the stop request before closing…")
            event.ignore()
        else:
            if self._delegation_hub is not None:
                self.disconnect_supervisor()
            event.accept()
