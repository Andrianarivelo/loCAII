"""Application entry point and explicit, GPU-free visual fixture mode."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from PySide6.QtCore import QLockFile, QTimer
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import QApplication


def build_services(home: Path):
    """Compose replaceable services using one private application-state directory."""
    from .storage import Store
    from .models import ModelManager
    from .media import MediaService
    from .engine import Engine
    from .supervision import Supervisor

    home.mkdir(parents=True, exist_ok=True)
    store = Store(home)
    models = ModelManager(home / "models.json", home / "model-runtime")
    media = MediaService(home / "media.json", home / "media-runtime", models)
    engine = Engine(store, models, Supervisor(store))
    engine.media = media
    return store, engine, models, media


def populate_demo(window) -> None:
    """Show a clearly labelled fixture for deterministic visual inspection only."""
    window.clear_messages()
    window.chat_heading.setText("A clearer picture of your workspace.")
    window.add_message(
        "user",
        "Review my project folder, identify the key files, and propose a small, testable next step.",
    )
    window.add_message(
        "assistant",
        "I've mapped the project and saved a working checkpoint.\n\n### A good place to start\n\nThe application is split into **models**, **tools**, **memory**, and the **desktop interface**. Each can evolve independently.\n\n1. Check that a model can complete a small file task.\n2. Verify the result against the original request.\n3. Keep a checkpoint so the next iteration starts with evidence.\n\nThe next useful step is a focused end-to-end test using a temporary workspace.",
    )
    window.activity.setHtml(
        "<p style='color:#426c5b'><b>VISUAL DEMO · no tools executed</b></p><p><b>list_directory</b> · example receipt</p><pre>src/\ntests/\ndocs/\nCONCEPT.md</pre><p><b>Plan checkpoint</b></p><p>Inspect → Implement → Verify</p><p style='color:#74816c'>This screen is a presentation fixture. Switch to a real task to run a model.</p>"
    )
    window.memory_view.setMarkdown(
        "# Visual demo checkpoint\n\nThis is a UI fixture, not a completed task.\n\n## Next step\nRun a synthetic end-to-end file task and inspect its receipts."
    )
    window.run_status.setText("◈ Visual demo")


def main(argv: list[str] | None = None) -> int:
    """Start the desktop, optionally capture it without invoking model inference."""
    parser = argparse.ArgumentParser(description="loCAII local AI desktop")
    parser.add_argument(
        "--screenshot",
        type=Path,
        help="Save a screenshot and exit without loading a model",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Show clearly labelled deterministic example content",
    )
    parser.add_argument(
        "--home", type=Path, help="Override private application state directory"
    )
    args = parser.parse_args(argv)
    home = args.home or Path(
        os.environ.get(
            "LOCAII_HOME", str(Path(__file__).resolve().parents[2] / ".local")
        )
    )
    os.environ["LOCAII_HOME"] = str(home)
    QApplication.setApplicationName("loCAII")
    QApplication.setOrganizationName("Andrianarivelo")
    app = QApplication.instance() or QApplication(sys.argv[:1])
    # A second desktop launch must not reset another window's durable task state.
    home.mkdir(parents=True, exist_ok=True)
    instance_lock = QLockFile(str(home / "desktop.lock"))
    instance_name = (
        "locaii-"
        + str(os.getuid())
        + "-"
        + hashlib.sha256(str(home.resolve()).encode()).hexdigest()[:20]
    )
    if not instance_lock.tryLock(0):
        existing = QLocalSocket()
        existing.connectToServer(instance_name)
        if existing.waitForConnected(1000):
            existing.write(b"activate")
            existing.waitForBytesWritten(250)
            existing.disconnectFromServer()
            return 0
        print(
            "loCAII is already starting or running for this state directory.",
            file=sys.stderr,
        )
        return 1
    server = QLocalServer()
    server.setSocketOptions(QLocalServer.UserAccessOption)
    if not server.listen(instance_name):
        QLocalServer.removeServer(instance_name)
        if not server.listen(instance_name):
            instance_lock.unlock()
            raise RuntimeError(
                "Could not create the desktop activation socket: "
                + server.errorString()
            )
    from .ui import MainWindow

    store, engine, models, media = build_services(home)
    window = MainWindow(store, engine, models, media)

    def activate_existing() -> None:
        connection = server.nextPendingConnection()
        if connection:
            connection.disconnectFromServer()
            connection.deleteLater()
        window.showNormal()
        window.raise_()
        window.activateWindow()

    server.newConnection.connect(activate_existing)
    window.show()
    if args.demo:
        populate_demo(window)
    if args.screenshot:

        def capture() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(args.screenshot)):
                print(f"Could not save screenshot: {args.screenshot}", file=sys.stderr)
                app.exit(1)
            else:
                app.quit()

        QTimer.singleShot(1200, capture)
    # Only the owned backend is released on application shutdown.
    app.aboutToQuit.connect(models.stop)
    code = app.exec()
    server.close()
    instance_lock.unlock()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
