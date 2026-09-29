"""PySide6 control panel. Launches ``python -m app run`` via QProcess."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtGui import QColor, QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import paths
from app.gui.status import StageStatus, run_argv, stage_statuses
from app.pipeline.run import STAGES


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


class PipelineWindow(QMainWindow):
    """Subject, stage readiness, and buttons that start ``run``."""

    def __init__(self, subject: int = 2) -> None:
        super().__init__()
        self.setWindowTitle("Layout pipeline")
        self.resize(1100, 720)

        self._proc = QProcess(self)
        self._proc.setWorkingDirectory(str(paths.REPO_ROOT))
        self._proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self._proc.setProcessEnvironment(env)
        self._proc.readyReadStandardOutput.connect(self._append_output)
        self._proc.finished.connect(self._on_finished)
        self._proc.errorOccurred.connect(self._on_error)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        intro = QLabel(
            "Starts python -m app run in a separate process. "
            "Align, fiducials, electrodes, and the 3D viewers stay the current PyVista windows."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Subject"))
        self._subject = QSpinBox()
        self._subject.setRange(1, 9999)
        self._subject.setValue(int(subject))
        self._subject.valueChanged.connect(lambda _v: self.refresh())
        controls.addWidget(self._subject)

        self._polish = QCheckBox("Polish")
        self._polish.setChecked(True)
        self._polish.toggled.connect(lambda _v: self.refresh())
        controls.addWidget(self._polish)

        self._visualize = QCheckBox("Visualize")
        self._visualize.setChecked(True)
        self._visualize.setToolTip(
            "After synthesize, polish, and smooth: 2D PNG and the interactive 3D view"
        )
        controls.addWidget(self._visualize)

        self._rotate = QCheckBox("Hub angle search")
        self._rotate.setToolTip("Synthesize: ±36° search around the fiducial hub clicks")
        controls.addWidget(self._rotate)

        self._simulate = QCheckBox("End at simulate")
        self._simulate.setToolTip("Run from here stops at simulate instead of gcode")
        controls.addWidget(self._simulate)

        controls.addStretch(1)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        controls.addWidget(self._refresh_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self.stop)
        controls.addWidget(self._stop_btn)
        layout.addLayout(controls)

        splitter = QSplitter()
        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Stage", "Kind", "Status", "Artifact", ""])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        splitter.addWidget(self._table)

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("monospace"))
        self._log.setPlaceholderText("Pipeline log")
        splitter.addWidget(self._log)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        self.statusBar().showMessage("Idle")
        self.refresh()

    def subject(self) -> int:
        return int(self._subject.value())

    def end_stage(self) -> str:
        return "simulate" if self._simulate.isChecked() else "gcode"

    def refresh(self) -> None:
        rows = stage_statuses(self.subject(), polish=self._polish.isChecked())
        running = self._running()
        self._table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            self._set_text(index, 0, row.name)
            self._set_text(index, 1, "interactive" if row.interactive else "automated")
            status = QTableWidgetItem(row.label)
            if row.skipped:
                status.setForeground(QColor("#666666"))
            elif row.ready:
                status.setForeground(QColor("#1b7f3a"))
            self._table.setItem(index, 2, status)
            self._set_text(index, 3, _display_path(row.artifact))
            self._table.setCellWidget(index, 4, self._actions(row, running))
        self._subject.setEnabled(not running)
        self._polish.setEnabled(not running)
        self._simulate.setEnabled(not running)
        self._stop_btn.setEnabled(running)

    def _set_text(self, row: int, column: int, text: str) -> None:
        self._table.setItem(row, column, QTableWidgetItem(text))

    def _actions(self, row: StageStatus, running: bool) -> QWidget:
        box = QWidget()
        layout = QHBoxLayout(box)
        layout.setContentsMargins(4, 2, 4, 2)
        from_here = QPushButton("Run from here")
        this_stage = QPushButton("This stage")
        enabled = not running and not row.skipped
        from_ok = enabled and STAGES.index(row.name) <= STAGES.index(self.end_stage())
        from_here.setEnabled(from_ok)
        this_stage.setEnabled(enabled)
        from_here.clicked.connect(lambda _checked=False, name=row.name: self.start_from(name))
        this_stage.clicked.connect(lambda _checked=False, name=row.name: self.start_stage(name))
        layout.addWidget(from_here)
        layout.addWidget(this_stage)
        return box

    def start_from(self, stage: str) -> None:
        self._start(stage, self.end_stage())

    def start_stage(self, stage: str) -> None:
        self._start(stage, stage)

    def _start(self, from_stage: str, to_stage: str) -> None:
        if self._running():
            return
        try:
            argv = run_argv(
                target=self.subject(),
                from_stage=from_stage,
                to_stage=to_stage,
                polish=self._polish.isChecked(),
                visualize=self._visualize.isChecked(),
                rotate=self._rotate.isChecked(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Cannot start", str(exc))
            return
        self._log.appendPlainText("$ " + " ".join(argv) + "\n")
        self.statusBar().showMessage(f"Running {from_stage} → {to_stage}")
        self._proc.start(argv[0], argv[1:])
        self.refresh()

    def stop(self) -> None:
        if self._running():
            self._log.appendPlainText("\nStopping pipeline…\n")
            self._proc.kill()

    def _running(self) -> bool:
        return self._proc.state() != QProcess.ProcessState.NotRunning

    def _append_output(self) -> None:
        data = bytes(self._proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self._log.moveCursor(QTextCursor.MoveOperation.End)
        self._log.insertPlainText(data)
        self._log.moveCursor(QTextCursor.MoveOperation.End)

    def _on_finished(self, exit_code: int, _status) -> None:
        self._log.appendPlainText(f"\nProcess exited {exit_code}.\n")
        self.statusBar().showMessage(f"Exited {exit_code}")
        self.refresh()

    def _on_error(self, error) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._log.appendPlainText(f"Failed to start: {self._proc.errorString()}\n")
            self.statusBar().showMessage("Failed to start")
            self.refresh()


def launch(subject: int | None = None) -> int:
    """Open the control panel. Returns the Qt application exit code."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = PipelineWindow(subject=2 if subject is None else subject)
    window.show()
    return int(app.exec())
