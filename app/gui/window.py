"""Tk control panel. Launches ``python -m app run`` in a child process."""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

from app import paths
from app.gui.status import StageStatus, run_argv, stage_statuses
from app.pipeline.run import STAGES


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(paths.REPO_ROOT))
    except ValueError:
        return str(path)


class PipelineWindow:
    """Subject, stage readiness, and buttons that start ``run``."""

    def __init__(self, subject: int = 2) -> None:
        self.root = tk.Tk()
        self.root.title("Layout pipeline")
        self.root.geometry("1100x720")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._proc: subprocess.Popen[str] | None = None
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader: threading.Thread | None = None

        intro = ttk.Label(
            self.root,
            wraplength=1060,
            text=(
                "Starts python -m app run in a separate process. "
                "Align, fiducials, electrodes, and the 3D viewers stay the current PyVista windows."
            ),
        )
        intro.pack(fill="x", padx=8, pady=(8, 4))

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=4)
        ttk.Label(controls, text="Subject").pack(side="left")
        self._subject = tk.IntVar(value=int(subject))
        spin = ttk.Spinbox(
            controls,
            from_=1,
            to=9999,
            width=6,
            textvariable=self._subject,
            command=self.refresh,
        )
        spin.pack(side="left", padx=(4, 12))
        spin.bind("<FocusOut>", lambda _e: self.refresh())
        spin.bind("<Return>", lambda _e: self.refresh())

        self._polish = tk.BooleanVar(value=True)
        self._visualize = tk.BooleanVar(value=True)
        self._rotate = tk.BooleanVar(value=False)
        self._simulate = tk.BooleanVar(value=False)
        ttk.Checkbutton(controls, text="Polish", variable=self._polish, command=self.refresh).pack(
            side="left", padx=4
        )
        ttk.Checkbutton(controls, text="Visualize", variable=self._visualize).pack(
            side="left", padx=4
        )
        ttk.Checkbutton(controls, text="Hub angle search", variable=self._rotate).pack(
            side="left", padx=4
        )
        ttk.Checkbutton(
            controls,
            text="End at simulate",
            variable=self._simulate,
            command=self.refresh,
        ).pack(side="left", padx=4)

        self._stop_btn = ttk.Button(controls, text="Stop", command=self.stop)
        self._stop_btn.pack(side="right")
        ttk.Button(controls, text="Refresh", command=self.refresh).pack(side="right", padx=4)

        paned = ttk.Panedwindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        table_wrap = ttk.Frame(paned)
        self._canvas = tk.Canvas(table_wrap, highlightthickness=0)
        scroll = ttk.Scrollbar(table_wrap, orient="vertical", command=self._canvas.yview)
        self._rows = ttk.Frame(self._canvas)
        self._rows.bind(
            "<Configure>",
            lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")),
        )
        self._canvas_window = self._canvas.create_window((0, 0), window=self._rows, anchor="nw")
        self._canvas.configure(yscrollcommand=scroll.set)
        self._canvas.bind(
            "<Configure>",
            lambda event: self._canvas.itemconfigure(self._canvas_window, width=event.width),
        )
        self._canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        paned.add(table_wrap, weight=3)

        self._log = scrolledtext.ScrolledText(paned, height=12, font=("Consolas", 10))
        paned.add(self._log, weight=2)

        self._status = ttk.Label(self.root, text="Idle")
        self._status.pack(fill="x", padx=8, pady=(0, 8))
        self.refresh()

    def subject(self) -> int:
        try:
            value = int(self._subject.get())
        except (tk.TclError, ValueError):
            return 2
        return min(9999, max(1, value))

    def end_stage(self) -> str:
        return "simulate" if self._simulate.get() else "gcode"

    def refresh(self) -> None:
        for child in self._rows.winfo_children():
            child.destroy()
        running = self._running()
        header = ttk.Frame(self._rows)
        header.pack(fill="x", pady=(0, 4))
        for text, width in (("Stage", 16), ("Kind", 14), ("Status", 10)):
            ttk.Label(header, text=text, width=width).pack(side="left")
        ttk.Label(header, text="Artifact").pack(side="left", fill="x", expand=True)

        for row in stage_statuses(self.subject(), polish=self._polish.get()):
            self._add_row(row, running)

        self._stop_btn.state(["!disabled"] if running else ["disabled"])

    def _add_row(self, row: StageStatus, running: bool) -> None:
        line = ttk.Frame(self._rows)
        line.pack(fill="x", pady=1)
        ttk.Label(line, text=row.name, width=16).pack(side="left")
        kind = "interactive" if row.interactive else "automated"
        ttk.Label(line, text=kind, width=14).pack(side="left")
        status = ttk.Label(line, text=row.label, width=10)
        if row.skipped:
            status.configure(foreground="#666666")
        elif row.ready:
            status.configure(foreground="#1b7f3a")
        status.pack(side="left")
        ttk.Label(line, text=_display_path(row.artifact)).pack(side="left", fill="x", expand=True)

        enabled = not running and not row.skipped
        from_ok = enabled and STAGES.index(row.name) <= STAGES.index(self.end_stage())
        this_stage = ttk.Button(
            line, text="This stage", command=lambda name=row.name: self.start_stage(name)
        )
        from_here = ttk.Button(
            line, text="Run from here", command=lambda name=row.name: self.start_from(name)
        )
        this_stage.pack(side="right")
        from_here.pack(side="right", padx=(0, 4))
        if not enabled:
            this_stage.state(["disabled"])
        if not from_ok:
            from_here.state(["disabled"])

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
                polish=bool(self._polish.get()),
                visualize=bool(self._visualize.get()),
                rotate=bool(self._rotate.get()),
            )
        except ValueError as exc:
            messagebox.showwarning("Cannot start", str(exc), parent=self.root)
            return
        self._append("$ " + " ".join(argv) + "\n\n")
        self._status.configure(text=f"Running {from_stage} → {to_stage}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=paths.REPO_ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
        except OSError as exc:
            self._append(f"Failed to start: {exc}\n")
            self._status.configure(text="Failed to start")
            self._proc = None
            self.refresh()
            return
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        self.root.after(100, self._drain)
        self.refresh()

    def _read_output(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            self._lines.put(None)
            return
        for line in proc.stdout:
            self._lines.put(line)
        proc.wait()
        self._lines.put(None)

    def _drain(self) -> None:
        finished = False
        try:
            while True:
                item = self._lines.get_nowait()
                if item is None:
                    finished = True
                    break
                self._append(item)
        except queue.Empty:
            pass
        if finished:
            code = self._proc.returncode if self._proc is not None else 1
            self._append(f"\nProcess exited {code}.\n")
            self._status.configure(text=f"Exited {code}")
            self._proc = None
            self.refresh()
            return
        if self._running():
            self.root.after(100, self._drain)

    def _append(self, text: str) -> None:
        self._log.insert("end", text)
        self._log.see("end")

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            self._append("\nStopping pipeline…\n")
            proc.kill()

    def _running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _on_close(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()
        self.root.destroy()


def launch(subject: int | None = None) -> int:
    """Open the control panel. Returns 0 when the window closes."""
    window = PipelineWindow(subject=2 if subject is None else subject)
    window.root.mainloop()
    return 0
