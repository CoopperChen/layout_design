"""Artifact readiness and ``run`` argv for the desktop control panel.

No Qt imports. The window reads these rows and starts the child process.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from app import paths
from app.pipeline.run import STAGES, PipelinePaths

# Stages that open a PyVista (or CNC) window. Layout visualize windows are
# controlled by the visualize checkbox, not by this set.
INTERACTIVE_STAGES: frozenset[str] = frozenset(
    {
        "reconstruct",
        "align-obj",
        "fiducials",
        "electrodes",
        "record-pm",
        "simulate",
    }
)


@dataclass(frozen=True)
class StageStatus:
    """One pipeline stage and whether its output is already on disk."""

    name: str
    interactive: bool
    ready: bool
    skipped: bool
    artifact: Path

    @property
    def label(self) -> str:
        if self.skipped:
            return "skipped"
        if self.ready:
            return "ready"
        return "missing"


def polished_layout_path(layout: Path) -> Path:
    """``synth_s{id}.json`` → ``synth_s{id}_repaired.json``."""
    if layout.stem.endswith("_repaired"):
        return layout
    return layout.parent / f"{layout.stem}_repaired.json"


def _artifact(name: str, pp: PipelinePaths) -> tuple[Path, bool]:
    if name == "reconstruct":
        path = paths.raw_scan(pp.target)
        return path, path.is_file()
    if name == "clear-islands":
        return pp.cleaned, pp.cleaned.is_file()
    if name == "align-obj":
        path = paths.aligned_textured_obj(pp.target)
        return path, path.is_file()
    if name == "fiducials":
        return pp.fiducials, pp.fiducials.is_file()
    if name == "cz":
        return pp.cz, pp.cz.is_file()
    if name == "electrodes":
        return pp.electrodes, pp.electrodes.is_file()
    if name == "synthesize":
        return pp.layout, pp.layout.is_file()
    if name == "polish":
        path = polished_layout_path(pp.layout)
        return path, path.is_file()
    if name == "smooth":
        return pp.smooth, pp.smooth.is_file()
    if name == "bundle":
        return pp.bundle, pp.bundle.is_dir()
    if name == "print-config":
        return pp.print_config, pp.print_config.is_file()
    if name == "record-pm":
        from app.postprocess.print_config import pm_is_measured

        return pp.print_config, pm_is_measured(pp.print_config)
    if name in {"gcode", "simulate"}:
        # simulate has no saved artifact; the row tracks its G-code input.
        return pp.gcode, pp.gcode.is_file()
    raise ValueError(f"Unknown stage {name!r}")


def stage_statuses(target: int, *, polish: bool = True) -> list[StageStatus]:
    """Readiness for every stage in ``STAGES`` for ``target``."""
    pp = PipelinePaths.for_target(target)
    rows: list[StageStatus] = []
    for name in STAGES:
        artifact, ready = _artifact(name, pp)
        skipped = name == "polish" and not polish
        rows.append(
            StageStatus(
                name=name,
                interactive=name in INTERACTIVE_STAGES,
                ready=ready and not skipped,
                skipped=skipped,
                artifact=artifact,
            )
        )
    return rows


def _check_stage(name: str) -> None:
    if name not in STAGES:
        raise ValueError(f"Unknown stage {name!r}; choose from: {', '.join(STAGES)}")


def run_argv(
    *,
    target: int,
    from_stage: str,
    to_stage: str,
    polish: bool = True,
    visualize: bool = True,
    rotate: bool = False,
) -> list[str]:
    """Build ``python -m app run`` for a child process.

    ``from_stage`` / ``to_stage`` match ``run --from`` / ``--to``.
    """
    _check_stage(from_stage)
    _check_stage(to_stage)
    if STAGES.index(from_stage) > STAGES.index(to_stage):
        raise ValueError(f"--from {from_stage} is after --to {to_stage}")
    if not polish and from_stage == "polish" and to_stage == "polish":
        raise ValueError("polish is off; choose another stage")

    argv = [
        sys.executable,
        "-m",
        "app",
        "run",
        "--target",
        str(int(target)),
        "--from",
        from_stage,
        "--to",
        to_stage,
    ]
    if not polish:
        argv.append("--no-polish")
    if not visualize:
        argv.append("--no-visualize")
    if rotate:
        argv.append("--rotate")
    return argv
