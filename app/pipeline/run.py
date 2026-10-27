"""Run mesh → layout → record-pm → gcode for one subject."""

from __future__ import annotations

import argparse

from app import paths
from app.subject import pm_is_measured

STAGES = ("mesh", "layout", "record-pm", "gcode")


def active_stages(from_stage: str, to_stage: str) -> list[str]:
    start = STAGES.index(from_stage)
    end = STAGES.index(to_stage)
    if end < start:
        raise ValueError(f"--to {to_stage} is before --from {from_stage}")
    return list(STAGES[start : end + 1])


def run_pipeline(args: argparse.Namespace) -> int:
    from app.cli import cmd_gcode, cmd_layout, cmd_mesh, cmd_record_pm

    stages = active_stages(args.from_stage, args.to_stage)
    print(f"subject {args.subject}: {' → '.join(stages)}")
    for stage in stages:
        if stage == "record-pm" and not args.force_record_pm:
            if pm_is_measured(paths.pm_yaml(args.subject)):
                print(f"record-pm already measured: {paths.pm_yaml(args.subject)}")
                continue
        print(f"\n== {stage} ==")
        if stage == "mesh":
            code = cmd_mesh(args)
        elif stage == "layout":
            code = cmd_layout(args)
        elif stage == "record-pm":
            record_args = argparse.Namespace(
                subject=args.subject,
                bind_ip=args.bind_ip,
                port=args.port,
                stale_ms=args.stale_ms,
                force=args.force_record_pm,
            )
            code = cmd_record_pm(record_args)
        else:
            code = cmd_gcode(args)
        if code != 0:
            print(f"{stage} failed ({code})")
            return code
    return 0
