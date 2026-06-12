"""CLI for G-code 3D simulation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app import paths
from app.runtime import setup_runtime


def _parse_layers(value: str) -> set[str]:
    allowed = {"mesh", "landmarks", "origin", "cnc", "tip", "arm", "programmed"}
    layers = {part.strip().lower() for part in value.split(",") if part.strip()}
    unknown = layers - allowed
    if unknown:
        raise ValueError(f"Unknown layers: {sorted(unknown)}; allowed: {sorted(allowed)}")
    return layers


def simulate_gcode(
    gcode: str | Path,
    bundle: str | Path,
    *,
    pm_file: str | Path | None = None,
    machine_config: str | Path | None = None,
    rot0y_deg: float = 0.0,
    rot0z_deg: float = 0.0,
    layers: set[str] | None = None,
    animate: bool = False,
    verbose: bool = False,
) -> None:
    setup_runtime()
    import numpy as np

    from app.postprocess.bundle.load import load_bundle
    from app.postprocess.gcode.config_loader import load_machine_config
    from app.postprocess.print_config import load_physical_landmarks, resolve_pm_config
    from app.simulator.kinematics.inverse import decode_postprocessor_paths
    from app.simulator.kinematics.machine_execute import (
        forward_states_from_gcode,
        rigid_geometry_checks,
    )
    from app.simulator.parser import parse_gcode_file
    from app.simulator.registration.mesh import register_mesh_full
    from app.simulator.viewer import SimulationScene, show_simulation

    gcode_path = Path(gcode)
    if not gcode_path.is_absolute():
        gcode_path = paths.REPO_ROOT / gcode_path

    bundle_path = Path(bundle)
    if not bundle_path.is_absolute():
        bundle_path = paths.REPO_ROOT / bundle_path

    subject_bundle = load_bundle(bundle_path)
    pm_path = resolve_pm_config(bundle_path, pm_file=pm_file)
    pm = load_physical_landmarks(pm_path)

    machine_path = (
        Path(machine_config)
        if machine_config
        else paths.postprocessor_machine_config()
    )
    if not machine_path.is_absolute():
        machine_path = paths.REPO_ROOT / machine_path
    machine = load_machine_config(machine_path)

    registration = register_mesh_full(
        subject_bundle,
        pm,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
        rot0y_deg=rot0y_deg,
        rot0z_deg=rot0z_deg,
    )
    mesh_points = registration.mesh_points
    mesh_faces = registration.mesh_faces
    landmarks = registration.calibration_registered

    gcode_matrix = parse_gcode_file(gcode_path)
    programmed_xyz = gcode_matrix[:, :3].copy()

    # Runtime: controller moves C pivot to programmed X,Y,Z (no postprocessor pass).
    cnc_xyz, b_pivot_xyz, tip_xyz, states = forward_states_from_gcode(
        gcode_matrix,
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
    )

    if verbose:
        scalp, print_tips, scalp_lm, _print_tips_lm = decode_postprocessor_paths(
            gcode_matrix,
            machine,
            mesh_points_machine=mesh_points,
            mesh_faces=mesh_faces,
        )
        checks = rigid_geometry_checks(
            states,
            a_mm=machine.a_mm,
            d_mm=machine.d_mm,
            b0_deg=machine.b0_deg,
            c0_deg=machine.c0_deg,
        )
        from scipy.spatial import cKDTree

        from app.postprocess.gcode.kinematics.machine_fk import (
            machine_to_registration_frame,
        )

        fk_kw = {
            "a_mm": machine.a_mm,
            "d_mm": machine.d_mm,
            "calgap_z_mm": machine.calgap_z_mm,
        }
        mesh_landmark = machine_to_registration_frame(mesh_points, **fk_kw)
        mesh_d_lm = cKDTree(mesh_landmark).query(scalp_lm)[0]
        mesh_d = cKDTree(mesh_points).query(scalp)[0]
        tip_mesh_d = cKDTree(mesh_points).query(tip_xyz)[0]
        print_tip_mesh_d = cKDTree(mesh_points).query(print_tips)[0]
        fk_vs_print = np.linalg.norm(tip_xyz - print_tips, axis=1)
        standoff = np.linalg.norm(print_tips - scalp, axis=1)

        print(
            f"simulate-gcode (forward FK): {len(gcode_matrix)} steps | "
            f"rigid arm a={machine.a_mm} mm (max err {checks['arm_length_max_err']:.2e}) | "
            f"tool d={machine.d_mm} mm (max err {checks['tool_length_max_err']:.2e}) | "
            f"arm·tool max |dot| {checks['perp_dot_max']:.2e}"
        )
        print(
            f"registration: landmark fit max {registration.landmark_fit_error_mm:.2f} mm | "
            f"decode scalp→mesh (landmark frame) median {float(np.median(mesh_d_lm)):.2f} mm"
        )
        print(
            f"decode scalp→mesh (machine frame) median {float(np.median(mesh_d)):.2f} mm | "
            f"decode tip→mesh median {float(np.median(print_tip_mesh_d)):.2f} mm "
            f"(standoff median {float(np.median(standoff)):.2f} mm, gap={machine.gap_size_mm}) | "
            f"FK tip→mesh median {float(np.median(tip_mesh_d)):.2f} mm | "
            f"FK vs decode tip median {float(np.median(fk_vs_print)):.2f} mm"
        )
        if float(np.median(mesh_d_lm)) > 1.0:
            print(
                "warning: G-code may not match this bundle/pm/rot0 — "
                "regenerate with convert-gcode using the same --pm-file and --rot0 flags",
                file=sys.stderr,
            )
        if registration.landmark_fit_error_mm > 2.0:
            print(
                "warning: measured pm differs from registered digital landmarks "
                f"(max {registration.landmark_fit_error_mm:.1f} mm); "
                "remeasure pm or regenerate G-code with convert-gcode",
                file=sys.stderr,
            )
        if float(np.median(fk_vs_print)) > 5.0:
            print(
                "note: rigid FK tip from C pivot differs from postprocessor decode — "
                "expected when runtime arm geometry differs from offline compensation",
            )

    names = subject_bundle.landmark_names or [
        "central",
        "left",
        "back",
    ]

    scene = SimulationScene(
        mesh_points=mesh_points,
        mesh_faces=mesh_faces,
        landmarks=landmarks,
        landmark_names=[n.replace("Landmark(", "").replace(")", "") for n in names],
        cnc_path=cnc_xyz,
        programmed_path=programmed_xyz,
        b_pivot_path=b_pivot_xyz,
        b_angles=gcode_matrix[:, 3],
        c_angles=gcode_matrix[:, 4],
        tip_path=tip_xyz,
        markers=gcode_matrix[:, 6],
        a_mm=machine.a_mm,
        d_mm=machine.d_mm,
        calgap_z_mm=machine.calgap_z_mm,
        b0_deg=machine.b0_deg,
        c0_deg=machine.c0_deg,
        layers=layers or {"mesh", "landmarks", "origin", "tip", "arm"},
    )

    title = f"Subject {subject_bundle.subject_id} — {gcode_path.name}"
    show_simulation(scene, title=title, animate=animate)


def cmd_simulate_gcode(args: argparse.Namespace) -> int:
    try:
        simulate_gcode(
            args.gcode,
            args.bundle,
            pm_file=args.pm_file,
            machine_config=args.machine_config,
            rot0y_deg=args.rot0y,
            rot0z_deg=args.rot0z,
            layers=_parse_layers(args.layers),
            animate=args.animate,
            verbose=args.verbose,
        )
        return 0
    except ImportError as exc:
        print(
            f"{exc}\nInstall PyVista and Open3D (see layout_design pyproject.toml).",
            file=sys.stderr,
        )
        return 1
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1


def add_simulate_gcode_parser(sub: argparse._SubParsersAction) -> None:
    sg = sub.add_parser(
        "simulate-gcode",
        help="3D viewer: forward FK — G-code X,Y,Z = C pivot, B,C = arm/tool",
    )
    sg.add_argument("--gcode", required=True, help="G-code .txt file")
    sg.add_argument("--bundle", required=True, help="Subject bundle directory")
    sg.add_argument(
        "--pm-file",
        help="Physical landmarks YAML (default: config/postprocessor/subjects/subject_{id}.yaml)",
    )
    sg.add_argument(
        "--machine-config",
        help="Machine YAML (default: config/postprocessor/machine_default.yaml)",
    )
    sg.add_argument("--rot0y", type=float, default=0.0, help="Bed Y rotation (deg)")
    sg.add_argument("--rot0z", type=float, default=0.0, help="Bed Z rotation (deg)")
    sg.add_argument(
        "--layers",
        default="mesh,landmarks,origin,tip,arm",
        help="Comma-separated: mesh,landmarks,origin,cnc,tip,arm,programmed",
    )
    sg.add_argument(
        "--animate",
        action="store_true",
        help="p key advances one G-code step",
    )
    sg.add_argument(
        "--verbose",
        action="store_true",
        help="Print rigid FK checks, registration fit, and tip vs mesh metrics",
    )
    sg.set_defaults(func=cmd_simulate_gcode)
