"""Interactive physical landmark capture from live CNC work pose."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from app import paths
from app.postprocess.cnc_work_pose import WorkPose, WorkPoseUdpClient
from app.postprocess.print_config import save_physical_landmarks

LANDMARK_ORDER = (
    "landmark_central",
    "landmark_left",
    "landmark_back",
)


def pm_from_raw_touches(raw_xyz: np.ndarray) -> np.ndarray:
    """
    Convert three absolute work XYZ touches into pm measurement frame.

    ``pm[0]`` is always the origin; left/back are relative to central.
    """
    pts = np.asarray(raw_xyz, dtype=float)
    if pts.shape != (3, 3):
        raise ValueError(f"raw touches must be 3x3, got {pts.shape}")
    origin = pts[0].copy()
    out = np.zeros((3, 3), dtype=float)
    out[1] = pts[1] - origin
    out[2] = pts[2] - origin
    return out


def _read_key() -> str | None:
    """Non-blocking single key (Windows msvcrt) or None."""
    try:
        import msvcrt
    except ImportError:
        return None
    if not msvcrt.kbhit():
        return None
    ch = msvcrt.getwch()
    if ch in ("\x00", "\xe0"):
        msvcrt.getwch()
        return None
    return ch


def _format_pose(pose: WorkPose | None, status: str) -> str:
    if pose is None:
        return f"{status} | XYZ ---  B/C ---"
    return (
        f"{status} | "
        f"X={pose.x:8.3f} Y={pose.y:8.3f} Z={pose.z:8.3f}  "
        f"B={pose.b_deg:7.2f} C={pose.c_deg:7.2f}"
    )


def record_physical_landmarks(
    subject_id: int | str,
    *,
    bind_ip: str = "0.0.0.0",
    port: int = 62100,
    stale_sec: float = 0.5,
    force: bool = False,
    output: Path | str | None = None,
) -> Path:
    """
    Jog tip to each calibration landmark; press Enter/Space to capture CNC work XYZ.

    Central becomes pm origin ``[0,0,0]``; left/back are stored relative to central.
    """
    out = Path(output) if output is not None else paths.postprocessor_subject_pm(subject_id)
    if not out.is_absolute():
        out = paths.REPO_ROOT / out
    if out.exists() and not force:
        raise FileExistsError(
            f"Print config already exists: {out} (use --force to overwrite)"
        )

    raw = [None, None, None]  # type: list[np.ndarray | None]
    poses_at_capture: list[WorkPose | None] = [None, None, None]
    idx = 0

    print(
        f"Record physical landmarks for subject {subject_id}\n"
        f"  Listening for Mach4 work pose on UDP {bind_ip}:{port}\n"
        f"  Order: {', '.join(LANDMARK_ORDER)}\n"
        f"  Keys: Enter/Space=capture (or save when all 3 done)  "
        f"n=next  p=prev  1/2/3=jump  s=save  q=quit\n"
        f"  Tip: touch each landmark with the end-effector, then capture.\n"
        f"  Central is stored as [0,0,0]; left/back are relative to that touch.\n"
    )
    if sys.platform != "win32":
        print(
            "  Note: non-Windows — type a key then Enter "
            "(e.g. Enter to capture, s Enter to save).\n"
        )

    def _try_save() -> Path | None:
        if any(p is None for p in raw):
            missing = [LANDMARK_ORDER[i] for i, p in enumerate(raw) if p is None]
            sys.stdout.write(f"\nStill missing: {', '.join(missing)}\n")
            return None
        pm = pm_from_raw_touches(np.vstack(raw))
        meta = {
            "raw_work_xyz_mm": [p.tolist() for p in raw],  # type: ignore[union-attr]
            "work_bc_deg": [
                {
                    "b": float(poses_at_capture[i].b_deg),  # type: ignore[union-attr]
                    "c": float(poses_at_capture[i].c_deg),  # type: ignore[union-attr]
                }
                for i in range(3)
            ],
            "udp_port": port,
        }
        save_physical_landmarks(out, pm, subject_id=subject_id, capture_meta=meta)
        sys.stdout.write(f"\nWrote {out}\n{pm}\n")
        return out

    with WorkPoseUdpClient(bind_ip=bind_ip, port=port, stale_sec=stale_sec) as client:
        last_line = ""
        while True:
            pose = client.latest()
            status = client.status_label()
            marks = []
            for i, name in enumerate(LANDMARK_ORDER):
                flag = "✓" if raw[i] is not None else "·"
                cursor = ">" if i == idx else " "
                marks.append(f"{cursor}{flag}{i + 1}:{name.replace('landmark_', '')}")
            line = (
                f"\r{'  '.join(marks)}  |  {_format_pose(pose, status)}    "
            )
            if line != last_line:
                sys.stdout.write(line)
                sys.stdout.flush()
                last_line = line

            key = _read_key()
            if key is None and sys.platform != "win32":
                # Line-oriented fallback: block briefly via select if available
                import select

                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if ready:
                    key = sys.stdin.readline().strip() or "\r"
                    if len(key) == 1:
                        pass
                    elif key.lower() in ("save", "s"):
                        key = "s"
                    elif key.lower() in ("quit", "q"):
                        key = "q"
                    elif key.lower() in ("next", "n"):
                        key = "n"
                    elif key.lower() in ("prev", "p"):
                        key = "p"
                    else:
                        key = "\r"
                else:
                    time.sleep(0.05)
                    continue
            elif key is None:
                time.sleep(0.05)
                continue

            key_l = key.lower()
            if key in ("\r", "\n", " "):
                if all(p is not None for p in raw):
                    saved = _try_save()
                    if saved is not None:
                        return saved
                    last_line = ""
                    continue
                if pose is None:
                    sys.stdout.write(
                        "\nNo live work pose — start Mach4 UDP publisher "
                        "(see config/postprocessor/README.md).\n"
                    )
                    last_line = ""
                    continue
                raw[idx] = np.asarray(pose.xyz, dtype=float)
                poses_at_capture[idx] = pose
                sys.stdout.write(
                    f"\nCaptured {LANDMARK_ORDER[idx]}: "
                    f"[{pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}] "
                    f"(B={pose.b_deg:.2f}, C={pose.c_deg:.2f})\n"
                )
                last_line = ""
                if idx < 2:
                    idx += 1
                else:
                    sys.stdout.write(
                        "All three captured — Space/Enter/S to save, Q to quit.\n"
                    )
            elif key_l == "n":
                idx = min(2, idx + 1)
            elif key_l == "p":
                idx = max(0, idx - 1)
            elif key in ("1", "2", "3"):
                idx = int(key) - 1
            elif key_l == "s":
                saved = _try_save()
                if saved is not None:
                    return saved
                last_line = ""
            elif key_l == "q" or key == "\x1b":
                sys.stdout.write("\nAborted — nothing written.\n")
                raise SystemExit(1)

    raise RuntimeError("unreachable")
