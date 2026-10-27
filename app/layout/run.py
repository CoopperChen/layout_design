"""Place electrodes and write layout.json."""

from __future__ import annotations

from app import paths
from app.layout.session import run_layout_session
from app.subject import save_layout


def run_layout(subject_id: int) -> int:
    obj = paths.head_obj(subject_id)
    stl = paths.head_stl(subject_id)
    if not obj.is_file() or not stl.is_file():
        raise FileNotFoundError(
            f"Subject {subject_id} has no aligned head.\n"
            f"  Expected {obj}\n"
            f"  and {stl}\n"
            f"Run: python -m app mesh --subject {subject_id}"
        )
    layout = run_layout_session(obj)
    if layout is None:
        print("Layout discarded.")
        return 1
    out = save_layout(subject_id, layout)
    print(f"Wrote {out} ({len(layout.electrodes)} electrodes)")
    return 0
