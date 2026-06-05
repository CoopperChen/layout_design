"""
Pipeline runtime: repo-root cwd + PYTHON package on sys.path.

Call setup_runtime() at the start of every CLI command and library entrypoint.
"""
from __future__ import annotations

import os
import sys

from app import paths


def setup_runtime() -> None:
    """Use layout_design/data/ with legacy PYTHON.tools imports."""
    os.chdir(paths.REPO_ROOT)
    app_dir = str(paths.APP_DIR)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)
    paths.ensure_data_tree()
