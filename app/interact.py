"""Accept and discard keys shared by every interactive window."""

from __future__ import annotations

ACCEPT_HELP = "Space / Enter / S = accept"
CANCEL_HELP = "Q / Esc = discard"
KEYS_HELP = f"{ACCEPT_HELP} · {CANCEL_HELP}"

_ACCEPT_KEYS = ("space", "Return", "s", "S")
_CANCEL_KEYS = ("q", "Q", "Escape")

KEY_SPACE = 32
KEY_ENTER = 257
KEY_ESCAPE = 256


def bind_accept_cancel(plotter, on_accept, on_cancel) -> None:
    """Space, Enter, and S accept. Q and Esc discard."""
    for key in _ACCEPT_KEYS:
        plotter.add_key_event(key, on_accept)
    for key in _CANCEL_KEYS:
        plotter.add_key_event(key, on_cancel)


def key_accepts(key: int) -> bool:
    return key in (KEY_ENTER, KEY_SPACE, ord("s"), ord("S"))


def key_discards(key: int) -> bool:
    return key in (KEY_ESCAPE, ord("q"), ord("Q"))
