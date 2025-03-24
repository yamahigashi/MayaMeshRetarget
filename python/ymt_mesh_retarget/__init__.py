import contextlib

from .logger import logger, set_log_level
from .logic import retarget
from .ui import show_ui


def reload() -> None:
    """Reload module files."""
    import importlib
    import sys

    importlib.reload(sys.modules[__name__])

    for file in "base", "joint", "mesh", "transform":
        with contextlib.suppress(KeyError):
            importlib.reload(sys.modules[f"{__name__}.objects.{file}"])

    for file in "alignment", "core", "geometry", "main", "mapping", "raycast", "weights":
        with contextlib.suppress(KeyError):
            importlib.reload(sys.modules[f"{__name__}.registration.{file}"])

    for file in "ui", "util", "inpaint", "cluster", "logic", "objects", "registration", "logger":
        with contextlib.suppress(KeyError):
            importlib.reload(sys.modules[f"{__name__}.{file}"])


__all__ = ["logger", "reload", "retarget", "set_log_level", "show_ui"]
