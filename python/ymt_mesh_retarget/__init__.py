from .ui import show_ui
from .logic import retarget


def reload():
    """Reload module files"""
    import sys
    import importlib
    importlib.reload(sys.modules[__name__])

    for file in "base", "joint", "mesh", "transform":
        try:
            importlib.reload(sys.modules[f"{__name__}.objects.{file}"])
        except KeyError:
            pass

    for file in "alignment", "core", "geometry", "main", "mapping", "raycast", "weights":
        try:
            importlib.reload(sys.modules[f"{__name__}.registration.{file}"])
        except KeyError:
            pass

    for file in "ui", "util", "inpaint", "cluster", "logic", "objects", "registration":
        try:
            importlib.reload(sys.modules[f"{__name__}.{file}"])
        except KeyError:
            pass

__all__ = ["show_ui", "reload", "retarget"]
