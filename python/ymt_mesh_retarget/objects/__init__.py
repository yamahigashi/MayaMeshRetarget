# objects/__init__.py
import sys
import typing

import numpy as np
from maya import cmds

from .joint import JointObject
from .mesh import MeshObject
from .np_points import NumpyPointsObject
from .transform import TransformObject


if typing.TYPE_CHECKING:
    from .base import RetargetableObject
    if sys.version_info >= (3, 10):
        # type alias for Literal
        from typing import TypeAlias
        RetargetableArg: TypeAlias = typing.Union[str, np.ndarray, "RetargetableObject"]
    else:
        RetargetableArg = typing.Union[str, np.ndarray, "RetargetableObject"]



def create_retargetable_object(path: "RetargetableArg") -> "RetargetableObject":
    """パスから適切なRetargetableObjectインスタンスを作成."""

    if isinstance(path, np.ndarray):
        return NumpyPointsObject(path)

    if not path or not cmds.objExists(path):
        raise ValueError(f"Invalid object path: {path}")

    node_type = cmds.nodeType(path)

    if node_type == "joint":
        return JointObject(path)

    elif node_type == "transform":
        # 子にshapeノードがあるかチェック
        shapes = cmds.listRelatives(path, shapes=True, fullPath=True)
        if shapes and cmds.nodeType(shapes[0]) == "mesh":
            return MeshObject(path)
        else:
            return TransformObject(path)

    else:
        # 汎用トランスフォームをデフォルトとして使用
        return TransformObject(path)
