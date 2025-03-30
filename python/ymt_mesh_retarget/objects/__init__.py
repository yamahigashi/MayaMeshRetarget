from typing import TYPE_CHECKING, Union, cast

import numpy as np
from maya import cmds

from .base import RetargetableObject
from .joint import JointObject
from .mesh import MeshObject
from .np_points import NumpyPointsObject
from .transform import TransformObject


if TYPE_CHECKING:
    import sys

    if sys.version_info >= (3, 10):
        from typing import TypeAlias
        RetargetableArg: TypeAlias = Union[str, np.ndarray, RetargetableObject]
    else:
        RetargetableArg = Union[str, np.ndarray, RetargetableObject]


def create_retargetable_object(path: RetargetableArg) -> RetargetableObject:
    """パスから適切なRetargetableObjectインスタンスを作成."""
    from .base import RetargetableObject

    # すでにRetargetableObjectインスタンスの場合はそのまま返す
    if isinstance(path, RetargetableObject):
        return path

    # NumPy配列の場合はNumpyPointsObjectを返す
    if isinstance(path, np.ndarray):
        return NumpyPointsObject(path)

    # 文字列の場合はオブジェクトタイプに基づいて適切なインスタンスを作成
    if not path or not isinstance(path, str) or not cmds.objExists(path):
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
