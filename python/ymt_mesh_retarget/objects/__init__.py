# objects/__init__.py
from maya import cmds

from .mesh import MeshObject
from .joint import JointObject
from .transform import TransformObject

import typing
if typing.TYPE_CHECKING:
    from .base import RetargetableObject  # noqa: F401


def create_retargetable_object(path):
    # type: (str) -> RetargetableObject
    """パスから適切なRetargetableObjectインスタンスを作成"""
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
