from scipy.spatial.transform import Rotation

import numpy as np
from maya import cmds

from ..util import get_dag_path
from .base import RetargetableObject


class TransformObject(RetargetableObject):
    """トランスフォームオブジェクト用の実装."""

    def __init__(self, transform_path: str) -> None:
        if isinstance(transform_path, str):
            self.dag_path = get_dag_path(transform_path)
            if not self.dag_path:
                raise ValueError(f"Invalid transform path: {transform_path}")
        else:
            self.dag_path = transform_path

        self.name = self.dag_path.fullPathName()

    def get_points(self, sampling_stride: int = 1) -> np.ndarray:
        """トランスフォームの位置を点として取得."""
        pos = cmds.xform(self.name, query=True, worldSpace=True, translation=True)
        return np.array([pos])

    def get_transforms(self) -> list[dict]:
        """トランスフォームの変換情報を取得."""
        # 位置を取得
        pos = cmds.xform(self.name, query=True, worldSpace=True, translation=True)

        # 回転をクォータニオンとして取得
        rot = cmds.xform(self.name, query=True, worldSpace=True, rotation=True)
        quat = Rotation.from_euler("xyz", rot, degrees=True).as_quat()

        # スケールを取得
        scale = cmds.getAttr(f"{self.name}.scale")[0]

        return [
            {
                "path": self.name,
                "position": np.array(pos),
                "rotation": quat,
                "scale": np.array(scale),
            },
        ]

    def duplicate(self, suffix: str = "_retarget") -> "TransformObject":
        """トランスフォームを複製."""
        duplicate = cmds.duplicate(self.name, parentOnly=True)[0]
        short_name = cmds.ls(duplicate, shortNames=True)[0]
        if not short_name.endswith(suffix):
            duplicate = cmds.rename(duplicate, f"{short_name}{suffix}")
        return self.__class__.create_from_path(duplicate)

    def apply_transforms(self, transform_data: list[dict]) -> None:
        """変換情報をトランスフォームに適用."""
        data = transform_data[0]["position"]  # matrix
        pos = data.flatten().tolist()[0]

        # 位置を適用
        cmds.xform(self.name, worldSpace=True, translation=pos)

        # # 回転を適用
        # euler = Rotation.from_quat(data["rotation"]).as_euler('xyz', degrees=True)
        # cmds.xform(self.name, worldSpace=True, rotation=euler)

        # スケールを適用
        # for i, axis in enumerate(['x', 'y', 'z']):
        #     cmds.setAttr(f"{self.name}.scale{axis.upper()}", data["scale"][i])

    def calculate_threshold_distance(self, coefficient: float) -> float:
        """しきい値距離の計算."""
        # トランスフォームの場合はバウンディングボックスを使用
        bbox = cmds.exactWorldBoundingBox(self.name)
        diag = np.sqrt(sum((np.array(bbox[3:6]) - np.array(bbox[0:3])) ** 2))
        return diag * coefficient

    def get_children(self, type_filter: str = None) -> list["TransformObject"]:
        """子オブジェクトを取得."""
        if type_filter:
            children = cmds.listRelatives(self.name, children=True, type=type_filter, fullPath=True) or []
        else:
            children = cmds.listRelatives(self.name, children=True, fullPath=True) or []
        return [self.__class__.create_from_path(child) for child in children]

    @staticmethod
    def create_from_path(path: str) -> "TransformObject":
        """パスからインスタンスを作成."""
        return TransformObject(path)
