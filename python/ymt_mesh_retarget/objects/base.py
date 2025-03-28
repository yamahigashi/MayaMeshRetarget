from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from ..logger import logger


class RetargetableObject(ABC):
    """抽象基底クラス: リターゲット可能なオブジェクトの共通インターフェース."""

    name: str

    @abstractmethod
    def get_points(self, sampling_stride: int = 1) -> np.ndarray:
        """サンプリングされた点群を取得."""
        pass

    @abstractmethod
    def get_transforms(self) -> list[dict]:
        """変換情報（位置、回転、スケール）を取得."""
        pass

    @abstractmethod
    def duplicate(self, suffix: str = "_retarget") -> "RetargetableObject":
        """オブジェクトを複製."""
        pass

    @abstractmethod
    def apply_transforms(self, transform_data: list[dict]) -> None:
        """変換情報を適用."""
        pass

    @abstractmethod
    def calculate_threshold_distance(self, coefficient: float) -> float:
        """閾値距離を計算."""
        pass

    @abstractmethod
    def get_children(self, type_filter: Optional[str] = None) -> list["RetargetableObject"]:
        """子オブジェクトを取得（階層処理用）."""
        pass

    @staticmethod
    @abstractmethod
    def create_from_path(path: Union[str, om.MDagPath]) -> "RetargetableObject":
        """パスからインスタンスを作成するファクトリメソッド."""
        pass

    def get_parent_name(self) -> Optional[str]:
        """親オブジェクトの名前を取得."""
        parents = cmds.listRelatives(self.name, parent=True, fullPath=True)
        if parents:
            return parents[0]

        return None

    def parent_retarget(self, name: str, suffix: str = "_retarget") -> str:
        """リターゲット用の親オブジェクトを作成."""
        original_parent = self.get_parent_name()
        if not original_parent:
            logger.debug(f"Parent not found, {name}")
            return name

        parent_path_parts = original_parent.split("|")
        parent_path_parts = [f"{part}{suffix}" for part in parent_path_parts[1:]]
        retargeted_parent_name = "|" + "|".join(parent_path_parts)
        retargeted_parent = cmds.ls(retargeted_parent_name, long=True)
        if not retargeted_parent:
            logger.debug(f"Retargeted parent not found, {original_parent} -> {retargeted_parent_name}")

        parented_node = cmds.parent(name, retargeted_parent)

        return parented_node
