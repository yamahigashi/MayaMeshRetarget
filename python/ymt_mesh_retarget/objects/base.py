from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Union
from maya.api import OpenMaya as om

class RetargetableObject(ABC):
    """抽象基底クラス: リターゲット可能なオブジェクトの共通インターフェース"""

    name: str
    
    @abstractmethod
    def get_points(self, sampling_stride: int = 1) -> np.ndarray:
        """サンプリングされた点群を取得"""
        pass
    
    @abstractmethod
    def get_transforms(self) -> list[dict]:
        """変換情報（位置、回転、スケール）を取得"""
        pass
    
    @abstractmethod
    def duplicate(self, suffix: str="_retarget") -> "RetargetableObject":
        """オブジェクトを複製"""
        pass
    
    @abstractmethod
    def apply_transforms(self, transform_data: list[dict]) -> None:
        """変換情報を適用"""
        pass
    
    @abstractmethod
    def calculate_threshold_distance(self, coefficient: float) -> float:
        """閾値距離を計算"""
        pass
    
    @abstractmethod
    def get_children(self, type_filter: Optional[str] = None) -> list["RetargetableObject"]:
        """子オブジェクトを取得（階層処理用）"""
        pass
    
    @staticmethod
    @abstractmethod
    def create_from_path(path: Union[str, om.MDagPath]) -> "RetargetableObject":
        """パスからインスタンスを作成するファクトリメソッド"""
        pass
