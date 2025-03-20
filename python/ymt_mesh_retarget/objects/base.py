from abc import ABC, abstractmethod
import numpy as np

class RetargetableObject(ABC):
    """抽象基底クラス: リターゲット可能なオブジェクトの共通インターフェース"""
    
    @abstractmethod
    def get_points(self, sampling_stride=1):
        """サンプリングされた点群を取得"""
        pass
    
    @abstractmethod
    def get_transforms(self):
        """変換情報（位置、回転、スケール）を取得"""
        pass
    
    @abstractmethod
    def duplicate(self, suffix="_retarget"):
        """オブジェクトを複製"""
        pass
    
    @abstractmethod
    def apply_transforms(self, transform_data):
        """変換情報を適用"""
        pass
    
    @abstractmethod
    def calculate_threshold_distance(self, coefficient):
        """しきい値距離を計算"""
        pass
    
    @abstractmethod
    def get_children(self, type_filter=None):
        """子オブジェクトを取得（階層処理用）"""
        pass
    
    @staticmethod
    @abstractmethod
    def create_from_path(path):
        """パスからインスタンスを作成するファクトリメソッド"""
        pass
