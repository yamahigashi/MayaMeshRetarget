import numpy as np
from maya import cmds
from maya.api import OpenMaya as om

from ..cluster import cluster_vertices
from ..inpaint import inpaint_distance
from ..util import convert_points_to_numpy, get_mesh_dag, get_mesh_fn, set_points
from .base import RetargetableObject


class MeshObject(RetargetableObject):
    """メッシュオブジェクト用の実装."""

    def __init__(self, mesh_path: str | om.MDagPath) -> None:
        if isinstance(mesh_path, str):
            self.dag_path = get_mesh_dag(mesh_path)  # type: ignore
            if not self.dag_path:
                raise ValueError(f"Invalid mesh path: {mesh_path}")
            if not isinstance(self.dag_path, om.MDagPath):
                raise ValueError(f"Invalid mesh path type: {type(self.dag_path)}")
        else:
            self.dag_path = mesh_path  # type: om.MDagPath

        self.mesh_fn = get_mesh_fn(self.dag_path)
        self.name = self.dag_path.fullPathName()

    def get_points(self, sampling_stride: int = 1) -> np.ndarray:
        """メッシュの頂点をnumpy配列として取得."""
        return convert_points_to_numpy(self.dag_path, sampling_stride)

    def get_transforms(self) -> list[dict]:
        """メッシュの頂点座標をtransforms配列として取得."""
        points = self.get_points()
        transforms = []
        for i, point in enumerate(points):
            transforms.append(
                {
                    "index": i,
                    "position": point,
                    "rotation": np.array([0, 0, 0, 1]),  # 単位クォータニオン
                    "scale": np.array([1, 1, 1]),
                },
            )
        return transforms

    def duplicate(self, suffix: str = "_retarget") -> "MeshObject":
        """メッシュを複製."""
        mesh_name = self.dag_path.fullPathName().split("|")[-1]
        new_name = f"{mesh_name}{suffix}"
        duplicate = cmds.duplicate(self.name, name=new_name)[0]
        return self.__class__.create_from_path(duplicate)

    def apply_transforms(self, transform_data: list[dict]) -> None:
        """変換情報をメッシュの頂点に適用."""
        points = om.MPointArray()
        for p in transform_data:
            index = p["index"]
            position = next(iter(p["position"].flatten().tolist()))
            point = om.MPoint(position[0], position[1], position[2])
            points.insert(point, index)
        set_points(self.name, points)

    def calculate_threshold_distance(self, coefficient: float) -> float:
        """しきい値距離の計算."""
        from ..util import calculate_threshold_distance

        return calculate_threshold_distance(self.dag_path, coefficient)

    def get_children(self, type_filter: str = None) -> list["RetargetableObject"]:
        """子オブジェクトを取得（メッシュの場合は空リスト）."""
        return []

    @staticmethod
    def create_from_path(path: str) -> "MeshObject":
        """パスからインスタンスを作成."""
        return MeshObject(path)

    # メッシュ固有のメソッド
    def cluster_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        """頂点クラスタリング."""
        return cluster_vertices([self.dag_path])

    def inpaint_distance_matrix(self, source_path: str, distances: np.ndarray, labels: np.ndarray, threshold_coeff: float, angle: float) -> np.ndarray:
        """距離行列のインペイント処理."""
        return inpaint_distance(source_path, [self.dag_path], distances, labels, threshold_coeff, angle)
