import typing
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
from maya import cmds
from maya.api import OpenMaya as om

from ..cluster import cluster_vertices
from ..inpaint import inpaint_distance
from ..logger import logger
from ..util import (
    convert_points_to_numpy,
    get_mesh_dag,
    get_mesh_fn,
    get_short_name,
    get_skin_cluster,
    set_points,
    timeit,
)
from .base import RetargetableObject


if typing.TYPE_CHECKING:
    from ..types import (
        IndexArray,
        VertexArray,
    )


class MeshObject(RetargetableObject):
    """Implementation of RetargetableObject for mesh objects."""

    def __init__(self, mesh_path: Union[str, om.MDagPath]) -> None:
        """Initialize."""
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

        # Caches for vertex data
        self._normals_cache: dict[int, np.ndarray] = {}
        self._laplacians_cache: dict[int, np.ndarray] = {}
        self._weights_cache: dict[int, np.ndarray] = {}

        self._is_normals_cached = False
        self._is_laplacians_cached = False
        self._is_weights_cached = False

        self.precompute_vertex_normals()
        self.precompute_laplacians()
        self.precompute_weight_vectors()

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

    def get_parent_name(self) -> Optional[str]:
        """親オブジェクトの名前を取得."""
        if cmds.objectType(self.name) == "mesh":
            parents = cmds.listRelatives(self.name, parent=True, fullPath=True)
            parents = cmds.listRelatives(parents[0], parent=True, fullPath=True)
        else:
            parents = cmds.listRelatives(self.name, parent=True, fullPath=True)
        if parents:
            return parents[0]

        return None

    def duplicate(self, suffix: str = "_retarget") -> "MeshObject":
        """メッシュを複製."""

        mesh_name = self.dag_path.fullPathName().split("|")[-1]
        new_name = f"{mesh_name}{suffix}"

        trans = cmds.listRelatives(self.name, parent=True, fullPath=True)[0]
        duplicate = cmds.duplicate(trans, name=new_name)[0]
        duplicate = self.parent_retarget(duplicate)

        short_name = get_short_name(trans)
        duplicate = cmds.rename(duplicate, f"{short_name}{suffix}")
        duplicate = cmds.listRelatives(duplicate, shapes=True, fullPath=True)[0]

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

    def get_children(self, type_filter: Optional[str] = None) -> list["RetargetableObject"]:  # noqa: ARG002
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

    def inpaint_distance_matrix(
        self,
        source_path: str,
        distances: np.ndarray,
        labels: np.ndarray,
        threshold_coeff: float,
        angle: float,
    ) -> np.ndarray:
        """距離行列のインペイント処理."""
        return inpaint_distance(source_path, [self.dag_path], distances, labels, threshold_coeff, angle)

    def get_vertex_normal(self, vertex_index: int) -> np.ndarray:
        """Get the normal vector for a specific vertex.

        Args:
            vertex_index: The index of the vertex

        Returns:
            np.ndarray: Normalized normal vector
        """
        # Return from cache if available
        if self._is_normals_cached and vertex_index in self._normals_cache:
            return self._normals_cache[vertex_index]

        # Get normal from Maya API
        normal_vector = self.mesh_fn.getVertexNormal(vertex_index, True, om.MSpace.kWorld)

        # Convert to numpy array and normalize
        normal = np.array([normal_vector.x, normal_vector.y, normal_vector.z], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm

        # Cache the normal
        self._normals_cache[vertex_index] = normal

        return normal

    @timeit
    def precompute_vertex_normals(self) -> None:
        """Precompute and cache all vertex normals."""
        if self._is_normals_cached:
            return

        num_vertices = self.mesh_fn.numVertices
        logger.info(f"Precomputing normals for {num_vertices} vertices...")

        normal_vectors = self.mesh_fn.getVertexNormals(True, om.MSpace.kWorld)
        if normal_vectors:
            for i, vec in enumerate(normal_vectors):
                normal = np.array([vec.x, vec.y, vec.z], dtype=np.float32)
                norm = np.linalg.norm(normal)
                if norm > 1e-6:
                    normal = normal / norm

                # Cache the normal
                self._normals_cache[i] = normal

        self._is_normals_cached = True
        logger.info(f"Finished precomputing {num_vertices} vertex normals.")

    def compute_laplacian_for_vertex(self, vertex_index: int) -> np.ndarray:
        """Compute Laplacian coordinates for a vertex.

        Uses cotangent weighting to compute the Laplacian coordinates.

        Args:
            vertex_index: The index of the vertex

        Returns:
            np.ndarray: Laplacian coordinates
        """
        # Return from cache if available
        if self._is_laplacians_cached and vertex_index in self._laplacians_cache:
            return self._laplacians_cache[vertex_index]

        raise NotImplementedError("Laplacian computation for individual vertices is not yet implemented.")

    @timeit
    def precompute_laplacians(self) -> None:
        """Precompute and cache Laplacian coordinates for all vertices."""
        if self._is_laplacians_cached:
            return

        logger.info("Computing Laplacian & mass matrix via compute_laplacian_and_mass_matrix...")
        L_csr, M_csr = compute_laplacian_and_mass_matrix(self.mesh_fn)  # (N×N), (N×N)  # noqa: N806

        # 頂点座標を (N, 3) の numpy配列で取得
        points = self.get_points()  # 例: array([[x0,y0,z0],[x1,y1,z1],...]], shape=(N,3))
        n_vertices = len(points)
        laplacian_coords = np.zeros_like(points)  # shape=(N,3)

        # 例：ラプラシアン生値: L * p
        # 列ごとに掛け算する (scipyの行列は 2次元配列対応だが、行列×行列でもOK)
        # 単純化のため列ごとに行う例：
        for dim in range(3):
            laplacian_coords[:, dim] = L_csr.dot(points[:, dim])

        # もし質量行列 M も使う場合 (Mが対角行列なので各要素を割り算するイメージ):
        # for dim in range(3):
        #     Lp_dim = L_csr.dot(points[:, dim])   # shape=(N,)
        #     # areaが0だと割り算できないので注意
        #     # M_csr.diagonal() が各頂点の面積データになる
        #     # 例: laplacian_coords[:, dim] = Lp_dim / np.maximum(M_csr.diagonal(), 1e-8)

        # 辞書キャッシュに格納する
        for i in range(n_vertices):
            self._laplacians_cache[i] = laplacian_coords[i]

        self._is_laplacians_cached = True
        logger.info(f"Finished precomputing Laplacian coords for {n_vertices} vertices.")

    def get_vertex_weight_vector(self, vertex_index: int, influence_names: Optional[list[str]] = None) -> np.ndarray:
        """Get the skin weight vector for a vertex.

        Args:
            vertex_index: The index of the vertex
            influence_names: Optional list of influence names to filter by

        Returns:
            np.ndarray: Weight vector, normalized if any weights exist
        """
        # Return from cache if available
        if self._is_weights_cached and vertex_index in self._weights_cache:
            return self._weights_cache[vertex_index]

        # Get the skinCluster from the mesh
        mesh_path = self.dag_path.fullPathName()
        skin_clusters = cmds.listConnections(mesh_path, type="skinCluster")

        if not skin_clusters:
            # No skin cluster found, return empty array
            logger.warning(f"No skinCluster found for {mesh_path}")
            weights = np.array([], dtype=np.float64)
            self._weights_cache[vertex_index] = weights
            return weights

        skin_cluster = skin_clusters[0]

        # Get all influences in the skinCluster
        all_influences = cmds.skinCluster(skin_cluster, query=True, influence=True)
        if not all_influences:
            weights = np.array([], dtype=np.float64)
            self._weights_cache[vertex_index] = weights
            return weights

        # If influence_names is provided, filter the influences
        influences_to_use = influence_names if influence_names else all_influences

        # Create a weight vector with an entry for each influence
        weights = np.zeros(len(influences_to_use), dtype=np.float64)

        # Get the weight values for each influence
        for i, influence in enumerate(influences_to_use):
            if influence in all_influences:
                weight_value = cmds.skinPercent(
                    skin_cluster, f"{mesh_path}.vtx[{vertex_index}]",
                    transform=influence, query=True,
                )
                weights[i] = weight_value

        # Normalize the weight vector if it's not all zeros
        weights_sum = np.sum(weights)
        if weights_sum > 1e-6:
            weights = weights / weights_sum

        # Cache the weight vector
        self._weights_cache[vertex_index] = weights

        return weights

    @timeit
    def precompute_weight_vectors(self, influence_names: Optional[list[str]] = None) -> None:
        """Precompute and cache weight vectors for all vertices.

        This is a high-performance approach that uses MFnSkinCluster.getWeights(...)
        to avoid calling cmds.skinPercent per vertex.

        Args:
            influence_names: Optional list of influence names to filter by

        Returns:
            None
        """
        if self._is_weights_cached:
            return

        num_vertices = self.mesh_fn.numVertices
        logger.info(f"Precomputing weight vectors for {num_vertices} vertices...")

        mesh_path = self.dag_path.fullPathName()
        skin_clusters = cmds.listConnections(mesh_path, type="skinCluster")

        if not skin_clusters:
            logger.warning(f"No skinCluster found for {mesh_path}. Skipping weight precomputation.")
            self._is_weights_cached = True
            return

        skin_cluster = skin_clusters[0]

        # 1) 取得するInfluenceを決定
        all_influences = cmds.skinCluster(skin_cluster, query=True, influence=True)
        if not all_influences:
            logger.warning("No influences found in skinCluster.")
            self._is_weights_cached = True
            return

        if influence_names is None:
            influences_to_use = all_influences
        else:
            # ユーザが指定した influence_names に限定
            influences_to_use = [inf for inf in all_influences if inf in influence_names]
            if not influences_to_use:
                logger.warning("No matched influences in the given influence_names. All weights empty.")
                for i in range(num_vertices):
                    self._weights_cache[i] = np.zeros(0, dtype=np.float64)
                self._is_weights_cached = True
                return

        # 2) Maya API から全頂点のウェイトを一括取得 (MFnSkinCluster)
        fn_skin = get_skin_cluster(self.dag_path)  # 例外は呼び出し側でキャッチ済み

        # 全influenceのMDagPathを取得
        influence_objects = fn_skin.influenceObjects()  # om.MDagPathArray
        num_influences = len(influence_objects)

        # 全頂点の component を用意
        vert_indices = om.MIntArray(list(range(num_vertices)))
        vert_comp = om.MFnSingleIndexedComponent().create(om.MFn.kMeshVertComponent)
        om.MFnSingleIndexedComponent(vert_comp).addElements(vert_indices)

        # 全influence分の index array を作成
        influence_indices = om.MIntArray(list(range(num_influences)))

        # getWeightsでまとめて取得 (MFloatArrayで返る)
        raw_weights = fn_skin.getWeights(self.dag_path, vert_comp, influence_indices)

        # 3) N x num_influences の2次元配列にreshape
        weights_all = np.array(raw_weights, dtype=np.float64).reshape(num_vertices, num_influences)

        # 4) 必要な influence の列だけ抽出
        #    (influence_names が Noneなら全列使用)
        use_indices = []
        for inf_name in influences_to_use:
            idx = all_influences.index(inf_name)
            use_indices.append(idx)

        sub_weights = weights_all[:, use_indices]  # shape=(N, len(use_indices))

        # 5) 各頂点のウェイトを正規化 (合計1になるように)
        row_sums = sub_weights.sum(axis=1)
        row_sums[row_sums < 1e-8] = 1.0  # 0除算回避
        sub_weights /= row_sums[:, None]

        # 6) self._weights_cache に書き込み
        for i in range(num_vertices):
            self._weights_cache[i] = sub_weights[i]  # shape=(len(use_indices),)

        self._is_weights_cached = True
        logger.info(
            f"Finished precomputing {num_vertices} vertex weight vectors "
            f"({len(use_indices)} influences) via bulk getWeights."  # noqa: COM812
        )


def add_laplacian_entry_in_place(
        L: sp.lil_matrix,  # noqa: N803
        tri_positions: np.ndarray,
        tri_indices: np.ndarray,
) -> None:
    """Add laplacian entry in-place.

    Args:
        L: Laplacian matrix (**modified in-place**) (n_vertices, n_vertices)
        tri_positions: Triangle positions
        tri_indices: Triangle indices

    Returns:
        None
    """

    i1 = tri_indices[0]
    i2 = tri_indices[1]
    i3 = tri_indices[2]

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]

    # calculate cotangent
    cotan1 = compute_cotangent(v2, v1, v3)
    cotan2 = compute_cotangent(v1, v2, v3)
    cotan3 = compute_cotangent(v1, v3, v2)

    # update laplacian matrix
    L[i1, i2] += cotan1
    L[i2, i1] += cotan1
    L[i1, i1] -= cotan1
    L[i2, i2] -= cotan1

    L[i2, i3] += cotan2
    L[i3, i2] += cotan2
    L[i2, i2] -= cotan2
    L[i3, i3] -= cotan2

    L[i1, i3] += cotan3
    L[i3, i1] += cotan3
    L[i1, i1] -= cotan3
    L[i3, i3] -= cotan3


def add_area_in_place(
    areas: np.ndarray,
    tri_positions: np.ndarray,
    tri_indices: np.ndarray,
) -> None:
    """Add area in-place.

    Args:
        areas: Areas (**modified in-place**) (n_vertices,)
        tri_positions: Triangle positions
        tri_indices: Triangle indices

    Returns:
        None
    """

    v1 = tri_positions[0]
    v2 = tri_positions[1]
    v3 = tri_positions[2]
    area = 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))

    for idx in tri_indices:
        areas[idx] += area


def compute_laplacian_and_mass_matrix(mesh: om.MFnMesh) -> tuple[sp.csr_matrix, sp.dia_matrix]:
    """Compute laplacian matrix from mesh.

    treat area as mass matrix.
    """

    # initialize sparse laplacian matrix
    n_vertices = mesh.numVertices
    L = sp.lil_matrix((n_vertices, n_vertices))  # noqa: N806
    areas = np.zeros(n_vertices)

    # for each edge and face, calculate the laplacian entry and area
    face_iter = om.MItMeshPolygon(mesh.dagPath())
    while not face_iter.isDone():

        n_tri = face_iter.numTriangles()

        for j in range(n_tri):

            tri_positions, tri_indices = face_iter.getTriangle(j)
            add_laplacian_entry_in_place(L, tri_positions, tri_indices)
            add_area_in_place(areas, tri_positions, tri_indices)

        face_iter.next()

    L_csr = L.tocsr()  # noqa: N806
    M_csr = sp.diags(areas)  # noqa: N806

    return L_csr, M_csr


def compute_cotangent(v1: om.MPoint, v2: om.MPoint, v3: om.MPoint) -> float:
    """Compute cotangent from three points."""

    edeg1 = v2 - v1
    edeg2 = v3 - v1

    norm1 = edeg1 ^ edeg2

    area = norm1.length()
    cotan = edeg1 * edeg2 / area

    return cotan
