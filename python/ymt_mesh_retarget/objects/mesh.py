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
    get_skin_weights,
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

    def precompute_data(self) -> None:
        """Precompute all vertex data for the mesh."""
        self.precompute_vertex_normals()
        self.precompute_laplacians()
        self.precompute_weight_vectors()

    def get_points(self, sampling_stride: int = 1) -> "VertexArray":
        """メッシュの頂点をnumpy配列として取得."""
        return convert_points_to_numpy(self.dag_path, sampling_stride)

    def get_smoothed_points(self, iterations: int = 10, smoothing_factor: float = 0.8) -> "VertexArray":
        """Return smoothed vertex positions using Laplacian smoothing."""
        return laplacian_smooth(self.mesh_fn, iterations, smoothing_factor)

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

    def duplicate(self, suffix: str = "_retarget", parent: Optional[str] = None) -> "MeshObject":
        """メッシュを複製."""

        mesh_name = self.dag_path.fullPathName().split("|")[-1]
        new_name = f"{mesh_name}{suffix}"

        trans = cmds.listRelatives(self.name, parent=True, fullPath=True)[0]
        duplicate = cmds.duplicate(trans, name=new_name)[0]
        duplicate = self.parent_retarget(duplicate, parent=parent)

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
        L_csr = compute_cotangent_laplacian(self.dag_path)  # (N, N) csr_matrix

        # 頂点座標を (N, 3) の numpy配列で取得
        points = self.get_points()  # 例: array([[x0,y0,z0],[x1,y1,z1],...]], shape=(N,3))
        n_vertices = len(points)
        laplacian_coords = np.zeros_like(points)  # shape=(N,3)

        # 例：ラプラシアン生値: L * p
        # 列ごとに掛け算する (scipyの行列は 2次元配列対応だが、行列×行列でもOK)
        # 単純化のため列ごとに行う例：
        for dim in range(3):
            laplacian_coords[:, dim] = L_csr.dot(points[:, dim])

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


def build_adjacency_list(dag_path: om.MDagPath) -> list[list[int]]:
    """MItMeshVertex を使用し、各頂点に隣接する頂点のリストを返す。"""
    mesh_vertex_iter = om.MItMeshVertex(dag_path)
    adjacency_list = [[] for _ in range(mesh_vertex_iter.count())]

    while not mesh_vertex_iter.isDone():
        idx = mesh_vertex_iter.index()
        connected_indices = mesh_vertex_iter.getConnectedVertices()

        for cidx in connected_indices:
            adjacency_list[idx].append(cidx)

        mesh_vertex_iter.next()

    return adjacency_list


def build_laplacian_matrix_from_adjacency(adjacency_list: list[list[int]]) -> np.ndarray:
    """隣接リストから一様重みのラプラシアン行列 L を作成する。

    L は (N x N) の NumPy array
    """
    N = len(adjacency_list)
    L = np.zeros((N, N), dtype=np.float64)

    for i, neighbors in enumerate(adjacency_list):
        deg_i = len(neighbors)
        if deg_i == 0:
            # 孤立点(通常メッシュでは考えにくい)を回避するため
            L[i, i] = 1.0
            continue

        # 対角成分
        L[i, i] = 1.0
        # 非対角成分
        for j in neighbors:
            L[i, j] = -1.0 / deg_i

    return L


def compute_uniform_weight_laplacian(mesh_fn: om.MFnMesh) -> sp.csr_array:
    """一様重みのラプラシアン行列を計算する。"""
    adjacency_list = build_adjacency_list(mesh_fn.dagPath())
    L = build_laplacian_matrix_from_adjacency(adjacency_list)

    return sp.csr_matrix(L).tocsr()


def compute_cotangent_laplacian(dag_path: om.MDagPath) -> sp.csr_array:
    """Compute cotangent Laplacian matrix for the given mesh.

    Args:
        dag_path (om.MDagPath): DAG path to the mesh

    Returns:
        sp.csr_array: Computed cotangent Laplacian matrix
    """

    # メッシュ関数セットを作成
    mesh_fn = om.MFnMesh(dag_path)

    # 頂点数と座標の取得
    num_vertices = mesh_fn.numVertices
    points = mesh_fn.getPoints(om.MSpace.kWorld)  # ワールド座標 (または kObject)

    # --------------------------------------------------
    #  辺ごとの重みを保持するための辞書を用意
    #  key: (minIndex, maxIndex) のタプル, value: 重み (cotangent の和)
    # --------------------------------------------------
    edgeWeights = {}

    def sorted_edge_key(i: int, j: int) -> tuple[int, int]:
        """Sort two indices and return as a tuple."""
        return (i, j) if i < j else (j, i)

    # --------------------------------------------------
    #  フェイスごとに三角分割してコタンジェント重みを計算
    # --------------------------------------------------
    # getTriangles() を使うと、三角分割された頂点インデックスが取得できる。
    # 戻り値: (triangleCounts, triangleVertices)
    #  - triangleCounts は各フェイスがいくつの三角形に分割されたかを示すリスト
    #  - triangleVertices は分割された三角形の頂点インデックスを並べた1次元リスト
    triangle_counts, triangle_vertices = mesh_fn.getTriangles()

    # triangle_vertices はフェイス順に三角形が並んでおり、
    # 1つの三角形につき3頂点(インデックス)が続く。
    triIndex = 0  # triangle_vertices を走査するためのカウンタ

    # 各フェイスに対応する三角形数に従って取り出し
    for _face_id, tri_count in enumerate(triangle_counts):
        for _ in range(tri_count):
            # 三角形の頂点インデックスを取得
            i0 = triangle_vertices[triIndex]
            i1 = triangle_vertices[triIndex + 1]
            i2 = triangle_vertices[triIndex + 2]
            triIndex += 3

            # 各頂点のワールド座標を numpy array に変換
            p0 = np.array([points[i0].x, points[i0].y, points[i0].z], dtype=np.float64)
            p1 = np.array([points[i1].x, points[i1].y, points[i1].z], dtype=np.float64)
            p2 = np.array([points[i2].x, points[i2].y, points[i2].z], dtype=np.float64)

            # 三角形の辺と向かい合う角度のコタンジェントを求める
            # cot(α) = (b^2 + c^2 - a^2) / (4 * 面積) などの公式を利用
            # ただし直接内積・外積を使って角度αを求め、cot(α) = cos(α)/sin(α) としてもよい

            # 三辺ベクトル
            v0 = p1 - p0
            v1 = p2 - p1
            v2 = p0 - p2

            # 各辺の長さ
            l0 = np.linalg.norm(v0)  # 辺 (p0, p1)
            l1 = np.linalg.norm(v1)  # 辺 (p1, p2)
            l2 = np.linalg.norm(v2)  # 辺 (p2, p0)

            # 三角形の面積 (2D の外積の大きさ / 2)
            # 3Dベクトルの場合でも、(v0 × (p2 - p0)) の大きさ/2 などで算出できる
            # ここでは v0 × v(三番目) の絶対値 / 2
            crossVec = np.cross(v0, p2 - p0)
            area = np.linalg.norm(crossVec) * 0.5
            if area < 1e-12:
                # 面積が極端に小さい場合の対策としてスキップや continue する処理を加味しても良い
                continue

            # 辺 (p1, p2) に対向する角度は頂点 p0 における角度
            # cot(α0)
            # cosAlpha0 = np.dot(v0, p2 - p0) / (l0 * np.linalg.norm(p2 - p0))
            # sin(α0) は外積からも求められるが、面積を使う方が数値的に安定しやすい
            # 三角形面積 = 0.5 * l0 * l(p2-p0) * sin(α0) なので
            # sinAlpha0 = (2.0 * area) / (l0 * np.linalg.norm(p2 - p0))
            # 安全に arccos, arcsin してから cot(α) = cos(α)/sin(α) にしても良いが、
            # 面積から cot(α) を求める直接式:
            #   cot(α) = (l1^2 + l2^2 - l0^2) / (4 * area)
            # を利用することが多いです。
            cotAlpha0 = (l1**2 + l2**2 - l0**2) / (4.0 * area)
            cotAlpha1 = (l2**2 + l0**2 - l1**2) / (4.0 * area)
            cotAlpha2 = (l0**2 + l1**2 - l2**2) / (4.0 * area)

            # ----------------------------------------------------
            #  コタンジェント重みは、辺 (i, j) に対して
            #    w_ij = ( cot(α) + cot(β) ) / 2
            #  として使われることが多い (2つの隣接三角形の角度α, β の和)
            #
            # しかし、ここでは各三角形について局所的に以下を加算:
            #    w(i0, i1) += cotAlpha2
            #    w(i1, i2) += cotAlpha0
            #    w(i2, i0) += cotAlpha1
            #
            # というように、三角形の対向角に対応するコタンジェントを各辺に対して加算。
            # 後で隣接三角形と合算されることで、結果的に (cot(α) + cot(β)) が入るイメージ。
            # ----------------------------------------------------
            # 三角形でのエッジ (i0, i1) に対する重み
            edgeKey01 = sorted_edge_key(i0, i1)
            edgeWeights[edgeKey01] = edgeWeights.get(edgeKey01, 0.0) + cotAlpha2

            # エッジ (i1, i2)
            edgeKey12 = sorted_edge_key(i1, i2)
            edgeWeights[edgeKey12] = edgeWeights.get(edgeKey12, 0.0) + cotAlpha0

            # エッジ (i2, i0)
            edgeKey20 = sorted_edge_key(i2, i0)
            edgeWeights[edgeKey20] = edgeWeights.get(edgeKey20, 0.0) + cotAlpha1

    # --------------------------------------------------
    #  疎行列用のデータ格納リスト (row, col, data)
    # --------------------------------------------------
    rowIndices = []
    colIndices = []
    values = []

    # 対角成分を求めるために各頂点 i ごとの重み合計を一時的に保持
    diagVal = np.zeros(num_vertices, dtype=np.float64)

    # エッジ重みを L 行列に反映
    # L[i,j] = - w_ij,  L[i,i] = Σ_j w_ij
    for (i, j), w in edgeWeights.items():
        # i と j が接続されている場合、L[i,j] と L[j,i] に -w を入れる
        # ただし最終的には対角成分に w を加算する
        rowIndices.append(i)
        colIndices.append(j)
        values.append(-w)

        rowIndices.append(j)
        colIndices.append(i)
        values.append(-w)

        # 対角成分に加算
        diagVal[i] += w
        diagVal[j] += w

    # 対角成分を追加
    for i in range(num_vertices):
        rowIndices.append(i)
        colIndices.append(i)
        values.append(diagVal[i])

    # scipy.sparse で (num_vertices x num_vertices) の疎行列を生成
    L = sp.coo_matrix((values, (rowIndices, colIndices)), shape=(num_vertices, num_vertices))

    # 必要に応じて形式を変換 (例: csr_matrix)
    L_csr = L.tocsr()

    # normalize by max value
    max_val = np.abs(L_csr.data).max()
    L_csr.data /= max_val

    return L_csr


@timeit
def laplacian_smooth(
    mesh_fn: om.MFnMesh,
    iterations: int = 10,
    smoothing_factor: float = 0.5,
    keep_border: bool = True,
    method: str = "uniform",
) -> "VertexArray":
    """Apply Laplacian smoothing (cotangent-based) to the mesh's vertex positions.

    This function:
      - Computes (if not precomputed) the Laplacian (L) and mass (M) matrices.
      - Iteratively updates the vertex positions as:
            X_{t+1} = X_t - α * M⁻¹ L X_t
        where α = smoothing_factor.
      - Optionally keeps boundary vertices fixed.

    TODO: Implement stride parameter for faster processing.

    Args:
        mesh_fn (om.MFnMesh):
            Maya MFnMesh object for the mesh to smooth.
        iterations (int, optional):
            Number of smoothing iterations. Defaults to 10.
        smoothing_factor (float, optional):
            Blend factor (α) used in the update step. Typically in the range (0, 1).
        keep_border (bool, optional):
            If True, boundary vertices remain fixed in place. Defaults to True.
        method (str, optional):
            Smoothing method to use. Defaults to "uniform". Options: "uniform", "cotangent".

    Returns:
        np.ndarray:
            A (num_vertices, 3) array of the final smoothed positions.
    """
    num_vertices = mesh_fn.numVertices
    if num_vertices == 0:
        logger.warning("Mesh has no vertices. Returning empty array.")
        return np.zeros((0, 3), dtype=np.float64)

    # 1) Get the current world-space positions of the mesh
    mesh_dag = mesh_fn.dagPath()
    positions = convert_points_to_numpy(mesh_dag)  # shape: (N, 3)

    # 2) Compute the Laplacian (L) and mass (M) matrices
    logger.info("Computing cotangent-based Laplacian for smoothing...")
    if method == "uniform":
        L_csr = compute_uniform_weight_laplacian(mesh_fn)
    elif method == "cotangent":
        L_csr = compute_cotangent_laplacian(mesh_dag)
    else:
        raise ValueError(f"Invalid smoothing method: {method}")

    # 3) Identify border (boundary) vertices if keep_border = True
    border_mask = np.zeros(num_vertices, dtype=bool)
    if keep_border:
        vert_iter = om.MItMeshVertex(mesh_fn.object())
        while not vert_iter.isDone():
            idx = vert_iter.index()
            if vert_iter.onBoundary():
                border_mask[idx] = True
            vert_iter.next()

    # 4) Iteratively apply Laplacian smoothing
    for _ in range(iterations):
        # (N, 3) result of L * X
        LX = L_csr.dot(positions)

        # Perform the update
        new_positions = positions - smoothing_factor * LX

        # If keep_border, restore border vertices to original positions
        if keep_border:
            new_positions[border_mask] = positions[border_mask]

        positions = new_positions

    # check for NaNs or infs
    if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
        logger.warning("NaN or inf values found in smoothed positions. Reverting to original positions.")
        positions = convert_points_to_numpy(mesh_dag)

    return positions


def create_smoothed_mesh(
    mesh_path: Union[str, list[str]],
    iterations: int = 10,
    smoothing_factor: float = 0.5,
    method: str = "uniform",
    parent: Optional[str] = None,
) -> Union[MeshObject, list[MeshObject]]:
    """Create a smoothed mesh object from the input mesh.

    Args:
        mesh_path (str):
            Full path to the input mesh.
        iterations (int, optional):
            Number of smoothing iterations. Defaults to 10.
        smoothing_factor (float, optional):
            Blend factor (α) used in the update step. Typically in the range (0, 1).
        method (str, optional):
            Smoothing method to use. Defaults to "uniform". Options: "uniform", "cotangent".
        parent (str, optional):
            Parent object name for the new mesh. Defaults to None.

    Returns:
        MeshObject:
            A new MeshObject instance representing the smoothed mesh.
    """
    if isinstance(mesh_path, list):
        result = []
        for path in mesh_path:
            smoothed_mesh = create_smoothed_mesh(path, iterations, smoothing_factor)
            result.append(smoothed_mesh)
        return result

    mesh_obj = MeshObject.create_from_path(mesh_path)
    smoothed_points = laplacian_smooth(
        mesh_obj.mesh_fn,
        iterations,
        smoothing_factor,
        keep_border=False,
        method=method,
    )
    smoothed_mesh = mesh_obj.duplicate(suffix="_smoothed", parent=parent)
    points = om.MPointArray()
    for p in smoothed_points:
        points.append(om.MPoint(p[0], p[1], p[2]))
    set_points(smoothed_mesh.name, points)
    return smoothed_mesh


def create_shrunk_mesh(
    mesh_path: Union[str, list[str]],
    factor: float = 0.5,
) -> Union[MeshObject, list[MeshObject]]:
    """Create a smoothed mesh object from the input mesh.

    Args:
        mesh_path (str):
            Full path to the input mesh.
        factor (float, optional):
            Shrink factor. Defaults to 0.5.

    Returns:
        MeshObject:
            A new MeshObject instance representing the smoothed mesh.
    """
    from ..registration.alignment import (
        get_joint_tree,
        shrink_mesh_toward_skeleton,
    )

    if isinstance(mesh_path, list):
        result = []
        for path in mesh_path:
            smoothed_mesh = create_shrunk_mesh(path, factor)
            result.append(smoothed_mesh)
        return result

    mesh_obj = MeshObject.create_from_path(mesh_path)
    vertices = convert_points_to_numpy(mesh_obj.dag_path)
    weights, joints = get_skin_weights(mesh_obj.dag_path)
    joint_paths, joint_group, bone_group = get_joint_tree(joints)

    shrinked_vertices = shrink_mesh_toward_skeleton(
        vertices=vertices,
        joint_group=joint_group,
        bone_group=bone_group,
        weights=weights,
        shrink_factor=factor,
    )

    shrinked_mesh = mesh_obj.duplicate(suffix="_shrinked")
    points = om.MPointArray()
    for p in shrinked_vertices:
        points.append(om.MPoint(p[0], p[1], p[2]))

    set_points(shrinked_mesh.name, points)

    return shrinked_mesh
