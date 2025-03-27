"""Mesh Retargeting Tool.

This module provides functionality for retargeting mesh deformation using radial basis functions (RBF)
and skin weight-based clustering. It is designed for use with Autodesk Maya and leverages both
OpenMaya API and SciPy for mesh manipulation and deformation.

Main Features:
- Mesh retargeting using RBF interpolation
- Various RBF kernels supported (e.g., Gaussian, Thin-Plate)
- Clustering of vertices by weight similarity and topology
- Efficient distance matrix computation and inpainting for missing values
- Supports both rigid and smooth transformations

Usage:
------
1. Define source and target meshes.
2. Select the appropriate RBF kernel and configure parameters.
3. Run the retargeting function to apply the transformation to a target mesh or set of meshes.

"""

import time
import typing
import warnings
from collections.abc import Sequence
from typing import Callable, Union

from scipy.spatial.distance import cdist

import numpy as np
from maya import cmds, mel
from maya.api import (
    OpenMaya as om,
)

# from scipy.spatial.transform import Rotation  # TODO: implement later
from sklearn.decomposition import PCA

from . import (
    # inpaint,
    # cluster,
    util,
)
from .logger import (
    logger,
)
from .objects import (
    MeshObject,
    create_retargetable_object,
)
from .objects.base import RetargetableObject


if typing.TYPE_CHECKING:

    Kernel = Callable[[np.ndarray, float], np.ndarray]


##############################################################################
class RBF:
    """Various RBF kernels for mesh deformation using radial basis functions.

    Each RBF kernel uses a distance matrix and a radius to control the smoothing
    and influence range of the deformation. The `radius` parameter acts as a
    scaling factor for the distances between vertices and influences how far
    the deformation extends across the mesh.

    - A smaller `radius` will result in sharper, more localized deformations.
    - A larger `radius` will produce smoother, more gradual deformations over
      a broader area.

    In practice, `radius` should be selected relative to the size of the target
    mesh. A common approach is to base the `radius` on the diagonal length of
    the target mesh's bounding box, scaled by a coefficient. This allows the
    deformation to adapt to the scale of the mesh.

    Example:
    --------
    If the bounding box of the target mesh has a diagonal length of 10 units,
    a coefficient of 0.1 would give a `radius` of 1 unit, which provides
    a reasonable balance between local and global deformations.

    """

    @classmethod
    def linear(cls, matrix: np.ndarray, radius: float) -> np.ndarray:  # noqa: ARG003
        """Linear RBF - No scaling applied to distances."""
        return matrix

    @classmethod
    def gaussian(cls, matrix: np.ndarray, radius: float) -> np.ndarray:
        # type: (np.ndarray, float) -> np.ndarray
        """Gaussian RBF - Distances are scaled using a Gaussian decay function.

        This kernel is effective for producing smooth, gradual deformations.
        A smaller `radius` leads to faster decay and more localized effects.
        """
        return np.exp(-(matrix**2) / (radius**2))

    @classmethod
    def thin_plate(cls, matrix: np.ndarray, radius: float) -> np.ndarray:
        # type: (np.ndarray, float) -> np.ndarray
        """Thin plate spline RBF - Produces smooth, surface-based deformations.

        The `radius` scales the distances and determines the curvature of
        the deformation. Smaller radii result in sharper curvatures.
        """
        result = (matrix / radius) ** 2
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = np.where(result > 0, np.log(result), result)
        return result

    @classmethod
    def multi_quadratic_biharmonic(cls, matrix: np.ndarray, radius: float) -> np.ndarray:
        # type: (np.ndarray, float) -> np.ndarray
        """Multi-quadratic biharmonic RBF - Blends distances with a quadratic term.

        The `radius` controls the extent of influence. Larger values result in
        broader, smoother deformations.
        """
        return np.sqrt((matrix**2) + (radius**2))

    @classmethod
    def inv_multi_quadratic_biharmonic(cls, matrix: np.ndarray, radius: float) -> np.ndarray:
        # type: (np.ndarray, float) -> np.ndarray
        """Inverse multi-quadratic biharmonic RBF - Inverse decay of distances.

        The `radius` determines the decay rate. Small radii yield sharp fall-offs,
        while larger radii provide broader influences.
        """
        return 1.0 / np.sqrt((matrix**2) + (radius**2))

    @classmethod
    def beckert_wendland_c2_basis(cls, matrix: np.ndarray, radius: float) -> np.ndarray:
        # type: (np.ndarray, float) -> np.ndarray
        """Beckert-Wendland C2 basis RBF - Compact support kernel.

        This RBF produces localized deformations within a certain radius.
        The `radius` parameter defines the extent of this support, with
        smaller values yielding more localized effects.
        """
        arg = matrix / radius
        first = np.where(1 - arg > 0, (1 - arg) ** 4, 0)
        second = (4 * arg) + 1
        return first * second


def __select_rbf_kernel(kernel_name: str) -> typing.Callable:
    # type: (str) -> typing.Callable
    """Select an RBF kernel by name."""
    kernels = {
        "linear": RBF.linear,
        "gaussian": RBF.gaussian,
        "thin_plate": RBF.thin_plate,
        "multi_quadratic": RBF.multi_quadratic_biharmonic,
        "inv_multi_quadratic": RBF.inv_multi_quadratic_biharmonic,
        "beckert_wendland": RBF.beckert_wendland_c2_basis,
    }

    if kernel_name not in kernels:
        raise ValueError(f"Invalid kernel name: {kernel_name}")

    return kernels[kernel_name]


def calculate_rbf_weight_matrix(
        source_points: np.ndarray,
        target_points: np.ndarray,
        kernel: typing.Callable,
        radius: float,
        epsilon: float = 1e-8,
) -> np.ndarray:
    # type: (np.ndarray, np.ndarray, Kernel, float, float) -> np.ndarray
    """Calculate the weight matrix for the RBF interpolation."""
    identity = np.ones((source_points.shape[0], 1))
    dist = get_distance_matrix(source_points, source_points, kernel, radius)
    dist += np.eye(dist.shape[0]) * epsilon

    dim = 3
    a = np.bmat(
        [
            [dist, identity, source_points],
            [identity.T, np.zeros((1, 1)), np.zeros((1, dim))],
            [source_points.T, np.zeros((dim, 1)), np.zeros((dim, dim))],
        ],
    )
    b = np.bmat([[target_points], [np.zeros((1, dim))], [np.zeros((dim, dim))]])

    try:
        return np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        rank_a = np.linalg.matrix_rank(a)
        rank_b = np.linalg.matrix_rank(b)
        mes = (
            "Singular matrix - check the source points for duplicates"
            f", the rank of A is {rank_a} and the rank of B is {rank_b}"
        )
        logger.error(mes)

    raise ValueError("Failed to solve the linear system")


def get_distance_matrix(v1: np.ndarray, v2: np.ndarray, kernel: typing.Callable, radius: float) -> np.ndarray:
    # type: (np.ndarray, np.ndarray, Kernel, float) -> np.ndarray
    """Calculate the distance matrix between two sets of points using the specified RBF."""
    matrix = cdist(v1, v2, "euclidean")
    if kernel != RBF.linear:
        matrix = kernel(matrix, radius)
    return matrix


##############################################################################
# Mesh retargeting main functions
##############################################################################
@util.timeit
def retarget(
    source: str,
    target: str,
    objects: Union[list[str], str],
    kernel: Union[Callable, str] = RBF.linear,
    radius_coefficient: float = 0.0005,
    angle: float = 180.0,
    sampling_stride: int = 1,
    apply_rigid_transform: bool = False,
    inpaint: bool = True,
    maintain_hierarchy: bool = True,
) -> Sequence[str]:
    """Run the mesh retarget.

    Args:
        source: Source mesh
        target: Modified source mesh
        objects: List of retargetable objects
        kernel: One of the RBF functions (default: RBF.linear)
        radius_coefficient: Smoothing parameter for the RBF (default: 0.0005)
        angle: Angle threshold for inpainting (default: 180.0)
        sampling_stride: Vertex stride to sample on the source mesh (default: 1)
        apply_rigid_transform: Whether to apply rigid transformation (default: False)
        inpaint: Whether to inpaint unconvinced vertices (default: True)
        maintain_hierarchy: Whether to maintain hierarchical structure (default: True)

    Returns:
        List of retargeted object names
    """
    source_obj = create_retargetable_object(source)
    target_obj = create_retargetable_object(target)

    # util.get_mesh_dag(source)
    # util.get_mesh_dag(target)

    retarget_objects = []
    if isinstance(objects, str):
        objects = [objects]

    for obj in objects:
        ret = create_retargetable_object(obj)
        if ret is not None:
            retarget_objects.append(ret)

    if isinstance(kernel, str):
        kernel = __select_rbf_kernel(kernel)

    start_time = time.time()

    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, beginProgress=True, status="Preparing Retargeting", maxValue=100)

    try:
        # リターゲット処理の実行
        result = __retarget(
            source_obj,
            target_obj,
            retarget_objects,
            kernel,
            radius_coefficient,
            angle,
            sampling_stride,
            apply_rigid_transform,
            inpaint,
            maintain_hierarchy,
        )
        return result
    finally:
        # 進捗バーの終了
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, endProgress=True)

        end_time = time.time()
        print(f"Retargeting completed in {end_time - start_time:.2f} seconds ({kernel.__name__})")

    # return [m.fullPathName() for m in deformed_meshes]


def __retarget(
    source_obj: RetargetableObject,
    target_obj: RetargetableObject,
    retarget_objects: list[RetargetableObject],
    kernel: Callable,
    radius_coefficient: float,
    angle: float,
    sampling_stride: int,
    apply_rigid_transform: bool,
    inpaint: bool,
    maintain_hierarchy: bool,
) -> Sequence[str]:
    """Run the mesh retarget implementation.

    Args:
        source_obj: Source object to retarget from
        target_obj: Target object with modified form
        retarget_objects: List of objects to retarget
        kernel: RBF kernel function to use
        radius_coefficient: Smoothing parameter for the RBF
        angle: Angle threshold for inpainting
        sampling_stride: Vertex sampling stride
        apply_rigid_transform: Whether to apply rigid transformations
        inpaint: Whether to inpaint distance matrix
        maintain_hierarchy: Whether to maintain hierarchical structure

    Returns:
        List of retargeted object names
    """
    # Extract points from source and target meshes
    source_points = source_obj.get_points(sampling_stride)
    target_points = target_obj.get_points(sampling_stride)

    if source_points.shape != target_points.shape:
        raise ValueError("Source and target meshes must have the same number of vertices")

    radius = source_obj.calculate_threshold_distance(radius_coefficient)
    weights = calculate_rbf_weight_matrix(source_points, target_points, kernel, radius)

    # オブジェクトごとに処理
    results = []
    for obj in retarget_objects:
        # オブジェクトの複製
        new_obj = obj.duplicate()
        print(f"Processing {new_obj.name}")

        # 変形処理（オブジェクトのタイプに応じた処理が内部で実行される）
        __apply_rbf_deformation(
            source_obj,
            target_obj,
            obj,
            new_obj,
            weights,
            kernel,
            radius_coefficient,
            angle,
            sampling_stride,
            apply_rigid_transform,
            inpaint,
            maintain_hierarchy,
        )

        results.append(new_obj.name)

    return results


@util.timeit
def __apply_rbf_deformation(
    source_obj: RetargetableObject,
    target_obj: RetargetableObject,
    original_obj: RetargetableObject,
    new_obj: RetargetableObject,
    weights: np.ndarray,
    kernel: typing.Callable,
    radius_coefficient: float,
    angle: float,
    sampling_stride: int = 1,
    apply_rigid_transform: bool = False,
    inpaint: bool = True,
    maintain_hierarchy: bool = True,
) -> None:
    """Apply RBF deformation to the target object.

    Args:
        source_obj: Source object to retarget from
        target_obj: Target object with modified form
        original_obj: Original object to deform
        new_obj: New object to apply the deformation to
        weights: RBF weights for the transformation
        kernel: RBF kernel function to use
        radius_coefficient: Smoothing parameter for the RBF
        angle: Angle threshold for inpainting
        sampling_stride: Vertex sampling stride
        apply_rigid_transform: Whether to apply rigid transformations
        inpaint: Whether to inpaint distance matrix
        maintain_hierarchy: Whether to maintain hierarchical structure

    Returns:
        None
    """
    # ソース点群とオブジェクトの点群を取得
    source_points = source_obj.get_points(sampling_stride)
    object_points = original_obj.get_points()

    # 半径を計算
    radius = source_obj.calculate_threshold_distance(radius_coefficient)

    # ソースとオブジェクト間の距離行列を計算
    distances = get_distance_matrix(object_points, source_points, kernel, radius)

    # オブジェクトの完全な変換情報を取得
    transforms = original_obj.get_transforms()

    # メッシュ特有の処理（クラスタリングとインペイント）
    labels = None
    if apply_rigid_transform and isinstance(original_obj, MeshObject):
        # メッシュの場合はクラスタリングを実行
        labels = original_obj.cluster_vertices()

        if inpaint:
            # 距離行列のインペイント
            distances = original_obj.inpaint_distance_matrix(
                source_obj.dag_path,
                distances,
                labels,
                radius_coefficient,
                angle,
            )

    # RBF補間を使用して変換後のポイントを計算
    identity = np.ones((object_points.shape[0], 1))
    h_combined = np.bmat([[distances, identity, object_points]])
    deformed_points = np.dot(h_combined, weights)

    # 変換情報構造体を更新
    for i, transform in enumerate(transforms):
        # インデックスが範囲内にあることを確認
        if i < len(deformed_points):
            transform["position"] = deformed_points[i]

    # TODO: implement later
    # 回転とスケールの処理（メッシュ以外の場合）
    # if not isinstance(original_obj, MeshObject):
    #     # ソースとターゲットの変換情報
    #     source_transforms = source_obj.get_transforms()
    #     target_transforms = target_obj.get_transforms()
    #
    #     # 各変換情報に対して処理
    #     for i, transform in enumerate(transforms):
    #         # 最も近いソースポイントを見つける
    #         source_positions = np.array([t["position"] for t in source_transforms])
    #
    #         closest_idx = np.argmin(np.sum((transform["position"] - source_positions)**2, axis=1))
    #
    #         # 回転の補間
    #         if "rotation" in transform and closest_idx < len(source_transforms):
    #             source_rot = Rotation.from_quat(source_transforms[closest_idx]["rotation"])
    #             target_rot = Rotation.from_quat(target_transforms[closest_idx]["rotation"])
    #
    #             # ソースからターゲットへの相対回転を計算
    #             rel_rot = source_rot.inv() * target_rot
    #
    #             # 元の回転に相対回転を適用
    #             orig_rot = Rotation.from_quat(transform["rotation"])
    #             transform["rotation"] = (orig_rot * rel_rot).as_quat()
    #
    #         # スケールの補間
    #         if "scale" in transform and closest_idx < len(source_transforms):
    #             source_scale = source_transforms[closest_idx]["scale"]
    #             target_scale = target_transforms[closest_idx]["scale"]
    #
    #             # スケール比率を計算
    #             scale_ratio = target_scale / np.maximum(source_scale, 1e-6)
    #             transform["scale"] = transform["scale"] * scale_ratio

    # 剛体変換の適用（メッシュのクラスタリング時）
    if apply_rigid_transform and labels is not None:
        transforms = __apply_rigid_transform_to_clusters(
            original_obj.get_points(),
            deformed_points,
            labels,
            transforms,
        )

    # 変換を新しいオブジェクトに適用
    new_obj.apply_transforms(transforms)

    # 階層構造を処理（maintain_hierarchyがTrueの場合）
    if maintain_hierarchy:
        # 子オブジェクトを処理
        children = original_obj.get_children()
        for child in children:
            # 子オブジェクトの複製は既に行われているはず（ジョイント階層など）
            # 対応する新しい子オブジェクトを見つける
            new_child_name = f"{child.name.split('|')[-1]}_retarget"
            new_child = None

            for potential_child in new_obj.get_children():
                if potential_child.name.endswith(new_child_name):
                    new_child = potential_child
                    break

            if new_child:
                # 子オブジェクトに対しても再帰的に処理を適用
                __apply_rbf_deformation(
                    source_obj,
                    target_obj,
                    child,
                    new_child,
                    weights,
                    kernel,
                    radius_coefficient,
                    angle,
                    sampling_stride,
                    apply_rigid_transform,
                    inpaint,
                    maintain_hierarchy,
                )


@util.timeit
def __apply_rigid_transform_to_clusters(
    before_points: np.ndarray,
    after_points: np.ndarray,
    labels: np.ndarray,
    transforms: list,
) -> list:
    """クラスターごとに剛体変換を適用."""
    unique_clusters = np.unique(labels[labels >= 0])

    for cluster_id in unique_clusters:
        # クラスターに属するインデックスを取得
        cluster_indices = np.where(labels == cluster_id)[0]

        # クラスターの点群
        before_cluster_points = before_points[cluster_indices]
        after_cluster_points = after_points[cluster_indices]

        # PCA分析によるRST変換の適用
        # 平均位置の計算
        mean_before = np.mean(before_cluster_points, axis=0)
        mean_after = np.mean(after_cluster_points, axis=0)

        # 中心化
        centered_before = np.asarray(before_cluster_points - mean_before)
        centered_after = np.asarray(after_cluster_points - mean_after)

        # PCA分析
        pca_before = PCA(n_components=min(3, before_cluster_points.shape[0]))
        pca_before.fit(centered_before)

        pca_after = PCA(n_components=min(3, after_cluster_points.shape[0]))
        pca_after.fit(centered_after)

        # スケール係数の計算
        before_extent = np.sqrt(pca_before.explained_variance_)
        after_extent = np.sqrt(pca_after.explained_variance_)
        scale_factors = after_extent / np.maximum(before_extent, 1e-6)
        uniform_scale = np.mean(scale_factors)

        # PCA空間でのスケーリング
        transformed_points = centered_before @ pca_before.components_.T
        scaled_points = transformed_points * uniform_scale

        # 元の空間に戻す
        rigid_transformed_points = scaled_points @ pca_before.components_ + mean_after

        # 変換情報を更新
        for i, idx in enumerate(cluster_indices):
            if idx < len(transforms):
                transforms[idx]["position"] = rigid_transformed_points[i]

    return transforms


@util.timeit
def __apply_uniform_scale_to_clusters(
    before_points: np.ndarray,
    after_points: np.ndarray,
    labels: np.ndarray,
    _weights: np.ndarray,
) -> np.ndarray:
    unique_clusters = np.unique(labels[labels >= 0])
    for cluster_id in unique_clusters:
        cluster_indices = np.where(labels == cluster_id)[0]
        before_cluster_points = before_points[cluster_indices]
        after_cluster_points = after_points[cluster_indices]

        points = __apply_rigid_transform_with_scaling(before_cluster_points, after_cluster_points)
        after_points[cluster_indices] = points

    return after_points


def __calculate_mesh_distance_matrix(
    source_points: np.ndarray,
    mesh_path: om.MDagPath,
    kernel: typing.Callable,
    radius: float,
    _weights: np.ndarray,
    _apply_rigid_transform: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the distance matrix for a single mesh."""
    points = util.convert_points_to_numpy(mesh_path)

    dist = get_distance_matrix(points, source_points, kernel, radius)

    return dist, points


def __apply_rigid_transform_with_scaling(
    before_cluster_points: np.ndarray,
    after_cluster_points: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transformation to the source cluster to match the target cluster.

    The transformation consists of a rotation, translation, and uniform scaling.
    """
    # Calculate the mean position of the source and target clusters
    mean_position_source = np.mean(before_cluster_points, axis=0)
    mean_position_temp = np.mean(after_cluster_points, axis=0)

    # Convert to centered coordinates
    centered_source = before_cluster_points - mean_position_source
    centered_temp = np.asarray(after_cluster_points - mean_position_temp)

    # PCA analysis
    pca_source = PCA(n_components=min(3, before_cluster_points.shape[0]))
    pca_source.fit(centered_source)

    pca_temp = PCA(n_components=min(3, after_cluster_points.shape[0]))
    pca_temp.fit(centered_temp)

    # Calculate the extent of the source and target clusters
    source_extent = np.sqrt(pca_source.explained_variance_)
    temp_extent = np.sqrt(pca_temp.explained_variance_)

    # Calculate the scale factors
    scale_factors = temp_extent / source_extent
    uniform_scale_factor = np.mean(scale_factors)

    # Apply the scaling to the source cluster in the PCA space
    transformed_points = centered_source @ pca_source.components_.T
    scaled_points = transformed_points * uniform_scale_factor

    # Transform back to the original space and return the deformed points
    scaled_points_world = scaled_points @ pca_source.components_ + mean_position_temp
    return scaled_points_world


def __apply_deformed_vertex_positions(
    mesh_path: om.MDagPath,
    deformed_points: np.ndarray,
    apply_rigid_transform: bool,
    inpaint: bool,
) -> om.MDagPath:
    """Sets the deformed points to the mesh.

    Duplicates the mesh and applies the deformed points.

    :param mesh_path: The mesh to apply deformed positions
    :param deformed_points: The calculated deformed positions as a numpy array
    :param apply_rigid_transform: Whether to apply a rigid transformation to the deformed points
    :param inpaint: Whether to inpaint the distance matrix for unconvinced vertices
    """
    # Convert deformed points to MPoint objects for Maya
    deformed_mpoints = [om.MPoint(*p) for p in deformed_points.tolist()]

    # Duplicate the shape and apply the deformed points
    mesh_name = mesh_path.fullPathName().split("|")[-2]

    name_parts = [mesh_name]
    if apply_rigid_transform:
        name_parts.append("rigid")
    if inpaint:
        name_parts.append("inpaint")

    new_name = "_".join(name_parts)
    dupe = cmds.duplicate(mesh_path.fullPathName(), name=new_name)[0]
    util.set_points(dupe, deformed_mpoints)

    dag = util.get_dag_path(dupe)
    if not dag:
        raise ValueError("Invalid mesh name")

    return dag
