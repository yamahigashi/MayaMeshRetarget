# -*- coding: utf-8 -*-
"""
Mesh Retargeting Tool

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
import sys
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

from maya.api import (
    OpenMaya as om,
)
from maya import (
    mel,
    cmds
)

from . import (
    inpaint,
    cluster,
    util,
)

if sys.version_info[0] >= 3:
    import typing  # noqa: F401
    if typing.TYPE_CHECKING:
        from typing import (
            Callable,  # noqa: F401
        )
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
    def linear(cls, matrix, radius):
        # type: (np.ndarray, float) -> np.ndarray
        """Linear RBF - No scaling applied to distances."""
        return matrix

    @classmethod
    def gaussian(cls, matrix, radius):
        # type: (np.ndarray, float) -> np.ndarray
        """Gaussian RBF - Distances are scaled using a Gaussian decay function.
        
        This kernel is effective for producing smooth, gradual deformations.
        A smaller `radius` leads to faster decay and more localized effects.
        """
        return np.exp(-(matrix ** 2) / (radius ** 2))

    @classmethod
    def thin_plate(cls, matrix, radius):
        # type: (np.ndarray, float) -> np.ndarray
        """Thin plate spline RBF - Produces smooth, surface-based deformations.

        The `radius` scales the distances and determines the curvature of 
        the deformation. Smaller radii result in sharper curvatures.
        """
        result = (matrix / radius) ** 2
        np.warnings.filterwarnings("ignore")
        result = np.where(result > 0, np.log(result), result)
        np.warnings.filterwarnings("always")
        return result

    @classmethod
    def multi_quadratic_biharmonic(cls, matrix, radius):
        # type: (np.ndarray, float) -> np.ndarray
        """Multi-quadratic biharmonic RBF - Blends distances with a quadratic term.

        The `radius` controls the extent of influence. Larger values result in
        broader, smoother deformations.
        """
        return np.sqrt((matrix ** 2) + (radius ** 2))

    @classmethod
    def inv_multi_quadratic_biharmonic(cls, matrix, radius):
        # type: (np.ndarray, float) -> np.ndarray
        """Inverse multi-quadratic biharmonic RBF - Inverse decay of distances.

        The `radius` determines the decay rate. Small radii yield sharp fall-offs, 
        while larger radii provide broader influences.
        """
        return 1.0 / np.sqrt((matrix ** 2) + (radius ** 2))

    @classmethod
    def beckert_wendland_c2_basis(cls, matrix, radius):
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


def __select_rbf_kernel(kernel_name):
    # type: (str) -> Kernel
    """Select an RBF kernel by name."""

    kernels = {
        "linear": RBF.linear,
        "gaussian": RBF.gaussian,
        "thin_plate": RBF.thin_plate,
        "multi_quadratic": RBF.multi_quadratic_biharmonic,
        "inv_multi_quadratic": RBF.inv_multi_quadratic_biharmonic,
        "beckert_wendland": RBF.beckert_wendland_c2_basis
    }

    if kernel_name not in kernels:
        raise ValueError(f"Invalid kernel name: {kernel_name}")

    return kernels[kernel_name]


def __calculate_rbf_weight_matrix(source_points, target_points, kernel, radius):
    # type: (np.ndarray, np.ndarray, Kernel, float) -> np.ndarray
    """Calculate the weight matrix for the RBF interpolation."""
    identity = np.ones((source_points.shape[0], 1))
    dist = __get_distance_matrix(source_points, source_points, kernel, radius)
    dim = 3
    a = np.bmat([
        [dist, identity, source_points],
        [identity.T, np.zeros((1, 1)), np.zeros((1, dim))],
        [source_points.T, np.zeros((dim, 1)), np.zeros((dim, dim))]
    ])
    b = np.bmat([[target_points], [np.zeros((1, dim))], [np.zeros((dim, dim))]])
    return np.linalg.solve(a, b)


def __get_distance_matrix(v1, v2, kernel, radius):
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
        source,
        target,
        meshes,
        kernel=RBF.linear,
        radius_coefficient=0.0005,
        sampling_stride=1,
        apply_rigid_transform=False,
        inpaint=True
):
    # type: (str, str, list[str]|str, Kernel|str, float, int, bool, bool) -> list[str]
    """Run the mesh retarget

    :param source: Source mesh
    :param target: Modified source mesh
    :param meshes: List of meshes to retarget
    :param kernel: One of the RBF functions. See class RBF
    :param radius_coefficient: Smoothing parameter for the RBF
    :param sampling_stride: Vertex stride to sample on the source mesh. Increase to speed
                   up the calculation but less accurate.
    :param apply_rigid_transform: Whether to apply a rigid transformation to the deformed points
    :param inpaint: Whether to inpaint the distance matrix for unconvinced vertices
    """
    source_dag = util.get_mesh_dag(source)
    target_dag = util.get_mesh_dag(target)

    if isinstance(meshes, str):
        meshes = [meshes]

    if isinstance(kernel, str):
        kernel = __select_rbf_kernel(kernel)

    meshes_dag = [util.get_mesh_dag(mesh) for mesh in meshes]
    meshes_dag = [m for m in meshes_dag if m is not None]

    # Remove duplicate meshes, MDagPath could not be compared directly and is not hashable
    tmp = []
    tmp2 = []
    for dag in meshes_dag:
        name = dag.fullPathName()
        if name not in tmp:
            tmp2.append(dag)
            tmp.append(name)

    meshes_dag = tmp2

    if not source_dag or not target_dag or not meshes_dag:
        raise ValueError("Invalid mesh name")

    start_time = time.time()
    try:
        deformed_meshes = __retarget(
                source_dag,
                target_dag,
                meshes_dag,
                kernel,
                radius_coefficient,
                sampling_stride,
                apply_rigid_transform,
                inpaint)

    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        bar = mel.eval("$tmp = $gMainProgressBar")
        cmds.progressBar(bar, edit=True, endProgress=True)

    end_time = time.time()
    print(f"Retargeting completed in {end_time - start_time:.2f} seconds ({kernel.__name__})")

    return [m.fullPathName() for m in deformed_meshes]


def __retarget(
        source_path,
        target_path,
        mesh_paths,
        kernel=RBF.linear,
        radius_coefficient=0.0005,
        sampling_stride=1,
        apply_rigid_transform=False,
        inpaint=True
):
    # type: (om.MDagPath, om.MDagPath, list[om.MDagPath], Kernel, float, int, bool, bool) -> list[om.MDagPath]
    """Run the mesh retarget.
    :param source: Source mesh
    :param target: Modified source mesh
    :param meshes: List of meshes to retarget
    :param kernel: One of the RBF functions. See class RBF
    :param radius_coefficient: Smoothing parameter for the RBF
    :param sampling_stride: Vertex sampling_stride to sample on the source mesh. Increase to speed up
                   the calculation but less accurate.
    :param apply_rigid_transform: Whether to apply a rigid transformation to the deformed points
    :param inpaint: Whether to inpaint the distance matrix for unconvinced vertices
    """

    # Extract points from source and target meshes
    source_points = util.convert_points_to_numpy(source_path, sampling_stride)
    target_points = util.convert_points_to_numpy(target_path, sampling_stride)

    if source_points.shape != target_points.shape:
        raise ValueError("Source and target meshes must have the same number of vertices")

    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Prepare Retargeting",
                maxValue=100)

    # Get the weight matrix
    radius = __calculate_radius_from_bounding_box(target_path, radius_coefficient)
    weights = __calculate_rbf_weight_matrix(source_points, target_points, kernel, radius)

    deformed_points = __calculate_rbf_deformed_positions(
            source_path,
            mesh_paths,
            source_points,
            weights,
            kernel,
            radius,
            apply_rigid_transform,
            inpaint)

    if not cmds.about(batch=True):
        cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Calculate deformed positions",
                maxValue=len(mesh_paths))

    deformed_meshes = []
    for mesh_name, position in deformed_points.items():
        mesh_path = util.get_mesh_dag(mesh_name)
        if not mesh_path:
            print(f"Invalid mesh name: {mesh_name}")
            continue

        deformed = __apply_deformed_vertex_positions(mesh_path, position, radius, kernel)
        deformed_meshes.append(deformed)

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1)

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)

    return deformed_meshes


@util.timeit
def __calculate_rbf_deformed_positions(
        source_path,
        mesh_paths,
        source_points,
        weights,
        kernel,
        radius,
        apply_rigid_transform=False,
        do_inpaint=True
):
    # type: (om.MDagPath, list[om.MDagPath], np.ndarray, np.ndarray, Kernel, float, bool, bool) -> dict[str, np.ndarray]
    """Applies the retargeting process to a single mesh

    Clusters the vertices of the mesh based on skinning weights and topology,
    then computes the deformed points using the RBF interpolation and updates the mesh.

    :param mesh_path: The mesh to deform
    :param source_points: The source mesh vertices
    :param weights: The weight matrix for the RBF interpolation
    :param kernel: The RBF kernel function
    :param radius: The radius parameter for the RBF
    :param apply_rigid_transform: Whether to apply a rigid transformation to the deformed points
    """

    all_points = []
    all_distances = []
    indices_map = {}
    vertex_offset = 0

    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Calculate deformed positions",
                maxValue=len(mesh_paths))

    for mesh_path in mesh_paths:

        mesh_name = mesh_path.fullPathName()
        dist, points = __calculate_mesh_distance_matrix(
                source_points,
                mesh_path,
                kernel,
                radius,
                weights,
                apply_rigid_transform)

        indices_map[mesh_name] = np.arange(vertex_offset, vertex_offset + points.shape[0])
        vertex_offset += points.shape[0]

        all_points.append(points)
        all_distances.append(dist)
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1)

    # Concatenate all data along the first axis (rows)
    combined_points = np.vstack(all_points)
    combined_distances = np.vstack(all_distances)
    if apply_rigid_transform:
        labels = cluster.cluster_vertices(mesh_paths)
    else:
        labels = np.full(vertex_offset, -1, dtype=int)
    print(f"clustered {len(np.unique(labels))} clusters")

    identity = np.ones((combined_points.shape[0], 1))

    if do_inpaint:
        combined_distances = inpaint.inpaint_distance(
            source_path,
            mesh_paths,
            combined_distances,
            labels)

    h_combined = np.bmat([[combined_distances, identity, combined_points]])
    deformed_points = np.dot(h_combined, weights)

    if apply_rigid_transform:
        deformed_points = __apply_uniform_scale_to_clusters(
            combined_points,
            deformed_points,
            labels,
            weights)

    deformed_positions = {}  # type: dict[str, np.ndarray]
    for mesh_path in mesh_paths:
        indices = indices_map[mesh_path.fullPathName()]
        deformed_positions[mesh_path.fullPathName()] = deformed_points[indices]

    return deformed_positions


@util.timeit
def __apply_uniform_scale_to_clusters(
        before_points,
        after_points,
        labels,
        weights
):

    unique_clusters = np.unique(labels[labels >= 0])
    for cluster_id in unique_clusters:

        cluster_indices = np.where(labels == cluster_id)[0]
        before_cluster_points = before_points[cluster_indices]
        after_cluster_points = after_points[cluster_indices]

        points = __apply_rigid_transform_with_scaling(before_cluster_points, after_cluster_points)
        after_points[cluster_indices] = points

    return after_points


@util.timeit
def __calculate_mesh_distance_matrix(source_points, mesh_path, kernel, radius, weights, apply_rigid_transform):
    # type: (np.ndarray, om.MDagPath, Kernel, float, np.ndarray, bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, float]]
    """Calculate the distance matrix for a single mesh."""

    points = util.convert_points_to_numpy(mesh_path)

    dist = __get_distance_matrix(points, source_points, kernel, radius)

    return dist, points


def __apply_rigid_transform_with_scaling(before_cluster_points, after_cluster_points):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
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


def __apply_deformed_vertex_positions(mesh_path, deformed_points, radius, kernel):
    # type: (om.MDagPath, np.ndarray, float, Kernel) -> om.MDagPath
    """Sets the deformed points to the mesh.
    
    Duplicates the mesh and applies the deformed points.
    
    :param mesh_path: The mesh to apply deformed positions
    :param deformed_points: The calculated deformed positions as a numpy array
    :param radius: The radius parameter for the RBF (used in naming the new mesh)
    :param kernel: The RBF kernel function (used in naming the new mesh)
    """

    # Convert deformed points to MPoint objects for Maya
    deformed_mpoints = [om.MPoint(*p) for p in deformed_points.tolist()]

    # Duplicate the shape and apply the deformed points
    mesh_name = mesh_path.fullPathName().split("|")[-2]

    rounded_radius = round(radius, 2)
    new_name = "{}_{}_{}".format(mesh_name, rounded_radius, kernel.__name__).replace(".", "_")
    dupe = cmds.duplicate(mesh_path.fullPathName(), name=new_name)[0]
    util.set_points(dupe, deformed_mpoints)

    dag = util.get_dag_path(dupe)
    if not dag:
        raise ValueError("Invalid mesh name")

    return dag


def __calculate_radius_from_bounding_box(mesh_path, coefficient=0.1):
    # type: (om.MDagPath|str, float) -> float
    """
    Calculate the radius based on the bounding box size of the target mesh.
    
    :param mesh_path: The target mesh DAG path
    :param coefficient: A scaling factor for the bounding box size (default 0.1)
    :return: Computed radius
    """
    bbox = util.get_bounding_box(mesh_path)
    diagonal_length = np.linalg.norm(np.array(bbox.min) - np.array(bbox.max))
    return diagonal_length * coefficient
