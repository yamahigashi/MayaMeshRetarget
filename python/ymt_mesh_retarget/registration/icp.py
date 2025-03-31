"""Iterative Closest Point (ICP) implementation for mesh retargeting.

This module provides ICP-based mesh registration with joint parameter optimization.
It aligns source mesh to target mesh by optimizing joint parameters through iterative
closest point matching and numerical optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from ..logger import logger
from ..objects import JointObject, MeshObject
from ..util import (
    compose_matrix,
    decompose_matrix,
    get_bounding_box,
    get_dag_path,
    get_inverse_bind_matrix,
    get_skin_weights,
    timeit,
)
from .core import BoneNode, JointNode


np.random.seed(42)


@dataclass
class ICPOptions:
    """Configuration options for ICP mesh registration.

    Attributes:
        max_iterations: Maximum number of iterations for the ICP algorithm
        convergence_threshold: Convergence threshold for early stopping
        sampling_rate: Percentage of vertices to use for correspondence finding (0.0-1.0)
        optimization_method: SciPy optimization method to use
        staged_optimization: Whether to use staged optimization (root→major→all joints)
        include_scale: Whether to include scale parameters in optimization
        root_only_iterations: Number of iterations to optimize only root joint
        major_joints_iterations: Number of iterations to optimize major joints
        damping_factor: Damping factor for parameter updates (0.0-1.0)
        verbose: Whether to print detailed information during optimization
        translation_ratio: Ratio of translation parameters in the optimization
        rotation_range: Range of rotation parameters in the optimization (degrees)
        scale_min: Minimum scale factor for optimization
        scale_max: Maximum scale factor for optimization
    """

    max_iterations: int = 2
    convergence_threshold: float = 0.001
    # convergence_threshold: float = 0.0001
    sampling_rate: float = 0.8
    optimization_method: str = "L-BFGS-B"
    staged_optimization: bool = True
    include_scale: bool = True
    root_only_iterations: int = 4
    major_joints_iterations: int = 1
    damping_factor: float = 0.5
    verbose: bool = True
    translation_ratio: float = 0.3
    rotation_range: float = 0.0
    scale_min: float = 0.5
    scale_max: float = 3.0
    majar_joint_names: list[str] = field(default_factory=lambda: ["root", "spine", "hip"])
    majar_joint_names_2nd: list[str] = field(default_factory=lambda: ["shoulder", "thigh", "leg"])


@dataclass
class JointParameter:
    """Joint transformation parameters for optimization.

    Attributes:
        translation: Translation parameters [tx, ty, tz]
        rotation: Rotation parameters [rx, ry, rz] (Euler angles in degrees)
        scale: Scale parameters [sx, sy, sz]
        joint_index: Index of the joint in the skeleton hierarchy
        active: Whether this joint is active in the current optimization stage
    """
    local_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))
    initial_matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float64))

    joint_index: int = -1
    active: bool = False

    def __init__(
        self,
        joint_path: Union[str, om.MDagPath, None] = None,
        local_matrix: Optional[np.ndarray] = None,
        joint_index: int = -1,
        active: bool = False,
    ) -> None:
        """Initialize joint parameters.

        Args:
            joint_path: Full path to the joint node in the Maya DAG
            local_matrix: Local transformation matrix for the joint
            joint_index: Index of the joint in the skeleton hierarchy
            active: Whether this joint is active in the current optimization stage
        """
        if isinstance(joint_path, str):
            dag_path = get_dag_path(joint_path)
        else:
            dag_path = joint_path

        if local_matrix is None and dag_path is None:
            raise ValueError("Invalid joint path or matrix for JointParameter")

        if local_matrix is None and dag_path is None:
            raise ValueError("Invalid joint path or matrix for JointParameter")

        if local_matrix is None:
            if dag_path is None:
                raise ValueError("Invalid joint path or matrix for JointParameter")
            local_matrix = cmds.xform(dag_path.fullPathName(), query=True, matrix=True, objectSpace=True)

        if local_matrix is None:
            raise ValueError("Invalid local matrix for JointParameter")

        self.local_matrix = local_matrix
        self.initial_matrix = local_matrix
        self.joint_index = joint_index
        self.active = active

    @property
    def translation(self) -> NDArray[np.float64]:
        """Get translation parameters."""
        return self.local_matrix[:3, 3]

    @translation.setter
    def translation(self, value: NDArray[np.float64]) -> None:
        """Set translation parameters."""
        self.local_matrix[:3, 3] = value

    @property
    def rotation(self) -> NDArray[np.float64]:
        """Get rotation parameters."""
        _, r, s = decompose_matrix(self.local_matrix)
        return r

    @rotation.setter
    def rotation(self, value: NDArray[np.float64]) -> None:
        """Set rotation parameters."""
        t, _, s = decompose_matrix(self.local_matrix)
        self.local_matrix = compose_matrix(t, value, s)

    @property
    def scale(self) -> NDArray[np.float64]:
        """Get scale parameters."""
        _, _, s = decompose_matrix(self.local_matrix)
        return s

    @scale.setter
    def scale(self, value: NDArray[np.float64]) -> None:
        """Set scale parameters."""
        t, r, _ = decompose_matrix(self.local_matrix)
        self.local_matrix = compose_matrix(t, r, value)

    @property
    def initial_translation(self) -> NDArray[np.float64]:
        """Get initial translation parameters."""
        return self.initial_matrix[:3, 3]

    @property
    def initial_rotation(self) -> NDArray[np.float64]:
        """Get initial rotation parameters."""
        _, r, s = decompose_matrix(self.initial_matrix)
        return r

    @property
    def initial_scale(self) -> NDArray[np.float64]:
        """Get initial scale parameters."""
        _, _, s = decompose_matrix(self.initial_matrix)
        return s

    def to_flat_array(self, include_scale: bool = True) -> NDArray[np.float64]:
        """Convert parameters to a flat array for optimization.

        Args:
            include_scale: Whether to include scale parameters

        Returns:
            Flat array of parameters
        """
        if include_scale:
            return np.concatenate([self.translation, self.rotation, self.scale])
        else:
            return np.concatenate([self.translation, self.rotation])

    def from_flat_array(self, flat_params: NDArray[np.float64], include_scale: bool = True) -> None:
        """Set parameters from a flat array.

        Args:
            flat_params: Flat array of parameters
            include_scale: Whether to include scale parameters
        """
        if include_scale and len(flat_params) >= 9:
            t = flat_params[0:3]
            r = flat_params[3:6]
            s = flat_params[6:9]
        else:
            t = flat_params[0:3]
            r = flat_params[3:6]
            s = np.ones(3, dtype=np.float64)
        self.local_matrix = compose_matrix(t, r, s)


class SkeletonState:
    """Representation of the current skeleton state during optimization.

    This class maintains the state of all joints during the ICP optimization process,
    including their parameters, hierarchy information, and utilities for applying
    transformations to the mesh vertices.

    Attributes:
        joint_parameters: List of joint parameters for all joints
        joint_nodes: List of joint node objects
        bone_nodes: List of bone node objects
        weights: Skinning weights for the mesh
        bind_positions: Original mesh vertex positions
        parent_indices: List of parent joint indices for each joint
        inv_bind_mats: List of inverse bind matrices for each joint
    """

    def __init__(
        self,
        joint_nodes: list[JointNode],
        bone_nodes: list[BoneNode],
        weights: list[list[float]],
        bind_positions: NDArray[np.float64],
        parent_indices: list[int],
        inv_bind_mats: list[np.ndarray],
    ) -> None:
        """Initialize the skeleton state.

        Args:
            joint_nodes: List of joint nodes (each has name, index, maybe parent info).
            bone_nodes: List of bone nodes (unused here except for reference).
            weights: Skinning weights, shape: (numVertices) -> list of (weight for each joint).
            bind_positions: The mesh's bind pose vertex positions (numVertices, 3).
            parent_indices: For each joint, the index of its parent joint (or -1 if root).
            inv_bind_mats: For each joint, the inverse bind matrix (4x4) in world space.
        """
        self.joint_nodes = joint_nodes
        self.bone_nodes = bone_nodes
        self.weights = weights               # 2D: weights[v][j] -> float
        self.bind_positions = bind_positions # (N,3) in the bind pose
        self.parent_indices = parent_indices # len = num_joints
        self.inv_bind_mats = inv_bind_mats   # len = num_joints, each is (4,4) np array

        # Initialize local T-R-S parameters for each joint
        self.joint_parameters: list[JointParameter] = []
        for i, _jn in enumerate(joint_nodes):
            matrix = cmds.xform(_jn.detail_name, query=True, matrix=True, objectSpace=True)
            np_matrix = np.array(matrix).reshape(4, 4).T
            jp = JointParameter(
                local_matrix=np_matrix,
                joint_index=i,
                active=False,
            )
            self.joint_parameters.append(jp)

        self.active_indices: list[int] = []

    def set_active_joints(self, indices: Union[list[int], set[int]]) -> None:
        """Set which joints are active in the current optimization stage.

        Args:
            indices: List of joint indices to activate
        """
        for param in self.joint_parameters:
            param.active = (param.joint_index in indices)
        self.active_indices = list(indices)

    def get_active_parameters_flat(self, include_scale: bool = True) -> NDArray[np.float64]:
        """Fllaten active joint parameters into a 1D array.

        Args:
            include_scale: Whether to include scale parameters

        Returns:
            Flat array of all active parameters
        """
        params_list = []
        for param in self.joint_parameters:
            if param.active:
                params_list.append(param.to_flat_array(include_scale))
        return np.concatenate(params_list) if params_list else np.array([])

    def set_active_parameters_from_flat(
        self,
        flat_params: NDArray[np.float64],
        include_scale: bool = True,
    ) -> None:
        """Unflatten parameters into each active joint's local T-R-S parameters.

        Args:
            flat_params: Flat array of parameters
            include_scale: Whether to include scale parameters
        """
        params_per_joint = 9 if include_scale else 6
        active_count = 0

        for param in self.joint_parameters:
            if param.active:
                start_idx = active_count * params_per_joint
                end_idx = start_idx + params_per_joint
                if start_idx < len(flat_params):
                    joint_params = flat_params[start_idx:end_idx]
                    param.from_flat_array(joint_params, include_scale)
                active_count += 1

    def compute_deformed_vertices(self) -> NDArray[np.float64]:
        """Compute skinned vertex positions given local TRS parameters and inverseBindMatrices.

        This applies the current joint transformations to the original vertices
        using skinning weights with optimized NumPy calculations.

        Returns:
            Deformed vertex positions
        """
        # 1) Compute global matrices (4x4) for each joint
        global_mats = self._compute_global_matrices()

        # 2) Skinning calculation
        #    v_deformed[i] = sum_j( w[i][j] * (global_mats[j] * invBind_j * bindPos_i ) )
        N = len(self.bind_positions)  # noqa: N806
        deformed = np.zeros((N, 3), dtype=np.float64)

        # Precompute homogeneous bind positions
        bind_h = np.hstack([self.bind_positions, np.ones((N, 1), dtype=np.float64)])  # shape (N,4)

        for i in range(N):
            weighted_pos = np.zeros(4, dtype=np.float64)
            w_list = self.weights[i]  # list of weights for each joint
            for j, wj in enumerate(w_list):
                if wj > 1e-5:
                    # global_mats[j] * inv_bind_mats[j]
                    mat_combo = global_mats[j] @ self.inv_bind_mats[j]
                    # transform the bind pos
                    pos_h = mat_combo @ bind_h[i]  # shape (4,)
                    weighted_pos += wj * pos_h
            # Convert back to 3D
            if abs(weighted_pos[3]) > 1e-6:
                weighted_pos /= weighted_pos[3]
            deformed[i] = weighted_pos[:3]

        return deformed

    def _compute_global_matrices(self) -> list[np.ndarray]:
        """Compute each joint's global 4x4 matrix by hierarchical FK."""
        n_joints = len(self.joint_parameters)
        global_mats = [np.eye(4, dtype=np.float64) for _ in range(n_joints)]

        for i in range(n_joints):
            jp = self.joint_parameters[i]

            # If parent == -1, it's root -> global = local
            # else global = parentGlobal * local
            p_idx = self.parent_indices[i]
            if p_idx < 0:
                global_mats[i] = jp.local_matrix
            else:
                global_mats[i] = global_mats[p_idx] @ jp.local_matrix

        return global_mats


@timeit
def compute_bbox_based_scaling(
    source_mesh: MeshObject, target_mesh: MeshObject,
) -> tuple[tuple[float, float, float], NDArray[np.float64]]:
    """Compute initial scaling and position based on bounding boxes.

    Args:
        source_mesh: Source mesh object
        target_mesh: Target mesh object

    Returns:
        Tuple of (scale_factor, position_offset)
    """
    if isinstance(source_mesh, str):
        source_mesh = MeshObject(source_mesh)
    if isinstance(target_mesh, str):
        target_mesh = MeshObject(target_mesh)

    # Get bounding boxes
    src_bbox = get_bounding_box(source_mesh.dag_path)
    tar_bbox = get_bounding_box(target_mesh.dag_path)

    # Compute diagonal lengths
    src_min = np.array([src_bbox.min.x, src_bbox.min.y, src_bbox.min.z])
    src_max = np.array([src_bbox.max.x, src_bbox.max.y, src_bbox.max.z])
    tar_min = np.array([tar_bbox.min.x, tar_bbox.min.y, tar_bbox.min.z])
    tar_max = np.array([tar_bbox.max.x, tar_bbox.max.y, tar_bbox.max.z])

    # Compute scale factor
    scale_factor = (tar_max - tar_min) / (src_max - src_min)

    # Compute center offset
    src_center = (src_min + src_max) / 2
    tar_center = (tar_min + tar_max) / 2
    position_offset = tar_center - src_center

    return scale_factor, position_offset


def find_closest_points(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
    sampling_rate: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Find closest points between source and target meshes.

    Args:
        source_points: Source mesh vertex positions
        target_points: Target mesh vertex positions
        sampling_rate: Percentage of source points to use (0.0-1.0)

    Returns:
        Tuple of (sampled_source_points, corresponding_target_points, sample_indices)
    """
    # Sample source points
    num_points = len(source_points)
    num_samples = max(1, int(num_points * sampling_rate))

    # If sampling rate is less than 1, randomly sample points
    if num_samples < num_points:
        sample_indices = np.random.choice(num_points, num_samples, replace=False)
        sampled_source_points = source_points[sample_indices]
    else:
        sampled_source_points = source_points
        sample_indices = np.arange(num_points)

    # Build kd-tree for target points
    target_tree = cKDTree(target_points)

    # Find closest target points
    _, indices = target_tree.query(sampled_source_points)

    # Get corresponding target points
    corresponding_target_points = target_points[indices]

    return sampled_source_points, corresponding_target_points, sample_indices.tolist()


def compute_error_metric_between_points(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
) -> float:
    """Compute error metric between source and target points.

    Args:
        source_points: Source mesh vertex positions
        target_points: Target mesh vertex positions (corresponding points)

    Returns:
        Mean squared error between point sets
    """
    return float(np.mean(np.sum((source_points - target_points) ** 2, axis=1)))


def compute_error_metric_between_meshes(
    source_mesh: MeshObject,
    target_mesh: MeshObject,
    sampling_rate: float = 1.0,
) -> float:
    """Compute error metric between two meshes after applying a transformation.

    Args:
        source_mesh: Source mesh object
        target_mesh: Target mesh object
        sampling_rate: Percentage of source points to use (0.0-1.0)

    Returns:
        Mean squared error between transformed source mesh and target mesh
    """
    if isinstance(source_mesh, str):
        source_mesh = MeshObject(source_mesh)

    if isinstance(target_mesh, str):
        target_mesh = MeshObject(target_mesh)

    source_points = source_mesh.get_smoothed_points()
    target_points = target_mesh.get_smoothed_points()
    sampled_src, corr_tgt, sample_idx = find_closest_points(
        source_points,
        target_points,
        sampling_rate,
    )

    return compute_error_metric_between_points(sampled_src, corr_tgt)


# ----------------------------------------------------------------
# Main Retargeting class
# ----------------------------------------------------------------
class MeshRetargetICP:
    """Main class for ICP-based mesh retargeting with joint optimization.

    This class implements the iterative closest point algorithm with joint parameter
    optimization for mesh retargeting between characters with different topologies.

    Attributes:
        source_mesh: Source mesh object
        target_mesh: Target mesh object
        source_root_joint: Root joint of the source skeleton
        options: ICP options
    """

    def __init__(
        self,
        source_mesh: Union[str, MeshObject],
        target_mesh: Union[str, MeshObject],
        source_root_joint: Union[str, JointObject],
        options: Optional[ICPOptions] = None,
    ) -> None:
        """Initialize ICP mesh retargeting.

        Args:
            source_mesh: Source mesh name or object
            target_mesh: Target mesh name or object
            source_root_joint: Source root joint name or object
            options: Optional ICP options
        """
        # Convert inputs to objects if needed
        self.source_mesh = source_mesh if isinstance(source_mesh, MeshObject) else MeshObject(source_mesh)
        self.target_mesh = target_mesh if isinstance(target_mesh, MeshObject) else MeshObject(target_mesh)
        self.source_root_joint = (
            source_root_joint if isinstance(source_root_joint, JointObject) else JointObject(source_root_joint)
        )

        # Set options
        self.options = options if options is not None else ICPOptions()

        # Initialize internal state
        self.joint_hierarchy = self.source_root_joint.get_joint_hierarchy()
        self.joint_objects = [JointObject(j) for j in self.joint_hierarchy]

        print(f"Joint hierarchy: {self.joint_hierarchy}")
        print(f"Joint objects: {self.joint_objects}")

        # Storage for results
        self.result_error = float("inf")
        self.iteration_errors = []
        self.transformed_source_vertices = None

        # Storage for joint nodes and bone nodes
        self.joint_nodes: list[JointNode] = []
        self.bone_nodes: list[BoneNode] = []
        self.skeleton_state: Optional[SkeletonState] = None

    @timeit
    def prepare_data(self) -> None:
        """Prepare mesh and skeleton data for optimization."""
        # 1) Get vertex data
        source_vertices = self.source_mesh.get_smoothed_points()  # (N,3)
        target_vertices = self.target_mesh.get_smoothed_points()  # (M,3)

        # 2) Skin weights + joint names
        #    Typically we also want the inverseBindMatrix for each joint from the skinCluster
        weights, joint_names = get_skin_weights(self.source_mesh.dag_path)

        # 3) Create joint nodes (with .matrix or position, etc.)
        joint_paths = []
        for joint_name in joint_names:
            try:
                dag_path = get_dag_path(joint_name)
                joint_paths.append(dag_path)
            except Exception:
                logger.warning(f"Failed to get DAG path for joint: {joint_name}")

        self.joint_nodes = []
        for i, path in enumerate(joint_paths):
            name = path.fullPathName()
            local_mat_list = cmds.xform(name, query=True, matrix=True, objectSpace=True)
            local_mat = np.array(local_mat_list).reshape(4,4)
            pos = local_mat[:3, 3]

            world_matrix_list = cmds.xform(name, query=True, matrix=True, worldSpace=True)
            node = JointNode(
                path=path,
                index=i,
                detail_name=name,
                position=np.array(pos, dtype=np.float64),
                matrix=world_matrix_list,
            )
            self.joint_nodes.append(node)

        # 4) Determine parent-child from Maya DAG
        #    parent_idx[i] = index of the parent joint, or -1 if none
        parent_indices = []
        for _, jnode in enumerate(self.joint_nodes):
            parent_names = cmds.listRelatives(jnode.detail_name, parent=True, fullPath=True) or []
            if not parent_names:
                parent_indices.append(-1)  # root
            else:
                p_name = parent_names[0]
                # find which jnode has this name
                p_index = -1
                for k, p_jnode in enumerate(self.joint_nodes):
                    if p_jnode.detail_name == p_name:
                        p_index = k
                        break
                parent_indices.append(p_index)

        # 5) Create bone nodes (optional usage here)
        self.bone_nodes = []
        for i, _jnode in enumerate(self.joint_nodes):
            if parent_indices[i] >= 0:
                bone = BoneNode(
                    start_joint_index=parent_indices[i],
                    end_joint_index=i,
                )
                self.bone_nodes.append(bone)

        # 6) Build inverseBindMatrices placeholder
        #    Real code would get them from the skinCluster properly.
        inv_bind_mats = []
        inv_bind_dict = get_inverse_bind_matrix(self.source_mesh.name)
        for jnode in self.joint_nodes:
            inv_bind = inv_bind_dict.get(jnode.detail_name, np.eye(4, dtype=np.float64))
            inv_bind_mats.append(inv_bind)

        # 7) Create the SkeletonState
        self.skeleton_state = SkeletonState(
            joint_nodes=self.joint_nodes,
            bone_nodes=self.bone_nodes,
            weights=weights,              # list of (N) each => #joints
            bind_positions=source_vertices,
            parent_indices=parent_indices,
            inv_bind_mats=inv_bind_mats,
        )

        if logger.isEnabledFor(10):  # DEBUG
            for i, jnode in enumerate(self.joint_nodes):
                logger.debug(f"Joint: {jnode.short_name}, parent={parent_indices[i]}")

        src_bbox = get_bounding_box(self.source_mesh.dag_path)
        src_min = np.array([src_bbox.min.x, src_bbox.min.y, src_bbox.min.z])
        src_max = np.array([src_bbox.max.x, src_bbox.max.y, src_bbox.max.z])
        self.source_diag = float(np.linalg.norm(src_max - src_min))

        logger.info(f"Prepared data: {len(source_vertices)} source vertices, "
                    f"{len(target_vertices)} target vertices, "
                    f"{len(self.joint_nodes)} joints, "
                    f"{len(self.bone_nodes)} bones"
                    f"source_diag: {self.source_diag}")


    @timeit
    def run_icp(self) -> float:
        """Run the ICP algorithm with joint parameter optimization.

        Returns:
            Final error metric value
        """
        if self.skeleton_state is None:
            self.prepare_data()

        if self.skeleton_state is None:
            raise ValueError("Failed to prepare data for ICP")

        # Reset error metrics
        self.result_error = float("inf")
        self.iteration_errors = []

        # Get target vertices for correspondence finding
        target_vertices = self.target_mesh.get_smoothed_points()  # (M,3)

        # Compute initial bbox-based scaling for root joint
        scale_factor, position_offset = compute_bbox_based_scaling(self.source_mesh, self.target_mesh)

        # Set initial root joint parameters
        root_param = self.skeleton_state.joint_parameters[0]
        current_translation = root_param.translation
        current_scale = root_param.scale
        root_param.translation = current_translation + position_offset
        root_param.scale = current_scale * scale_factor

        logger.info(f"Initial root joint translation: {current_translation} -> {root_param.translation}")
        logger.info(f"Initial root joint scale: {current_scale} -> {root_param.scale}")

        sampled_src, corr_tgt, sample_idx = find_closest_points(
            self.skeleton_state.bind_positions,
            target_vertices,
            self.options.sampling_rate,
        )
        prev_error = compute_error_metric_between_points(sampled_src, corr_tgt)
        logger.info(f"Before error: {prev_error:.6f}")

        # Staged optimization
        if self.options.staged_optimization:
            # Stage 1: Root joint only
            logger.info("Stage 1: Optimizing root joint only")
            self.skeleton_state.set_active_joints([0])  # Root joint index is 0
            self._run_optimization_stage(
                target_vertices,
                self.options.root_only_iterations,
            )

            # Stage 2: Major joints
            # Select major joints (root + limb bases) - typically indices 0, 1, 5, 9, 13 in humanoid skeletons
            # This is a simplification - you may need to select joints differently based on your skeleton
            major_joint_indices = {0}

            for i, jnode in enumerate(self.joint_nodes):
                short_name = jnode.short_name.lower()
                if any(key in short_name for key in self.options.majar_joint_names):
                    major_joint_indices.add(i)

            for i, jnode in enumerate(self.joint_nodes):
                short_name = jnode.short_name.lower()
                if any(key in short_name for key in self.options.majar_joint_names_2nd):
                    major_joint_indices.add(i)

            if len(major_joint_indices) <= 1:
                # fallback: first 5 joints if no matches
                major_joint_indices = set(range(min(5, len(self.joint_nodes))))

            logger.info(f"Stage 2: Optimizing major joints: {major_joint_indices}")
            self.skeleton_state.set_active_joints(major_joint_indices)
            self._run_optimization_stage(
                target_vertices,
                self.options.major_joints_iterations,
            )

            # Stage 3: All joints
            logger.info("Stage 3: Optimizing all joints")
            all_joint_indices = list(range(len(self.joint_nodes)))
            self.skeleton_state.set_active_joints(all_joint_indices)
            self._run_optimization_stage(
                target_vertices,
                self.options.max_iterations,
            )
        else:
            # Simple optimization - all joints at once
            logger.info("Optimizing all joints at once")
            all_joint_indices = list(range(len(self.joint_nodes)))
            self.skeleton_state.set_active_joints(all_joint_indices)
            self._run_optimization_stage(
                target_vertices,
                self.options.max_iterations,
            )

        # Compute final transformed vertices
        self.transformed_source_vertices = self.skeleton_state.compute_deformed_vertices()

        return self.result_error

    def _run_optimization_stage(
        self,
        target_vertices: NDArray[np.float64],
        max_iterations: int,
    ) -> None:
        """Run a single optimization stage with the current active joints.

        Repeatedly:
          1) deform source
          2) find nearest points
          3) measure error
          4) run local parameter optimization
          5) check convergence

        Args:
            target_vertices: Target mesh vertex positions
            max_iterations: Maximum number of iterations for this stage
        """

        state = self.skeleton_state
        if not state:
            raise ValueError("No skeleton state to optimize")

        sampled_src, corr_tgt, sample_idx = find_closest_points(
            state.compute_deformed_vertices(),
            target_vertices,
            self.options.sampling_rate,
        )
        prev_error = compute_error_metric_between_points(sampled_src, corr_tgt)
        logger.info(f"Initial error: {prev_error:.6f}")

        for iteration in range(max_iterations):
            # 1) Deform source with current skeleton params
            deformed_source_vertices = state.compute_deformed_vertices()

            # 2) Find closest points
            sampled_src, corr_tgt, sample_idx = find_closest_points(
                deformed_source_vertices,
                target_vertices,
                self.options.sampling_rate,
            )

            # 3) Compute error
            current_error = compute_error_metric_between_points(sampled_src, corr_tgt)
            self.iteration_errors.append(current_error)

            error_diff = abs(prev_error - current_error)
            if self.options.verbose:
                logger.info(f"Iteration {iteration}: error={current_error:.6f}, diff={error_diff:.6f}")

            if error_diff < self.options.convergence_threshold:
                logger.info(f"Converged at iteration {iteration+1}")
                self.result_error = current_error
                break

            # 4) Optimize w.r.t. sampled correspondences
            self._optimize_parameters(
                sample_idx,
                corr_tgt,
            )
            prev_error = current_error

        self.result_error = prev_error

    def _optimize_parameters(
        self,
        sample_indices: NDArray[np.int64],
        sample_targets: NDArray[np.float64],
    ) -> None:
        """Optimize joint parameters to minimize distance between point sets.

        Args:
            sample_indices: Indices of sampled vertices
            sample_targets: Corresponding target vertices
        """
        state = self.skeleton_state
        if not state:
            raise ValueError("No skeleton state to optimize")

        def objective_function(flat_params: NDArray[np.float64]) -> float:
            # 1) Update skeleton local TRS from optimizer
            state.set_active_parameters_from_flat(
                flat_params,
                self.options.include_scale,
            )

            # 2) Recompute deformed
            new_deformed = state.compute_deformed_vertices()

            # 3) Sample
            new_deformed_sample = new_deformed[sample_indices]

            # 4) Error
            return compute_error_metric_between_points(new_deformed_sample, sample_targets)

        # Current param guess
        initial_params = state.get_active_parameters_flat(self.options.include_scale)
        if len(initial_params) == 0:
            logger.warning("No active parameters")
            return

        # Build param bounds
        # translation : ±( translation_ratio * source_diag )
        # rotation    : ± rotation_range
        # scale       : [scale_min, scale_max]
        trans_bound = self.options.translation_ratio * self.source_diag
        rot_bound   = self.options.rotation_range
        scale_min   = self.options.scale_min
        scale_max   = self.options.scale_max

        # For each active joint, we compute:
        #   tx -> [curr_tx - trans_bound, curr_tx + trans_bound]
        #   ty -> ...
        #   tz -> ...
        #   rx -> [curr_rx - rot_bound,  curr_rx + rot_bound]
        #   ...
        #   sx,sy,sz -> [curr_s - ???, curr_s + ???] もしくは [scale_min, scale_max]
        #   (下例ではスケールは絶対指定で [scale_min, scale_max], それ以外は相対指定)
        bounds: list[tuple[float, float]] = []

        # bounds_per_joint = 9 if self.options.include_scale else 6
        active_idx_list = state.active_indices  # e.g. [0, 2, 5...]

        # We'll read the current T,R,S for each active joint
        # and build bounds correspondingly
        for joint_idx in active_idx_list:
            param = state.joint_parameters[joint_idx]

            t = param.initial_translation
            r = param.initial_rotation
            s = param.initial_scale

            # T: ± trans_bound around t
            tx_min = t[0]
            tx_max = t[0]
            ty_min = t[1] - trans_bound
            ty_max = t[1] + trans_bound
            tz_min = t[2] - trans_bound
            tz_max = t[2] + trans_bound
            bounds.append((tx_min, tx_max))  # for tx
            bounds.append((ty_min, ty_max))  # for ty
            bounds.append((tz_min, tz_max))  # for tz

            # R: ± rot_bound around r
            rx_min = r[0] - rot_bound
            rx_max = r[0] + rot_bound
            ry_min = r[1] - rot_bound
            ry_max = r[1] + rot_bound
            rz_min = r[2] - rot_bound
            rz_max = r[2] + rot_bound
            bounds.append((rx_min, rx_max))
            bounds.append((ry_min, ry_max))
            bounds.append((rz_min, rz_max))

            if self.options.include_scale:
                sx_min, sx_max = (s[0] * scale_min, s[0] * scale_max)
                sy_min, sy_max = (s[1] * scale_min, s[1] * scale_max)
                sz_min, sz_max = (s[2] * scale_min, s[2] * scale_max)
                bounds.append((sx_min, sx_max))
                bounds.append((sy_min, sy_max))
                bounds.append((sz_min, sz_max))

        result = minimize(
            objective_function,
            initial_params,
            method=self.options.optimization_method,
            bounds=bounds,
            options={"maxiter": 20, "disp": self.options.verbose},
        )

        # Apply final to skeleton
        state.set_active_parameters_from_flat(result.x, self.options.include_scale)

    def apply_results_to_maya(self) -> None:
        """Apply the optimized joint parameters to the Maya scene."""
        if self.skeleton_state is None:
            logger.warning("No skeleton state to apply")
            return

        for i, joint in enumerate(self.joint_nodes):
            params = self.skeleton_state.joint_parameters[i]
            joint_name = joint.detail_name
            t = params.translation
            tstr = f"{t[0]:.2f}, {t[1]:.2f}, {t[2]:.2f}"
            r = params.rotation
            rstr = f"{r[0]:.2f}, {r[1]:.2f}, {r[2]:.2f}"
            s = params.scale
            sstr = f"{s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}"
            logger.info(f"Apply to {joint.short_name} translation={tstr}, rotation={rstr}, scale={sstr}")

            cmds.xform(joint_name, matrix=params.local_matrix.T.flatten().tolist())

        logger.info(f"Applied optimized parameters to {len(self.joint_nodes)} joints")


def align_mesh_with_icp(
    source_mesh: Union[str, MeshObject],
    target_mesh: Union[str, MeshObject],
    source_root_joint: Union[str, JointObject],
    options: Optional[ICPOptions] = None,
) -> float:
    """Align source mesh to target mesh using ICP and joint optimization.

    This is the main function to call for mesh retargeting. It aligns the source mesh
    to the target mesh by optimizing joint parameters based on closest point matching.

    Args:
        source_mesh: Source mesh name or object
        target_mesh: Target mesh name or object
        source_root_joint: Source root joint name or object
        options: Optional ICP options

    Returns:
        Final error metric value
    """
    # Create ICP retargeting object
    retargeter = MeshRetargetICP(source_mesh, target_mesh, source_root_joint, options)

    # Prepare data
    retargeter.prepare_data()

    # Run ICP
    final_error = retargeter.run_icp()

    # Apply results to Maya scene
    retargeter.apply_results_to_maya()

    return final_error
