# -*- coding: utf-8 -*-
"""
Main mesh registration functionality.

This module provides the MeshRegistration class that coordinates the process
of finding correspondence points between meshes.
"""

from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
from numpy.typing import NDArray

from maya import cmds
from maya.api import OpenMaya as om
from maya.api import OpenMayaAnim as oma

from ..util import (
    timeit,
    get_skin_cluster,
    viewport_off,
    autokey_off,
    one_undo,
)
from ..objects import MeshObject, create_retargetable_object

# Import from submodules
from .core import (
    CorrespondencePoint,
    RegistrationOptions
)
from .mapping import (
    get_mapping_points,
    find_correspondence_using_skeleton,
    create_optimized_correspondence_points
)
from .raycast import (
    perform_raycast,
    perform_raycast_with_options
)
from .alignment import (
    calculate_alignment_transform,
    get_joint_tree,
    match_joint_trees,
)
from .utils import (
    get_default_registration_options,
    validate_registration_options
)


class MeshRegistration:
    """Mesh Registration Class

    This class implements functionality to find corresponding points between meshes
    with different topologies.
    """

    def __init__(self, 
                 source_mesh: Union[str, MeshObject], 
                 target_mesh: Union[str, MeshObject],
                 options: Optional[RegistrationOptions] = None):
        """Initialize MeshRegistration

        Args:
            source_mesh: Source mesh name or object
            target_mesh: Target mesh name or object
            options: Registration options (uses defaults if None)
        """
        # Convert to mesh objects
        if isinstance(source_mesh, str):
            ret = create_retargetable_object(source_mesh)
            if not isinstance(ret, MeshObject):
                raise ValueError(f"Invalid source mesh object: {source_mesh} ({type(ret)})")
            self.source_mesh = ret
        else:
            self.source_mesh = source_mesh

        if isinstance(target_mesh, str):
            ret = create_retargetable_object(target_mesh)
            if not isinstance(ret, MeshObject):
                raise ValueError(f"Invalid target mesh object: {target_mesh}")
            self.target_mesh = ret
        else:
            self.target_mesh = target_mesh

        # Set options
        self.options = options if options is not None else get_default_registration_options()
        self.options = validate_registration_options(self.options)
        
        # Storage for correspondence point results
        self.correspondence_points: List[CorrespondencePoint] = []
        
        # Storage for joint data
        self.source_joint_paths = None
        self.source_joint_group = None
        self.source_bone_group = None
        self.target_joint_paths = None
        self.target_joint_group = None
        self.target_bone_group = None

    def find_correspondence_pairs(
            self, 
            sample_rate: Optional[float] = None,
            sample_number: Optional[int] = None, 
            sample_degree: Optional[float] = None,
            weight_decay: Optional[float] = None,
            align_spaces: Optional[bool] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find correspondence point pairs between meshes

        Args:
            sample_rate: Sampling rate for vertices (0.0-1.0)
            sample_number: Number of sampling rays
            sample_degree: Angle range for sampling (degrees)
            weight_decay: Weight decay coefficient
            align_spaces: Whether to align source and target spaces

        Returns:
            Tuple containing:
            - Source correspondence point coordinates (N, 3)
            - Target correspondence point coordinates (N, 3)
        """
        print("Starting correspondence search with Skeleton-Aware algorithm...")

        # Update options if parameters provided
        if sample_rate is not None:
            self.options.sample_rate = sample_rate
        if sample_number is not None:
            self.options.sample_number = sample_number
        if sample_degree is not None:
            self.options.sample_degree = sample_degree
        if weight_decay is not None:
            self.options.weight_decay = weight_decay
        if align_spaces is not None:
            self.options.align_spaces = align_spaces
            
        # Validate options
        self.options = validate_registration_options(self.options)

        # Get information from meshes
        source_points = self.source_mesh.get_points()
        target_points = self.target_mesh.get_points()

        # Get skinning weight information
        source_weights, source_joints = self._get_skin_weights(self.source_mesh)
        target_weights, target_joints = self._get_skin_weights(self.target_mesh)

        # Build joint trees if not already built
        if self.source_joint_group is None or self.target_joint_group is None:
            self.source_joint_paths, self.source_joint_group, self.source_bone_group = get_joint_tree(source_joints)
            self.target_joint_paths, self.target_joint_group, self.target_bone_group = get_joint_tree(target_joints)

        # For space alignment between source and target
        transform_matrix = None
        original_joint_positions = None
        original_joint_matrices = None

        if self.options.align_spaces:
            print("Aligning source bones to target space...")
            transform_matrix = calculate_alignment_transform(
                self.source_joint_group, 
                self.target_joint_group
            )

            if transform_matrix is not None:
                # Store original joint positions for later restoration
                original_joint_positions = []
                original_joint_matrices = []
                for joint in self.source_joint_group:
                    original_joint_positions.append(joint.position.copy())
                    original_joint_matrices.append(joint.matrix)

                # Apply the transformation to the source joint positions
                for i, joint in enumerate(self.source_joint_group):
                    pos = transform_matrix[i, :3]
                    joint.position = np.array(pos).squeeze()

                # Match joint trees to align bones
                match_joint_trees(
                    self.source_mesh,
                    self.target_mesh,
                    self.source_joint_group,
                    self.target_joint_group
                )
            else:
                print("Alignment failed. Skipping space alignment.")

        # Get the Maya mesh triangles
        print("Getting source mesh triangle information...")
        mesh_fn = self.source_mesh.mesh_fn
        _tri_counts, tri_indices = mesh_fn.getTriangles()
        src_triangle_indices = np.array(tri_indices, dtype=np.int32)

        # Calculate mapping points
        print("Calculating mapping points...")
        tar_mapping_points = get_mapping_points(
            target_points,
            self.target_joint_group,
            self.target_bone_group,
            target_weights,
            target_joints
        )
        print(f"Mapping points: {len(tar_mapping_points)}")

        # Find correspondence points using raycast
        print(f"Finding correspondences with {self.options.sample_number} rays at {self.options.sample_degree} degrees using {self.options.num_threads} threads...")
        raycast_result_array = perform_raycast_with_options(
            self.source_mesh,
            self.target_mesh,
            tar_mapping_points,
            src_triangles=source_points,
            src_triangle_indices=src_triangle_indices,
            src_joint_group=self.source_joint_group,
            tar_joint_group=self.target_joint_group,
            src_bone_group=self.source_bone_group,
            tar_bone_group=self.target_bone_group,
            options=self.options,
            num_threads=self.options.num_threads
        )
        print(f"Raycast results: {len(raycast_result_array)}")

        # Create correspondence points
        self.correspondence_points = create_optimized_correspondence_points(
            raycast_result_array,
            tar_mapping_points,
            target_points,
            max_points_per_target=self.options.max_points_per_target,
            min_weight_threshold=self.options.min_weight_threshold,
            distance_weight=self.options.distance_weight,
            ray_weight=self.options.ray_weight
        )
        print(f"Optimized correspondence points: {len(self.correspondence_points)}")

        # Convert results to numpy arrays
        if len(self.correspondence_points) == 0:
            # If advanced correspondence search fails, try simple skeleton-based method
            print("Advanced correspondence search failed. Trying simple skeleton-based method...")
            self.correspondence_points = find_correspondence_using_skeleton(
                source_points,
                target_points,
                source_weights,
                target_weights,
                source_joints,
                target_joints,
                self.options.sample_rate,
                self.options.weight_decay
            )

            if len(self.correspondence_points) == 0:
                raise ValueError("No correspondence points found. Check mesh connectivity and skeleton binding.")
        else:
            print(f"Found {len(self.correspondence_points)} correspondence points with advanced method.")

        # Extract point arrays from correspondence points using mesh function sets
        source_indices = [cp.source_index for cp in self.correspondence_points]
        target_indices = [cp.target_index for cp in self.correspondence_points]

        # Get points using the mesh function sets
        source_mesh_fn = self.source_mesh.mesh_fn
        target_mesh_fn = self.target_mesh.mesh_fn

        # Create point arrays
        source_points = np.zeros((len(source_indices), 3))
        target_points = np.zeros((len(target_indices), 3))

        # Extract vertex positions directly from mesh function sets
        for i, idx in enumerate(source_indices):
            if idx >= 0:  # Skip invalid indices
                point = source_mesh_fn.getPoint(idx)
                source_points[i] = [point.x, point.y, point.z]

        for i, idx in enumerate(target_indices):
            if idx >= 0:  # Skip invalid indices
                point = target_mesh_fn.getPoint(idx)
                target_points[i] = [point.x, point.y, point.z]

        # Restore original coordinates if alignment was used
        if self.options.align_spaces and transform_matrix is not None and original_joint_positions is not None:
            print("Restoring source bone positions to original space...")

            # Restore original joint positions
            for i, joint in enumerate(self.source_joint_group):
                joint.position = original_joint_positions[i]
                pos_array = np.array(joint.position).squeeze()
                cmds.xform(joint.path.fullPathName(), ws=True, t=pos_array)

            # Note: Since we're now using vertex indices instead of positions,
            # we don't need to manually transform the correspondence points
            # They will be automatically updated when we query the mesh

            print("Source bones restored to original space.")

        if self.options.align_spaces and original_joint_matrices is not None:
            for i, joint in enumerate(self.source_joint_group):
                joint.matrix = original_joint_matrices[i]
                cmds.xform(joint.path.fullPathName(), ws=True, m=joint.matrix)

        print("Correspondence search completed.")
        print(f"Source points: {source_points.shape}, Target points: {target_points.shape}")

        return source_points, target_points

    def _get_skin_weights(self, mesh_obj: MeshObject) -> Tuple[List[List[float]], List[str]]:
        """Get skinning weight information from mesh

        Args:
            mesh_obj: Mesh object

        Returns:
            Tuple containing:
            - List of skinning weights per vertex
            - List of joint names
        """
        # Find skin cluster
        fn_skin = self._find_skin_cluster(mesh_obj.dag_path)
        if not fn_skin:
            raise ValueError(f"No skin cluster found for mesh: {mesh_obj.name}")

        # Get joint information
        influence_objects = fn_skin.influenceObjects()  # type: om.MDagPathArray
        num_influences = len(influence_objects)

        # Create list of joint names
        joint_names = []
        for i in range(len(influence_objects)):
            dag_path = influence_objects[i]
            joint_name = dag_path.fullPathName()
            joint_names.append(joint_name)

        # Get weights for each vertex
        mesh_fn = mesh_obj.mesh_fn
        num_vertices = mesh_fn.numVertices

        # Create vertex component
        vert_indices = om.MIntArray([i for i in range(num_vertices)])
        vert_component = om.MFnSingleIndexedComponent().create(om.MFn.kMeshVertComponent)
        om.MFnSingleIndexedComponent(vert_component).addElements(vert_indices)

        # Create influence indices
        influence_indices = om.MIntArray([i for i in range(len(influence_objects))])

        # Get skin weights
        weights = fn_skin.getWeights(mesh_obj.dag_path, vert_component, influence_indices)

        # Convert to list format
        weights_list = []
        for i in range(num_vertices):
            vertex_weights = []
            for j in range(num_influences):
                weight = weights[i * num_influences + j]
                vertex_weights.append(weight)
            weights_list.append(vertex_weights)

        return weights_list, joint_names

    def _find_skin_cluster(self, mesh_path: om.MDagPath) -> Optional[oma.MFnSkinCluster]:
        """Find skin cluster for mesh

        Args:
            mesh_path: Mesh DAG path

        Returns:
            Skin cluster function set or None if not found
        """
        try:
            return get_skin_cluster(mesh_path)
        except ValueError as e:
            print(f"Warning: {e}")
            return None

    def visualize_correspondences(self, line_thickness: int = 1) -> str:
        """Visualize correspondence points

        Visualizes correspondence points by drawing lines between point pairs.

        Args:
            line_thickness: Line thickness for visualization

        Returns:
            Name of the created visualization group node
        """
        if not self.correspondence_points:
            raise ValueError("No correspondence points available. Call find_correspondence_pairs() first.")

        # Get mesh function sets
        source_mesh_fn = self.source_mesh.mesh_fn
        target_mesh_fn = self.target_mesh.mesh_fn

        group_name = visualize_correspondences(
                self.correspondence_points,
                source_mesh_fn,
                target_mesh_fn,
                line_thickness
        )

        return group_name


def visualize_correspondences(
        correspondence_points: List[CorrespondencePoint],
        source_mesh_fn: om.MFnMesh,
        target_mesh_fn: om.MFnMesh,
        line_thickness: int = 1
    ) -> str:
    """Create correspondence lines between source and target meshes.

    Args:
        correspondence_points: List of correspondence points
        source_mesh_fn: Source mesh MFnMesh
        target_mesh_fn: Target mesh MFnMesh
        line_thickness: Line thickness

    Returns:
        Name of the created transform node
    """
    # Create empty transform node for correspondence lines
    transform_name = cmds.createNode("transform", name="correspondenceLines")

    # Create a line for each correspondence point
    for i, cp in enumerate(correspondence_points):
        if cp.source_index < 0 or cp.target_index < 0:
            continue

        # Set color based on weight (red -> yellow -> green)
        color = [1, min(cp.weight * 2, 1), 0]

        # Get vertex positions
        s_pt = source_mesh_fn.getPoint(cp.source_index)
        t_pt = target_mesh_fn.getPoint(cp.target_index)
        src_pos = (float(s_pt.x), float(s_pt.y), float(s_pt.z))
        tar_pos = (float(t_pt.x), float(t_pt.y), float(t_pt.z))

        # Create a temporary curve for the line
        temp_curve = cmds.curve(
            p=[src_pos, tar_pos],  # 2-point straight line curve
            d=1,                   # degree=1
            name=f"tmp_line_{i}"
        )

        # Rename the curve shape node
        shape_node = cmds.listRelatives(temp_curve, shapes=True, fullPath=True)[0]
        shape_node = cmds.rename(shape_node, f"corrLineShape_{i}")

        # Parent the curve shape node to the transform node 
        cmds.parent(shape_node, transform_name, shape=True, relative=True)

        # Set line properties
        cmds.setAttr(f"{shape_node}.overrideEnabled", 1)
        cmds.setAttr(f"{shape_node}.overrideRGBColors", 1)
        cmds.setAttr(f"{shape_node}.overrideColorRGB", *color)
        cmds.setAttr(f"{shape_node}.lineWidth", line_thickness)

        # Delete the temporary curve transform
        cmds.delete(temp_curve)

    return transform_name   


@one_undo
@viewport_off
@autokey_off
@timeit       
def find_correspondence_pairs(
        source_mesh: Union[str, MeshObject],
        target_mesh: Union[str, MeshObject],
        sample_rate: float = 1.0,
        sample_number: int = 4, 
        sample_degree: float = 5.0,
        weight_decay: float = 2.0,
        align_spaces: bool = True,
        visualize: bool = False,
        num_threads: Optional[int] = None,
        options: Optional[RegistrationOptions] = None) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convenience function to find correspondence points between meshes

    Args:
        source_mesh: Source mesh
        target_mesh: Target mesh
        sample_rate: Vertex sampling rate (for performance)
        sample_number: Number of raycast samples
        sample_degree: Raycast angle range (degrees)
        weight_decay: Weight decay coefficient
        align_spaces: Whether to align source and target spaces based on matching joints
        visualize: Whether to visualize results
        num_threads: Number of threads to use for parallel processing
        options: Registration options (overrides individual parameters if provided)

    Returns:
        Tuple containing correspondence point coordinates for source and target
    """
    import multiprocessing
    
    # If no thread count specified, use a reasonable default
    if num_threads is None:
        cpu_count = multiprocessing.cpu_count()
        num_threads = max(2, min(cpu_count - 1, 8))
    
    # If options provided, use them; otherwise create from individual parameters
    if options is None:
        opts = RegistrationOptions(
            sample_rate=sample_rate,
            sample_number=sample_number,
            sample_degree=sample_degree,
            weight_decay=weight_decay,
            align_spaces=align_spaces,
            num_threads=num_threads
        )
    else:
        opts = options
        # Override thread count if explicitly provided
        if num_threads is not None:
            opts.num_threads = num_threads
    
    # Create registration object and find correspondence pairs
    registration = MeshRegistration(source_mesh, target_mesh, options=opts)
    source_points, target_points = registration.find_correspondence_pairs()

    # Visualize if requested
    if visualize:
        registration.visualize_correspondences()

    return source_points, target_points