# -*- coding: utf-8 -*-
"""
Main mesh registration functionality.

This module provides the MeshRegistration class that coordinates the process
of finding correspondence points between meshes.
"""

from typing import List, Tuple, Union, Optional
import numpy as np

from maya import cmds
from maya.api import OpenMaya as om
from maya.api import OpenMayaAnim as oma

from ..util import timeit, get_skin_cluster
from ..objects import MeshObject, create_retargetable_object

# Import from submodules
from .core import CorrespondencePoint  # noqa: F401
from .mapping import (
    get_mapping_points,
    find_correspondence_using_skeleton,
    create_optimized_correspondence_points
)
from .raycast import perform_raycast
from .alignment import calculate_alignment_transform, get_joint_tree


class MeshRegistration:
    """Mesh Registration Class
    
    This class implements functionality to find corresponding points between meshes
    with different topologies.
    """
    
    def __init__(self, source_mesh, target_mesh):
        """Initialize MeshRegistration
        
        Args:
            source_mesh (str|MeshObject): Source mesh
            target_mesh (str|MeshObject): Target mesh
        """
        # Convert to mesh objects
        if isinstance(source_mesh, str):
            ret = create_retargetable_object(source_mesh)
            if not isinstance(ret, MeshObject):
                raise ValueError("Invalid source mesh object.")
            self.source_mesh = ret
        else:
            self.source_mesh = source_mesh
        
        if isinstance(target_mesh, str):
            ret = create_retargetable_object(target_mesh)
            if not isinstance(ret, MeshObject):
                raise ValueError("Invalid target mesh object.")
            self.target_mesh = ret
        else:
            self.target_mesh = target_mesh
        
        # Storage for correspondence point results
        self.correspondence_points = []  # type: List[CorrespondencePoint]
    
    def find_correspondence_pairs(
            self, 
            sample_rate: float = 1.0,
            sample_number: int = 32, 
            sample_degree: float = 5.0,
            weight_decay: float = 2.0,
            align_spaces: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Find correspondence point pairs between meshes
        
        Args:
            sample_rate (float): Sampling rate for vertices (0.0-1.0)
            sample_number (int): Number of sampling rays
            sample_degree (float): Angle range for sampling (degrees)
            weight_decay (float): Weight decay coefficient
            align_spaces (bool): Whether to align source and target spaces
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Coordinates of correspondence point pairs
                source_points: Source correspondence point coordinates (N, 3)
                target_points: Target correspondence point coordinates (N, 3)
        """
        print("Starting correspondence search with Skeleton-Aware algorithm...")
        
        # Get information from meshes
        source_points = self.source_mesh.get_points()
        target_points = self.target_mesh.get_points()
        
        # Get skinning weight information
        source_weights, source_joints = self._get_skin_weights(self.source_mesh)
        target_weights, target_joints = self._get_skin_weights(self.target_mesh)
        
        # Build joint trees
        source_joint_paths, src_joint_group, src_bone_group = get_joint_tree(source_joints)
        target_joint_paths, tar_joint_group, tar_bone_group = get_joint_tree(target_joints)
        
        # For space alignment between source and target
        transform_matrix = None
        original_joint_positions = None
        
        if align_spaces:
            # print(f"Aligning source bones to target space...")
            transform_matrix = calculate_alignment_transform(src_joint_group, tar_joint_group)
            
            if transform_matrix is not None:
                # Store original joint positions for later restoration
                original_joint_positions = []
                for joint in src_joint_group:
                    original_joint_positions.append(joint.position.copy())
                
                # Apply the transformation to the source joint positions
                for i, joint in enumerate(src_joint_group):
                    pos = transform_matrix[i, :3]
                    pos_array = np.array(pos).squeeze()
                    joint.position = pos
                    cmds.xform(joint.path.fullPathName(), ws=True, t=pos_array)
        
        # C) Build scene from the source mesh triangles
        #    1) Get the Maya mesh triangles
        print("Getting source mesh triangle information...")
        mesh_fn = self.source_mesh.mesh_fn
        _tri_counts, tri_indices = mesh_fn.getTriangles()
        src_triangle_indices = np.array(tri_indices, dtype=np.int32)
         
        # E) get mapping points
        print("Calculating mapping points...")
        tar_mapping_points = get_mapping_points(
            target_points,
            tar_joint_group,
            tar_bone_group,
            target_weights,
            target_joints
        )
        print(f"Mapping points: {len(tar_mapping_points)}")
         
        # F) Find correspondence points using raycast
        print(f"Finding correspondences with {sample_number} rays at {sample_degree} degrees...")
         
        raycast_result_array = perform_raycast(
            self.source_mesh,
            self.target_mesh,
            tar_mapping_points,
            src_triangles=source_points,       # (Ns,3)
            src_triangle_indices=src_triangle_indices,
            sample_number=sample_number,
            sample_degree=sample_degree,
            src_joint_group=src_joint_group,
            tar_joint_group=tar_joint_group,
            src_bone_group=src_bone_group,
            tar_bone_group=tar_bone_group
        )
        
        # Create correspondence points
        self.correspondence_points = create_optimized_correspondence_points(
            raycast_result_array,
            tar_mapping_points,
            target_points,
            max_points_per_target=1  # One correspondence point per target vertex
        )
        
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
                sample_rate,
                weight_decay
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
        if align_spaces and transform_matrix is not None and original_joint_positions is not None:
            print("Restoring source bone positions to original space...")
            
            # Restore original joint positions
            for i, joint in enumerate(src_joint_group):
                joint.position = original_joint_positions[i]
                pos_array = np.array(joint.position).squeeze()
                cmds.xform(joint.path.fullPathName(), ws=True, t=pos_array)
            
            # Note: Since we're now using vertex indices instead of positions,
            # we don't need to manually transform the correspondence points
            # They will be automatically updated when we query the mesh
            
            print("Source bones restored to original space.")
        
        print("Correspondence search completed.")
        print(f"Source points: {source_points.shape}, Target points: {target_points.shape}")
        
        return source_points, target_points
    
    def _get_skin_weights(self, mesh_obj: MeshObject) -> Tuple[List[List[float]], List[str]]:
        """Get skinning weight information from mesh
        
        Args:
            mesh_obj (MeshObject): Mesh object
            
        Returns:
            Tuple[List[List[float]], List[str]]: 
                Skinning weight information and list of joint names
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
            joint_name = dag_path.fullPathName().split("|")[-1]
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
            mesh_path (om.MDagPath): Mesh DAG path
            
        Returns:
            Optional[oma.MFnSkinCluster]: Skin cluster function set (None if not found)
        """
        return get_skin_cluster(mesh_path)
    
    def visualize_correspondences(self, line_thickness: int = 1) -> str:
        """Visualize correspondence points
        
        Visualizes correspondence points by drawing lines between point pairs.
        
        Args:
            line_thickness (int): Line thickness
            
        Returns:
            str: The name of the created group node
        """
        if not self.correspondence_points:
            raise ValueError("No correspondence points available. Call find_correspondence_pairs() first.")
        
        # Create group node for lines
        group_name = cmds.group(empty=True, name="correspondence_visualization")
        
        # Get mesh function sets
        source_mesh_fn = self.source_mesh.mesh_fn
        target_mesh_fn = self.target_mesh.mesh_fn
        
        # Draw line for each correspondence point
        for i, cp in enumerate(self.correspondence_points):
            # Skip invalid indices
            if cp.source_index < 0 or cp.target_index < 0:
                continue
                
            # Set line color based on weight (red-yellow-green)
            color = [1, min(cp.weight * 2, 1), 0]  # Higher weight is more yellow
            
            # Get vertex positions directly from mesh
            source_point = source_mesh_fn.getPoint(cp.source_index)
            target_point = target_mesh_fn.getPoint(cp.target_index)
            
            src_x = float(source_point.x)
            src_y = float(source_point.y)
            src_z = float(source_point.z)
            
            tar_x = float(target_point.x)
            tar_y = float(target_point.y)
            tar_z = float(target_point.z)
            
            # Draw line
            line_name = f"corr_line_{i}"
            curve = cmds.curve(
                degree=1,
                point=[
                    (src_x, src_y, src_z),
                    (tar_x, tar_y, tar_z)
                ],
                name=line_name
            )
            
            # Set line color and thickness
            cmds.setAttr(f"{curve}.overrideEnabled", 1)
            cmds.setAttr(f"{curve}.overrideRGBColors", 1)
            cmds.setAttr(f"{curve}.overrideColorRGB", color[0], color[1], color[2])
            cmds.setAttr(f"{curve}.lineWidth", line_thickness)
            
            # Add to group
            cmds.parent(curve, group_name)
        
        return group_name


def find_correspondence_pairs(
        source_mesh: Union[str, MeshObject],
        target_mesh: Union[str, MeshObject],
        sample_rate: float = 1.0,
        sample_number: int = 4, 
        sample_degree: float = 5.0,
        weight_decay: float = 2.0,
        align_spaces: bool = True,
        visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Convenience function to find correspondence points between meshes
    
    Args:
        source_mesh (str|MeshObject): Source mesh
        target_mesh (str|MeshObject): Target mesh
        sample_rate (float): Vertex sampling rate (for performance)
        sample_number (int): Number of raycast samples
        sample_degree (float): Raycast angle range (degrees)
        weight_decay (float): Weight decay coefficient
        align_spaces (bool): Whether to align source and target spaces based on matching joints
        visualize (bool): Whether to visualize results
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: Coordinates of correspondence point pairs
    """
    registration = MeshRegistration(source_mesh, target_mesh)
    source_points, target_points = registration.find_correspondence_pairs(
        sample_rate=sample_rate,
        sample_number=sample_number,
        sample_degree=sample_degree,
        weight_decay=weight_decay,
        align_spaces=align_spaces
    )
    
    if visualize:
        registration.visualize_correspondences()
    
    return source_points, target_points
