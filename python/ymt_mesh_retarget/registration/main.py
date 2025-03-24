"""Main mesh registration functionality.

This module provides the MeshRegistration class that coordinates the process
of finding correspondence points between meshes.
"""

from typing import Optional, Union

import numpy as np
from maya import cmds
from maya.api import OpenMaya as om
from maya.api import OpenMayaAnim as oma
from numpy.typing import NDArray

from ..logger import logger
from ..objects import MeshObject, create_retargetable_object
from ..util import (
    autokey_off,
    get_skin_cluster,
    one_undo,
    timeit,
    viewport_off,
)
from .alignment import (
    calculate_alignment_transform,
    get_joint_tree,
    match_joint_trees,
)

# Import from submodules
from .core import (
    CorrespondencePoint,
    RegistrationOptions,
)
from .mapping import (
    create_optimized_correspondence_points,
    find_correspondence_using_skeleton,
    get_mapping_points,
)
from .raycast import perform_raycast_with_options
from .utils import (
    get_default_registration_options,
    validate_registration_options,
)


class MeshRegistration:
    """Main class for finding correspondences between skinned meshes.

    The MeshRegistration class provides the primary functionality for establishing
    point correspondences between two skinned meshes with different topologies.
    It uses skeletal information to guide the mapping process, employing ray-casting
    techniques to identify corresponding points.

    This class maintains state across operations, allowing for advanced usage such as:
    - Incremental processing
    - Result visualization
    - Access to intermediate data
    - Customization of all registration parameters

    For simple use cases, consider the standalone `find_correspondence_pairs()` function.
    This class is intended for more complex workflows where you need access to the
    underlying data structures or want to customize the process further.

    Attributes:
        source_mesh: The source mesh object
        target_mesh: The target mesh object
        options: Registration options controlling the process
        correspondence_points: The found correspondence points

    Example:
        ```python
        from ymt_mesh_retarget.registration import (
            MeshRegistration,
            get_default_registration_options
        )

        # Create and customize options
        options = get_default_registration_options()
        options.sample_count = 3000 # Use 3000 sample vertices
        options.num_threads = 8    # Use 8 threads

        # Create registration object
        registration = MeshRegistration("sourceModel", "targetModel", options)

        # Find correspondence pairs
        source_points, target_points = registration.find_correspondence_pairs()

        # Visualize the results
        registration.visualize_correspondences()

        # Access the correspondence points directly if needed
        for cp in registration.correspondence_points:
            print(f"Source: {cp.source_index}, Target: {cp.target_index}, Weight: {cp.weight}")
        ```
    """

    def __init__(
        self,
        source_mesh: Union[str, MeshObject],
        target_mesh: Union[str, MeshObject],
        options: Optional[RegistrationOptions] = None,
    ) -> None:
        """Initialize a new MeshRegistration instance.

        Args:
            source_mesh: Source mesh name or MeshObject instance.
                This is the mesh that will be mapped to the target.

            target_mesh: Target mesh name or MeshObject instance.
                This is the mesh that the source will be mapped to.

            options: Registration options controlling the process.
                If None, default options will be used. You can get the defaults
                using get_default_registration_options().

        Raises:
            ValueError: If the provided meshes are invalid or don't exist.
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
        self.correspondence_points: list[CorrespondencePoint] = []

        # Storage for joint data
        self.source_joint_paths = None
        self.source_joint_group = None
        self.source_bone_group = None
        self.target_joint_paths = None
        self.target_joint_group = None
        self.target_bone_group = None

    def find_correspondence_pairs(
        self,
        sample_count: Optional[int] = None,
        sample_number: Optional[int] = None,
        sample_degree: Optional[float] = None,
        weight_decay: Optional[float] = None,
        align_spaces: Optional[bool] = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Find correspondence point pairs between source and target meshes.

        This method performs the correspondence search process, identifying matching points
        between the source and target meshes. It uses ray-casting techniques guided by skeletal
        information to find the best matches.

        You can optionally override specific registration options via the parameters.
        Any parameter passed as None will use the value from the options specified at
        initialization time.

        The method performs these key steps:
        1. Extract skin weight information from both meshes
        2. Build joint hierarchies for source and target
        3. Optionally align the source to match the target's space
        4. Calculate mapping points based on skeletal structure
        5. Perform ray-casting to find potential correspondences
        6. Optimize and filter correspondence points
        7. Store the results internally and return point coordinates

        Args:
            sample_count: Number of vertices to sample for correspondence search.
                Higher values improve accuracy but increase processing time.
                Default: Uses the value from options

            sample_number: Number of sampling rays per point.
                Higher values improve accuracy but increase processing time.
                Default: Uses the value from options

            sample_degree: Angle range for ray sampling in degrees.
                Controls the spread of rays around each point.
                Default: Uses the value from options

            weight_decay: Weight decay coefficient for joint influence.
                Controls how quickly the influence of a joint decreases with distance.
                Default: Uses the value from options

            align_spaces: Whether to align source and target coordinate spaces.
                When True, transforms the source to align with the target using joints.
                Default: Uses the value from options

        Returns:
            A tuple containing two numpy arrays:
            - Source correspondence point coordinates (N, 3)
            - Target correspondence point coordinates (N, 3)

            These point arrays can be used directly with RBF interpolation to
            transfer attributes between the meshes.

        Raises:
            ValueError: If no skin cluster is found on either mesh, or if no
                      correspondence points could be found between the meshes.

        Note:
            The correspondence points are also stored in the `correspondence_points`
            attribute for later access.
        """
        logger.info("Starting correspondence search with Skeleton-Aware algorithm...")

        # Update options if parameters provided
        if sample_count is not None:
            self.options.sample_count = sample_count
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
            logger.info("Aligning source bones to target space...")
            transform_matrix = calculate_alignment_transform(
                self.source_joint_group,
                self.target_joint_group,
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
                    self.target_joint_group,
                )
            else:
                logger.warning("Alignment failed. Skipping space alignment.")

        # Get the Maya mesh triangles
        logger.info("Getting source mesh triangle information...")
        mesh_fn = self.target_mesh.mesh_fn
        _tri_counts, tri_indices = mesh_fn.getTriangles()
        tar_triangle_indices = np.array(tri_indices, dtype=np.int32)

        # Calculate mapping points
        logger.info("Calculating mapping points...")
        src_mapping_points = get_mapping_points(
            source_points,
            self.source_joint_group,
            self.source_bone_group,
            source_weights,
            source_joints,
        )
        logger.info(f"Mapping points: {len(src_mapping_points)}")

        # Find correspondence points using raycast
        message = f"Finding correspondences with {self.options.sample_number} rays at "\
                "{self.options.sample_degree} degrees..."
        logger.info(message)

        raycast_result_array = perform_raycast_with_options(
            self.source_mesh,
            self.target_mesh,
            src_mapping_points,
            tar_triangles=target_points,
            tar_triangle_indices=tar_triangle_indices,
            src_joint_group=self.source_joint_group,
            tar_joint_group=self.target_joint_group,
            src_bone_group=self.source_bone_group,
            tar_bone_group=self.target_bone_group,
            options=self.options,
        )
        logger.info(f"Raycast results: {len(raycast_result_array)}")

        # Create correspondence points
        self.correspondence_points = create_optimized_correspondence_points(
            raycast_result_array,
            src_mapping_points,
            source_points,
            max_points_per_target=self.options.max_points_per_target,
            min_weight_threshold=self.options.min_weight_threshold,
            distance_weight=self.options.distance_weight,
            ray_weight=self.options.ray_weight,
        )
        logger.info(f"Optimized correspondence points: {len(self.correspondence_points)}")

        # Convert results to numpy arrays
        if len(self.correspondence_points) == 0:
            # If advanced correspondence search fails, try simple skeleton-based method
            logger.warning("Advanced correspondence search failed. Trying simple skeleton-based method...")
            self.correspondence_points = find_correspondence_using_skeleton(
                source_points,
                target_points,
                source_weights,
                target_weights,
                source_joints,
                target_joints,
                self.options.sample_count,
                self.options.weight_decay,
            )

            if len(self.correspondence_points) == 0:
                raise ValueError("No correspondence points found. Check mesh connectivity and skeleton binding.")
        else:
            logger.info(f"Found {len(self.correspondence_points)} correspondence points with advanced method.")

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
            logger.info("Restoring source bone positions to original space...")

            # Restore original joint positions
            for i, joint in enumerate(self.source_joint_group):
                joint.position = original_joint_positions[i]
                pos_array = np.array(joint.position).squeeze()
                cmds.xform(joint.path.fullPathName(), ws=True, t=pos_array)

        if self.options.align_spaces and original_joint_matrices is not None:
            for i, joint in enumerate(self.source_joint_group):
                joint.matrix = original_joint_matrices[i]
                cmds.xform(joint.path.fullPathName(), ws=True, m=joint.matrix)

        logger.info("Correspondence search completed.")
        logger.info(f"Source points: {source_points.shape}, Target points: {target_points.shape}")

        return source_points, target_points

    def _get_skin_weights(self, mesh_obj: MeshObject) -> tuple[list[list[float]], list[str]]:
        """Get skinning weight information from mesh.

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
        vert_indices = om.MIntArray(list(range(num_vertices)))
        vert_component = om.MFnSingleIndexedComponent().create(om.MFn.kMeshVertComponent)
        om.MFnSingleIndexedComponent(vert_component).addElements(vert_indices)

        # Create influence indices
        influence_indices = om.MIntArray(list(range(len(influence_objects))))

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
        """Find skin cluster for mesh.

        Args:
            mesh_path: Mesh DAG path

        Returns:
            Skin cluster function set or None if not found
        """
        try:
            return get_skin_cluster(mesh_path)
        except ValueError as e:
            logger.warning(f"{e}")
            return None

    def visualize_correspondences(self, line_thickness: int = 1) -> str:
        """Visualize correspondence points in the Maya viewport.

        Creates line objects in the Maya scene to visualize the correspondence between
        source and target points. Each line connects a source point to its corresponding
        target point. The lines are colored based on the correspondence weight:
        - Red: Low confidence correspondence
        - Yellow: Medium confidence
        - Green: High confidence correspondence

        This visualization is valuable for debugging and validating the registration
        results. It allows you to see which points were matched and the quality of
        the matching.

        Args:
            line_thickness: Line thickness for visualization (1-10).
                Higher values create thicker, more visible lines.
                Default: 1

        Returns:
            Name of the created visualization group node in the Maya scene.
            This transform node contains all the curve shapes representing the
            correspondence lines.

        Raises:
            ValueError: If no correspondence points are available. You must call
                      find_correspondence_pairs() before visualizing.

        Example:
            ```python
            registration = MeshRegistration("sourceModel", "targetModel")
            registration.find_correspondence_pairs()

            # Create visualization with thicker lines
            group_name = registration.visualize_correspondences(line_thickness=3)

            # Select the visualization in Maya
            from maya import cmds
            cmds.select(group_name)
            ```
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
            line_thickness,
        )

        return group_name


def visualize_correspondences(
    correspondence_points: list[CorrespondencePoint],
    source_mesh_fn: om.MFnMesh,
    target_mesh_fn: om.MFnMesh,
    line_thickness: int = 1,
) -> str:
    """Create visualization of correspondence points between source and target meshes.

    Creates a set of line curves in the Maya scene that connect corresponding points
    between the source and target meshes. Each line shows the mapping between a point
    on the source mesh and its corresponding point on the target mesh.

    The lines are color-coded based on the correspondence weight:
    - Red: Low confidence correspondence (weight near 0)
    - Yellow: Medium confidence correspondence (weight around 0.5)
    - Green: High confidence correspondence (weight near 1.0)

    This helps visualize both the correspondence mapping and the quality/confidence
    of each correspondence point.

    Args:
        correspondence_points: List of correspondence points containing the vertex
            indices and weights for each correspondence pair.

        source_mesh_fn: Source mesh MFnMesh object used to access vertex positions.

        target_mesh_fn: Target mesh MFnMesh object used to access vertex positions.

        line_thickness: Line thickness for visualization (1-10).
            Higher values create thicker, more visible lines.
            Default: 1

    Returns:
        Name of the created transform node that contains all the line curves.
        This node can be selected or manipulated in Maya like any other transform.

    Example:
        ```python
        from maya.api import OpenMaya as om

        # Get mesh function sets
        src_mesh_dag = om.MGlobal.getSelectionListByName("sourceMesh").getDagPath(0)
        tar_mesh_dag = om.MGlobal.getSelectionListByName("targetMesh").getDagPath(0)
        src_mesh_fn = om.MFnMesh(src_mesh_dag)
        tar_mesh_fn = om.MFnMesh(tar_mesh_dag)

        # Create visualization with thicker lines
        group_name = visualize_correspondences(
            correspondence_points,
            src_mesh_fn,
            tar_mesh_fn,
            line_thickness=3
        )
        ```
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
            d=1,  # degree=1
            name=f"tmp_line_{i}",
        )

        # Rename the curve shape node
        shape_node = cmds.listRelatives(temp_curve, shapes=True, fullPath=True)[0]
        shape_node = cmds.rename(shape_node, f"corrLineShape_{i}")

        # Parent the curve shape node to the transform node
        shape_node = cmds.parent(shape_node, transform_name, shape=True, relative=True)[0]

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
    sample_count: int = 3000,
    sample_number: int = 8,
    sample_degree: float = 25.0,
    weight_decay: float = 2.0,
    align_spaces: bool = True,
    visualize: bool = False,
    num_threads: Optional[int] = None,
    options: Optional[RegistrationOptions] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Find correspondence points between two skinned meshes.

    This is the main entry point for mesh registration. It identifies corresponding points
    between two skinned meshes with different topologies, using skeletal information to
    guide the mapping process. These correspondence points can be used for RBF interpolation
    when transferring attributes between the meshes.

    The function performs the following steps:
    1. Analyze skeletal information from both meshes
    2. Optionally align source and target spaces
    3. Identify potential correspondence points via ray-casting
    4. Filter and optimize correspondence pairs
    5. Optionally visualize the results

    Args:
        source_mesh: Source mesh name or MeshObject instance.
            This is the mesh that will be mapped to the target.

        target_mesh: Target mesh name or MeshObject instance.
            This is the mesh that the source will be mapped to.

        sample_count: Number of vertices to sample for correspondence search.
            Higher values improve accuracy but increase processing time.
            Default: 3000 vertices

        sample_number: Number of sample rays per point.
            Higher values improve accuracy but increase processing time.
            Default: 32 rays

        sample_degree: Raycast cone angle in degrees.
            Controls the angular spread of rays. Higher values cast rays in a wider
            cone, which helps detect more potential correspondences but may
            introduce noise. Lower values focus rays more directly.
            Default: 45.0 degrees

        weight_decay: Weight decay coefficient.
            Controls how quickly the influence of a joint decreases with distance.
            Higher values cause more rapid falloff.
            Default: 2.0

        align_spaces: Whether to align source and target coordinate spaces.
            When True, the function will attempt to transform the source mesh to align
            with the target mesh using the joint hierarchies before finding correspondences.
            Default: True

        visualize: Whether to visualize correspondence results in the Maya viewport.
            When True, creates curve objects in Maya that connect corresponding points.
            Default: False

        num_threads: Number of threads to use for parallel processing.
            If None, uses a reasonable default based on available CPU cores.
            Default: None (auto-detect)

        options: Complete RegistrationOptions instance.
            If provided, these options override all individual parameters above.
            Use this for more advanced configuration.
            Default: None

    Returns:
        A tuple containing two numpy arrays:
        - Source correspondence point coordinates (N, 3)
        - Target correspondence point coordinates (N, 3)

        These point arrays can be used directly with RBF interpolation functions to
        transfer attributes between the meshes.

    Example:
        Basic usage:
        ```python
        from ymt_mesh_retarget.registration import find_correspondence_pairs

        source_points, target_points = find_correspondence_pairs(
            source_mesh="sourceCharacter",
            target_mesh="targetCharacter",
            visualize=True
        )

        # Use the correspondences for attribute transfer
        # ...
        ```

        Advanced usage with custom options:
        ```python
        from ymt_mesh_retarget.registration import (
            find_correspondence_pairs,
            get_default_registration_options
        )

        # Customize options
        options = get_default_registration_options()
        options.sample_count = 1500 # Use 1500 sample vertices
        options.num_threads = 8    # Use 8 threads
        options.use_bvh = True     # Use BVH acceleration

        # Find correspondences with custom options
        source_points, target_points = find_correspondence_pairs(
            source_mesh="sourceCharacter",
            target_mesh="targetCharacter",
            options=options,
            visualize=True
        )
        ```

    Raises:
        ValueError: If no skin cluster is found on either mesh, or if no correspondence
                    points could be found between the meshes.
    """
    import multiprocessing

    # If no thread count specified, use a reasonable default
    if num_threads is None:
        cpu_count = multiprocessing.cpu_count()
        num_threads = max(2, min(cpu_count - 1, 8))

    # If options provided, use them; otherwise create from individual parameters
    if options is None:
        opts = RegistrationOptions(
            sample_count=sample_count,
            sample_number=sample_number,
            sample_degree=sample_degree,
            weight_decay=weight_decay,
            align_spaces=align_spaces,
            num_threads=num_threads,
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
