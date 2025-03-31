
import numpy as np
import pytest
from maya import cmds
from maya.api import OpenMaya as om

from ymt_mesh_retarget.registration.core import BoneNode, JointNode


@pytest.fixture
def simple_cube_mesh():
    """Create a simple cube mesh fixture."""
    # Create a simple cube
    cube_name = cmds.polyCube(width=2, height=2, depth=2)[0]

    # Dynamically import to avoid circular references
    try:
        # Delayed import
        from ymt_mesh_retarget.objects.mesh import MeshObject
        mesh_obj = MeshObject(cube_name)
        yield mesh_obj
    finally:
        # Clean up
        if cmds.objExists(cube_name):
            cmds.delete(cube_name)


@pytest.fixture
def simple_joint_hierarchy():
    """Create a simple joint hierarchy fixture."""
    # Create a simple joint hierarchy
    cmds.select(clear=True)
    root = cmds.joint(position=(0, 0, 0), name="root")
    child1 = cmds.joint(position=(1, 0, 0), name="child1")
    cmds.select(root)
    child2 = cmds.joint(position=(0, 1, 0), name="child2")

    # Create JointNode objects
    joint_nodes = []
    for i, joint_name in enumerate([root, child1, child2]):
        sel = om.MSelectionList()
        sel.add(joint_name)
        dag_path = sel.getDagPath(0)
        pos = cmds.xform(joint_name, query=True, translation=True, worldSpace=True)
        matrix = cmds.xform(joint_name, query=True, matrix=True, worldSpace=True)

        joint_node = JointNode(
            path=dag_path,
            index=i,
            detail_name=joint_name,
            position=np.array(pos, dtype=np.float64),
            matrix=matrix,
        )
        joint_nodes.append(joint_node)

    # Create bone nodes
    bone_nodes = [
        BoneNode(start_joint_index=0, end_joint_index=1),
        BoneNode(start_joint_index=0, end_joint_index=2),
    ]

    try:
        yield joint_nodes, bone_nodes
    finally:
        # Clean up
        if cmds.objExists(root):
            cmds.delete(root)


@pytest.fixture
def mock_mesh_object():
    """Create a mock MeshObject fixture."""
    class MockMeshObject:
        def __init__(self, name="mock_mesh") -> None:
            self.name = name
            self.vertex_count = 8
            self.vertices = np.array([
                [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
            ], dtype=np.float64)

        def get_vertices(self):
            return self.vertices

        def get_vertex_count(self):
            return self.vertex_count

    return MockMeshObject()


@pytest.fixture
def mesh_cube_pair():
    """Create a pair of source and target cube meshes."""
    # Source mesh (basic cube)
    src_mesh_name = cmds.polyCube(width=2, height=2, depth=2, name="source_mesh")[0]

    # Target mesh (slightly deformed cube)
    tar_mesh_name = cmds.polyCube(width=2.2, height=1.8, depth=2.1, name="target_mesh")[0]
    cmds.move(0.5, 0.3, -0.2, tar_mesh_name)

    try:
        # Return mesh objects
        from ymt_mesh_retarget.objects.mesh import MeshObject
        src_mesh = MeshObject(src_mesh_name)
        tar_mesh = MeshObject(tar_mesh_name)

        yield src_mesh, tar_mesh
    finally:
        # Clean up
        if cmds.objExists(src_mesh_name):
            cmds.delete(src_mesh_name)
        if cmds.objExists(tar_mesh_name):
            cmds.delete(tar_mesh_name)
