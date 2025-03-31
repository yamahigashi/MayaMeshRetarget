import numpy as np
from maya.api import OpenMaya as om

from ymt_mesh_retarget.registration.core import BoneNode, JointNode


def test_joint_node_properties(simple_joint_hierarchy):
    """Test JointNode properties."""
    # Get joint hierarchy from fixture
    joint_nodes, _ = simple_joint_hierarchy

    # Root joint
    root_joint = joint_nodes[0]

    # Test basic properties
    assert root_joint.detail_name == "root"
    assert isinstance(root_joint.path, om.MDagPath)
    assert root_joint.index == 0

    # Test position
    assert isinstance(root_joint.position, np.ndarray)
    assert root_joint.position.shape == (3,)
    assert np.allclose(root_joint.position, [0, 0, 0], atol=0.01)

    # Test transformation matrix
    assert len(root_joint.matrix) == 16  # 4x4 matrix

    # Child joints
    child1_joint = joint_nodes[1]
    assert child1_joint.detail_name == "child1"
    assert np.allclose(child1_joint.position, [1, 0, 0], atol=0.01)

    child2_joint = joint_nodes[2]
    assert child2_joint.detail_name == "child2"
    assert np.allclose(child2_joint.position, [0, 1, 0], atol=0.01)


def test_joint_node_short_name():
    """Test the short_name property of JointNode."""
    # Mock MDagPath
    class MockMDagPath:
        def __init__(self, name) -> None:
            self.name = name

        def fullPathName(self):
            return self.name

    # Test various name patterns
    test_names = [
        ("simpleName", "simpleName"),
        ("Root|Joint1", "Joint1"),
        ("Root|Parent|Child", "Child"),
        ("Namespace:Object", "Object"),
        ("Root|Namespace:Object", "Object"),
        ("Root|NS1:NS2:Object", "Object"),
        ("|Root|Joint", "Joint"),
    ]

    for full_name, expected_short_name in test_names:
        mock_path = MockMDagPath(full_name)
        joint = JointNode(
            path=mock_path,
            index=0,
            detail_name=full_name,
            position=np.zeros(3),
        )
        assert joint.short_name == expected_short_name


def test_bone_node_properties(simple_joint_hierarchy):
    """Test BoneNode properties."""
    # Get joint hierarchy and bones from fixture
    joint_nodes, bone_nodes = simple_joint_hierarchy

    # First bone (root → child1)
    bone1 = bone_nodes[0]

    # Test basic properties
    assert bone1.start_joint_index == 0
    assert bone1.end_joint_index == 1
    assert bone1.is_valid() is True

    # Test direction vector
    direction = bone1.get_direction(joint_nodes)
    expected_direction = np.array([1, 0, 0])  # X-axis direction
    assert np.allclose(direction, expected_direction, atol=1e-6)

    # Second bone (root → child2)
    bone2 = bone_nodes[1]
    assert bone2.start_joint_index == 0
    assert bone2.end_joint_index == 2

    direction2 = bone2.get_direction(joint_nodes)
    expected_direction2 = np.array([0, 1, 0])  # Y-axis direction
    assert np.allclose(direction2, expected_direction2, atol=1e-6)


def test_invalid_bone_node():
    """Test invalid BoneNode cases."""
    # Bone with invalid index
    invalid_bone = BoneNode(start_joint_index=-1, end_joint_index=1)
    assert invalid_bone.is_valid() is False

    # Create joint list
    joint_nodes = [
        JointNode(path=None, index=0, detail_name="j0", position=np.array([0, 0, 0])),
        JointNode(path=None, index=1, detail_name="j1", position=np.array([1, 0, 0])),
    ]

    # Direction of invalid bone should be None
    direction = invalid_bone.get_direction(joint_nodes)
    assert direction is None

    # Zero-length bone (joints at same position)
    zero_length_bone = BoneNode(start_joint_index=0, end_joint_index=2)

    # Add third joint at same position as first
    joint_nodes.append(
        JointNode(path=None, index=2, detail_name="j2", position=np.array([0, 0, 0])),
    )

    # Direction of zero-length bone should also be None
    direction = zero_length_bone.get_direction(joint_nodes)
    assert direction is None
