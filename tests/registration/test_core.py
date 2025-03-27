import numpy as np
import pytest

from maya.api import OpenMaya as om

from ymt_mesh_retarget.registration.core import (
    JointNode,
    BoneNode,
    CorrespondencePoint,
    MappingNode,
    MappingResult,
    RegistrationOptions
)


def test_joint_node():
    """JointNodeクラスのテスト"""
    # モックMDagPath
    class MockMDagPath:
        def __init__(self, name):
            self.name = name
            
        def fullPathName(self):
            return self.name
    
    # JointNodeを作成
    mock_path = MockMDagPath("Root|Joint1")
    position = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    joint = JointNode(
        path=mock_path,
        index=1,
        detail_name="Root|Joint1",
        position=position
    )
    
    # プロパティのテスト
    assert joint.path == mock_path
    assert joint.index == 1
    assert joint.detail_name == "Root|Joint1"
    assert np.array_equal(joint.position, position)
    
    # 短い名前を取得するメソッドのテスト
    assert joint.short_name == "Joint1"
    
    # 名前空間付きの場合もテスト
    joint2 = JointNode(
        path=MockMDagPath("Root|Character:Joint1"),
        index=2,
        detail_name="Root|Character:Joint1",
        position=position
    )
    assert joint2.short_name == "Joint1"


def test_bone_node():
    """BoneNodeクラスのテスト"""
    # 有効なBoneNodeを作成
    bone = BoneNode(start_joint_index=0, end_joint_index=1)
    
    # プロパティのテスト
    assert bone.start_joint_index == 0
    assert bone.end_joint_index == 1
    assert bone.is_valid() == True
    
    # 無効なBoneNodeのテスト
    invalid_bone = BoneNode(start_joint_index=-1, end_joint_index=1)
    assert invalid_bone.is_valid() == False
    
    # 方向ベクトル取得のテスト
    # モックJointNode
    joint1 = JointNode(
        path=None,
        index=0,
        detail_name="Joint1",
        position=np.array([0.0, 0.0, 0.0])
    )
    
    joint2 = JointNode(
        path=None,
        index=1,
        detail_name="Joint2",
        position=np.array([1.0, 0.0, 0.0])
    )
    
    joints = [joint1, joint2]
    
    # 有効なボーンの方向ベクトル
    direction = bone.get_direction(joints)
    assert isinstance(direction, np.ndarray)
    assert np.array_equal(direction, np.array([1.0, 0.0, 0.0]))
    
    # 無効なボーンの方向ベクトル
    direction = invalid_bone.get_direction(joints)
    assert direction is None
    
    # 長さゼロのボーンのテスト
    zero_length_bone = BoneNode(start_joint_index=0, end_joint_index=2)
    joint3 = JointNode(
        path=None,
        index=2,
        detail_name="Joint3",
        position=np.array([0.0, 0.0, 0.0])  # joint1と同じ位置
    )
    joints.append(joint3)
    
    direction = zero_length_bone.get_direction(joints)
    assert direction is None  # 長さがゼロの場合はNoneを返す


def test_correspondence_point():
    """CorrespondencePointクラスのテスト"""
    # 基本的なCorrespondencePointを作成
    cp = CorrespondencePoint(source_index=1, target_index=2, score=0.8)
    
    # プロパティのテスト
    assert cp.source_index == 1
    assert cp.target_index == 2
    assert cp.score == 0.8


def test_mapping_node():
    """MappingNodeクラスのテスト"""
    # 基本的なMappingNodeを作成
    point = np.array([1.0, 2.0, 3.0])
    node = MappingNode(point=point, bone_index=1, distance=2.5, weight=0.7)
    
    # プロパティのテスト
    assert np.array_equal(node.point, point)
    assert node.bone_index == 1
    assert node.distance == 2.5
    assert node.weight == 0.7
    
    # 不正な値のクランプのテスト
    invalid_node = MappingNode(point=point, weight=1.5)  # 1.0以上の重み
    assert invalid_node.weight == 1.0  # 1.0にクランプされる
    
    invalid_node2 = MappingNode(point=point, weight=-0.5)  # 0.0未満の重み
    assert invalid_node2.weight == 0.0  # 0.0にクランプされる


def test_mapping_result():
    """MappingResultクラスのテスト"""
    # 基本的なMappingResultを作成
    result = MappingResult(vertex_index=5)
    
    # 初期状態のテスト
    assert result.vertex_index == 5
    assert len(result.node_array) == 0
    assert result.get_best_node() is None
    
    # ノードを追加
    node1 = MappingNode(
        point=np.array([1.0, 2.0, 3.0]),
        bone_index=1,
        weight=0.5
    )
    
    node2 = MappingNode(
        point=np.array([4.0, 5.0, 6.0]),
        bone_index=2,
        weight=0.8
    )
    
    result.add_node(node1)
    result.add_node(node2)
    
    # ノード追加後のテスト
    assert len(result.node_array) == 2
    assert result.node_array[0] == node1
    assert result.node_array[1] == node2
    
    # 最良ノード取得のテスト
    best_node = result.get_best_node()
    assert best_node == node2  # 重みが大きい方が選ばれる


def test_registration_options():
    """RegistrationOptionsクラスのテスト"""
    # デフォルト値でオプションを作成
    options = RegistrationOptions()
    
    # デフォルト値のテスト
    assert options.sample_count == 3000
    assert options.sample_number == 16
    assert options.sample_degree == 25.0
    assert options.align_spaces == True
    assert options.alignment_method == "umeyama"
    assert options.use_partial_matching == True
    assert options.use_hierarchical_matching == True
    assert options.match_confidence_threshold == 0.5
    assert options.max_points_per_target == 1
    assert options.use_scoring_components == True
    assert len(options.scoring_components) == 0  # 初期状態では空リスト
    
    # カスタム値でオプションを作成
    custom_options = RegistrationOptions(
        sample_count=1000,
        sample_number=32,
        sample_degree=45.0,
        weight_decay=3.0,
        alignment_method="rbf",
        use_partial_matching=False,
        name_conversion_map={"LeftArm": "arm_L"}
    )
    
    # カスタム値のテスト
    assert custom_options.sample_count == 1000
    assert custom_options.sample_number == 32
    assert custom_options.sample_degree == 45.0
    assert custom_options.weight_decay == 3.0
    assert custom_options.alignment_method == "rbf"
    assert custom_options.use_partial_matching == False
    assert "LeftArm" in custom_options.name_conversion_map
    assert custom_options.name_conversion_map["LeftArm"] == "arm_L"