import os

import numpy as np
import pytest
from maya import cmds, standalone
from maya.api import OpenMaya as om

from ymt_mesh_retarget.registration.core import JointNode
from ymt_mesh_retarget.registration.utils import (
    calculate_average_nearest_distance,
    get_matched_info,
    mesh_bounding_box_center,
    mesh_bounding_box_size,
)


# テストのためのモックJointNodeを作成するユーティリティ関数
def create_mock_joint_node(name, position=None, index=0):
    """テスト用のモックJointNodeを作成する"""
    if position is None:
        position = np.zeros(3)
    
    # MDagPathのモックオブジェクト
    class MockDagPath:
        def __init__(self, name):
            self.name = name
        
        def fullPathName(self):
            return self.name
    
    mock_path = MockDagPath(name)
    
    return JointNode(
        path=mock_path,
        index=index,
        detail_name=name,
        position=np.array(position, dtype=np.float64)
    )


def test_get_matched_info():
    """get_matched_info関数のテスト"""
    # テスト用のジョイントノードを作成
    src_joints = [
        create_mock_joint_node("joint1", [0, 0, 0], 0),
        create_mock_joint_node("Left_arm", [1, 0, 0], 1),
        create_mock_joint_node("Right_arm", [-1, 0, 0], 2),
        create_mock_joint_node("leg_L", [0, -1, 0], 3),
        create_mock_joint_node("unknown_joint", [0, 1, 0], 4)
    ]
    
    tar_joints = [
        create_mock_joint_node("joint1", [0, 0, 0], 0),
        create_mock_joint_node("Left_arm", [1.2, 0, 0], 1),
        create_mock_joint_node("Right_arm", [-1.2, 0, 0], 2),
        create_mock_joint_node("L_leg", [0, -1.2, 0], 3),
        create_mock_joint_node("joint5", [0, 1.2, 0], 4)
    ]
    
    # 完全一致のみ
    src_indices, tar_indices, names, scores = get_matched_info(
        src_joints, 
        tar_joints, 
        use_partial_matching=False,
        use_hierarchical_matching=False
    )
    
    # 完全一致のみなので3つのジョイントがマッチすることを確認
    assert len(src_indices) == 3
    assert set(src_indices) == {0, 1, 2}
    assert set(tar_indices) == {0, 1, 2}
    assert len(scores) == 3
    assert all(score == 1.0 for score in scores)  # 完全一致は全て1.0
    
    # 部分一致を有効化
    src_indices, tar_indices, names, scores = get_matched_info(
        src_joints, 
        tar_joints, 
        use_partial_matching=True,
        use_hierarchical_matching=False
    )
    
    # 部分一致により"leg_L"と"L_leg"もマッチすることを確認
    assert len(src_indices) == 4
    assert 3 in src_indices  # leg_Lのインデックス
    
    # 名前変換マップを使ったテスト
    name_map = {"leg_L": "L_leg"}
    src_indices, tar_indices, names, scores = get_matched_info(
        src_joints, 
        tar_joints, 
        use_partial_matching=False,
        use_hierarchical_matching=False,
        name_conversion_map=name_map
    )
    
    # 名前変換マップにより"leg_L"と"L_leg"がマッチすることを確認
    assert len(src_indices) == 4
    assert 3 in src_indices  # leg_Lのインデックス


def test_mesh_bounding_box_functions():
    """メッシュバウンディングボックス関数のテスト"""
    # テスト用の単純な立方体を作成
    cube = cmds.polyCube(width=2, height=2, depth=2)[0]
    
    # メッシュオブジェクトのモック
    class MockMeshObject:
        def __init__(self, name):
            self.name = name
    
    mock_mesh = MockMeshObject(cube)
    
    # バウンディングボックスの中心をテスト
    center = mesh_bounding_box_center(mock_mesh)
    assert isinstance(center, np.ndarray)
    assert center.shape == (3,)
    # 原点中心の立方体なので中心は[0,0,0]に近いはず
    assert np.allclose(center, [0, 0, 0], atol=0.001)
    
    # バウンディングボックスのサイズをテスト
    size = mesh_bounding_box_size(mock_mesh)
    assert isinstance(size, np.ndarray)
    assert size.shape == (3,)
    # 2x2x2の立方体なのでサイズは[2,2,2]に近いはず
    assert np.allclose(size, [2, 2, 2], atol=0.001)
    
    # クリーンアップ
    cmds.delete(cube)


def test_calculate_average_nearest_distance():
    """平均最近傍距離計算のテスト"""
    # 正方形グリッド上の点群（各点は1.0単位で離れている）
    points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    
    avg_dist = calculate_average_nearest_distance(points)
    assert isinstance(avg_dist, float)
    
    # 正方形グリッドでは最近傍距離は1.0になるはず
    assert np.isclose(avg_dist, 1.0, atol=0.001)
    
    # ランダムな点群でもテスト
    rng = np.random.RandomState(42)  # 再現性のためにシードを固定
    random_points = rng.rand(100, 3) * 10.0
    
    avg_dist_random = calculate_average_nearest_distance(random_points)
    assert isinstance(avg_dist_random, float)
    assert avg_dist_random > 0.0