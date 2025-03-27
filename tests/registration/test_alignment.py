import numpy as np
import pytest

from maya import cmds

from ymt_mesh_retarget.registration.alignment import (
    calculate_umeyama_transform,
    calculate_weighted_umeyama_transform
)


def test_calculate_umeyama_transform():
    """Umeyama変換行列計算のテスト"""
    # ソース点群 - 単位立方体の頂点
    src_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    
    # ターゲット点群 - 単位立方体を平行移動、回転、スケール
    # 2倍に拡大して[2,3,4]だけ平行移動した単位立方体
    rot_matrix = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
        [0, 0, 1]
    ])
    
    scale = 2.0
    translation = np.array([2, 3, 4])
    
    tar_points = scale * (src_points @ rot_matrix.T) + translation
    
    # Umeyama変換行列を計算
    transform = calculate_umeyama_transform(src_points, tar_points)
    
    # 結果が4x4行列であることを確認
    assert transform.shape == (4, 4)
    
    # 変換行列を適用して元の点群を変換
    homogeneous_src = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    transformed_src = homogeneous_src @ transform.T
    transformed_src = transformed_src[:, :3]  # 同次座標から3D座標に戻す
    
    # 変換された点群とターゲット点群が一致することを確認
    assert np.allclose(transformed_src, tar_points, atol=1e-10)
    
    # 変換行列の各成分を検証
    # スケール
    scale_component = transform[:3, :3]
    extracted_scale = np.linalg.norm(scale_component[0])
    assert np.isclose(extracted_scale, scale, atol=1e-10)
    
    # 回転（正規化したスケール成分を取得）
    normalized_rotation = scale_component / extracted_scale
    assert np.allclose(normalized_rotation, rot_matrix, atol=1e-10)
    
    # 平行移動
    assert np.allclose(transform[:3, 3], translation, atol=1e-10)


def test_calculate_weighted_umeyama_transform():
    """重み付きUmeyama変換行列計算のテスト"""
    # ソース点群
    src_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    
    # ターゲット点群 - 単位立方体を平行移動、回転、スケール
    # 2倍に拡大して[2,3,4]だけ平行移動した単位立方体
    rot_matrix = np.array([
        [np.cos(np.pi/4), -np.sin(np.pi/4), 0],
        [np.sin(np.pi/4), np.cos(np.pi/4), 0],
        [0, 0, 1]
    ])
    
    scale = 2.0
    translation = np.array([2, 3, 4])
    
    tar_points = scale * (src_points @ rot_matrix.T) + translation
    
    # 重み - すべての点に均等な重み
    weights = np.ones(src_points.shape[0])
    
    # 重み付きUmeyama変換行列を計算
    transform = calculate_weighted_umeyama_transform(src_points, tar_points, weights)
    
    # 結果が4x4行列であることを確認
    assert transform.shape == (4, 4)
    
    # 変換行列を適用して元の点群を変換
    homogeneous_src = np.hstack((src_points, np.ones((src_points.shape[0], 1))))
    transformed_src = homogeneous_src @ transform.T
    transformed_src = transformed_src[:, :3]  # 同次座標から3D座標に戻す
    
    # 変換された点群とターゲット点群が一致することを確認
    assert np.allclose(transformed_src, tar_points, atol=1e-10)
    
    # 異なる重みでもテスト
    # 一部の点だけの重みを高くする
    biased_weights = np.array([1.0, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.1])
    biased_weights = biased_weights / np.sum(biased_weights)  # 正規化
    
    # 重み付きUmeyama変換行列を計算
    biased_transform = calculate_weighted_umeyama_transform(src_points, tar_points, biased_weights)
    
    # 変換行列を適用して元の点群を変換
    biased_transformed_src = homogeneous_src @ biased_transform.T
    biased_transformed_src = biased_transformed_src[:, :3]
    
    # 高い重みの点が元の点群とよく一致することを確認
    # 高重みの点の誤差
    high_weight_error = np.linalg.norm(biased_transformed_src[:4] - tar_points[:4])
    # 低重みの点の誤差
    low_weight_error = np.linalg.norm(biased_transformed_src[4:] - tar_points[4:])
    
    # 高い重みの点の方が変換精度が高い（誤差が小さい）ことを確認
    assert high_weight_error <= low_weight_error