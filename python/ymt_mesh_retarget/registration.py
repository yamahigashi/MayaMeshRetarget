# -*- coding: utf-8 -*-
"""
Mesh Registration Module

This module implements functionality to find corresponding points between meshes with different topologies.
Based on the techniques from the "Skeleton-Aware Skin Weight Transfer" paper,
it generates correspondence point pairs for RBF interpolation between source and target meshes
with different vertex counts.

Main features:
- Correspondence search using skeletal information
- Vertex sampling and reduction
- Visualization of mesh registration results
"""

import math
import random
from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
try:
    import embreex
    from embreex import rtcore_scene as rtcs
    from embreex.mesh_construction import TriangleMesh
except ImportError:
    cmds.warning("embreex library not found. Using standard raycasting instead.")
    # 通常のレイキャスト関数にフォールバック
    raise

from maya.api import OpenMaya as om
from maya.api import OpenMayaAnim as oma
from maya import (
    cmds,
    mel,
)

from . import (
    util,
)
from .logic import (
    calculate_rbf_weight_matrix,
    get_distance_matrix,
    RBF,
)
from .objects import MeshObject, create_retargetable_object


@dataclass
class CorrespondencePoint:
    """Data class for storing correspondence point information"""
    source_index: int  # Vertex index in source mesh
    target_index: int  # Vertex index in target mesh
    source_position: np.ndarray  # Position in source mesh (3, )
    target_position: np.ndarray  # Position in target mesh (3, )
    weight: float = 1.0  # Weight (confidence) of the correspondence point


@dataclass
class MappingNode:
    """Data class for storing mapping node information"""
    point: np.ndarray  # Position of the point
    bone_index: int = -1  # Bone index
    distance: float = 0.0  # Distance
    weight: float = 0.0  # Weight


@dataclass
class MappingResult:
    """Data class for storing mapping results"""
    vertex_index: int = -1  # Vertex index
    node_array: List[MappingNode] = None  # type: ignore

    def __post_init__(self):
        if self.node_array is None:
            self.node_array = []


@dataclass
class JointNode:
    path: om.MDagPath
    index: int = 0
    detail_name: str = ""
    position: np.ndarray = None


@dataclass
class BoneNode:
    start_joint_index: int = -1
    end_joint_index: int = -1


@dataclass
class TriangleWeightIndex:
    weight_distance: float = 10000.0
    triangle_index: int = -1


def rand_cone_vector(direction, angle_degree, num_samples):
    """ランダムな円錐方向ベクトルを生成する

    C++版 main.cpp の rand_cone_vector 関数の Python 実装

    Args:
        direction (np.ndarray): 円錐の中心方向ベクトル（単位ベクトルでなくても可）
        angle_degree (float): 円錐の角度（度）
        num_samples (int): 生成するサンプル数

    Returns:
        np.ndarray: ランダムな方向ベクトルの配列 (num_samples, 3)
    """
    # 方向ベクトルを正規化
    direction = np.array(direction, dtype=np.float64)
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-10:
        direction = np.array([0, 0, 1])
    else:
        direction = direction / direction_norm

    # 度からラジアンに変換
    cone_angle = angle_degree * math.pi / 180.0

    # 結果配列を初期化
    try:
        result = np.zeros((num_samples, 3))
    except TypeError:
        print(f"type of num_samples: {type(num_samples)}")
        raise

    # シード設定
    random.seed()

    for i in range(num_samples):
        # ランダムなz座標（円錐の高さ方向）を生成
        # z ∈ [cos(cone_angle), 1]
        z = (random.random() * (1.0 - math.cos(cone_angle))) + math.cos(cone_angle)

        # ランダムな極角
        phi = random.random() * 2.0 * math.pi

        # 単位球面上の点を生成
        x = math.sqrt(1.0 - z * z) * math.cos(phi)
        y = math.sqrt(1.0 - z * z) * math.sin(phi)

        # 標準基底 (0,0,1) から direction への回転行列を計算
        u = np.cross(np.array([0, 0, 1]), direction)
        u_norm = np.linalg.norm(u)

        if u_norm < 1e-10:
            # direction が [0,0,1] または [0,0,-1] に近い場合
            if direction[2] > 0:
                result[i] = np.array([x, y, z])
            else:
                result[i] = np.array([x, y, -z])
            continue

        u = u / u_norm
        rot = math.acos(np.clip(direction[2], -1.0, 1.0))  # z座標とのなす角

        # ロドリゲスの回転公式による回転
        cos_rot = math.cos(rot)
        sin_rot = math.sin(rot)

        # 回転行列の計算
        rot_matrix = np.zeros((3, 3))
        # 対角成分
        rot_matrix[0, 0] = cos_rot + (1.0 - cos_rot) * u[0] * u[0]
        rot_matrix[1, 1] = cos_rot + (1.0 - cos_rot) * u[1] * u[1]
        rot_matrix[2, 2] = cos_rot + (1.0 - cos_rot) * u[2] * u[2]

        # 非対角成分
        rot_matrix[0, 1] = (1.0 - cos_rot) * u[0] * u[1] - sin_rot * u[2]
        rot_matrix[0, 2] = (1.0 - cos_rot) * u[0] * u[2] + sin_rot * u[1]
        rot_matrix[1, 0] = (1.0 - cos_rot) * u[1] * u[0] + sin_rot * u[2]
        rot_matrix[1, 2] = (1.0 - cos_rot) * u[1] * u[2] - sin_rot * u[0]
        rot_matrix[2, 0] = (1.0 - cos_rot) * u[2] * u[0] - sin_rot * u[1]
        rot_matrix[2, 1] = (1.0 - cos_rot) * u[2] * u[1] + sin_rot * u[0]

        # 方向ベクトルの回転
        rand_dir = rot_matrix @ np.array([x, y, z])
        result[i] = rand_dir

    return result


def ray_triangle_intersection(orig, dir_vec, v0, v1, v2, scale=1.0):
    """レイと三角形の交差判定

    C++版 main.cpp の ray_triangle_intersection 関数の Python 実装

    Args:
        orig (np.ndarray): レイの始点
        dir_vec (np.ndarray): レイの方向ベクトル
        v0 (np.ndarray): 三角形の頂点0
        v1 (np.ndarray): 三角形の頂点1
        v2 (np.ndarray): 三角形の頂点2
        scale (float): 三角形のスケール（中心からのスケーリング）

    Returns:
        tuple: (交差判定の結果, 交差点, レイのパラメータt)
    """
    # type and shape checking
    EPSILON = 1e-12

    # スケーリング
    center_tri = (v0 + v1 + v2) / 3.0
    v0_scaled = (v0 - center_tri) * scale + center_tri
    v1_scaled = (v1 - center_tri) * scale + center_tri
    v2_scaled = (v2 - center_tri) * scale + center_tri

    # 三角形のエッジベクトル
    e1 = v1_scaled - v0_scaled
    e2 = v2_scaled - v0_scaled

    # 法線ベクトル
    n = np.cross(e1, e2)
    ndd = np.dot(dir_vec, n)

    # レイが三角形に向かって打たれているか確認
    if ndd < 0:
        return False, None, -1

    # Möller–Trumbore アルゴリズム
    h = np.cross(dir_vec, e2)
    a = np.dot(e1, h)

    if -EPSILON < a < EPSILON:
        return False, None, -1  # レイと三角形が平行

    f = 1.0 / a
    s = orig - v0_scaled
    u = f * np.dot(s, h)

    if u < 0.0 or u > 1.0:
        return False, None, -1

    q = np.cross(s, e1)
    v = f * np.dot(dir_vec, q)

    if v < 0.0 or u + v > 1.0:
        return False, None, -1

    t = f * np.dot(e2, q)

    if t > EPSILON:
        intersection_point = orig + dir_vec * t
        return True, intersection_point, t

    return False, None, -1


def triangle_interpolation(v1, v2, v3, p):
    """三角形内の点の重心座標を計算

    C++版 main.cpp の triangle_interpolation 関数の Python 実装

    Args:
        v1 (np.ndarray): 三角形の頂点1
        v2 (np.ndarray): 三角形の頂点2
        v3 (np.ndarray): 三角形の頂点3
        p (np.ndarray): 重心座標を計算する点

    Returns:
        tuple: (w1, w2, w3) 重心座標
    """
    # 三角形のローカル座標系を構築
    x = v3 - v1
    y = np.cross(v2 - v1, x)
    z = np.cross(y, x)

    # 変換行列を構築
    triangle_local_matrix = np.zeros((4, 4))
    triangle_local_matrix[0, 0] = x[0]
    triangle_local_matrix[0, 1] = y[0]
    triangle_local_matrix[0, 2] = z[0]

    triangle_local_matrix[1, 0] = x[1]
    triangle_local_matrix[1, 1] = y[1]
    triangle_local_matrix[1, 2] = z[1]

    triangle_local_matrix[2, 0] = x[2]
    triangle_local_matrix[2, 1] = y[2]
    triangle_local_matrix[2, 2] = z[2]

    triangle_local_matrix[3, 0] = v1[0]
    triangle_local_matrix[3, 1] = v1[1]
    triangle_local_matrix[3, 2] = v1[2]
    triangle_local_matrix[3, 3] = 1.0

    # 点をローカル座標系に変換
    triangle_local_matrix_inv = np.linalg.inv(triangle_local_matrix)

    tv1 = np.append(v1, 1.0) @ triangle_local_matrix_inv
    tv2 = np.append(v2, 1.0) @ triangle_local_matrix_inv
    tv3 = np.append(v3, 1.0) @ triangle_local_matrix_inv
    tp = np.append(p, 1.0) @ triangle_local_matrix_inv

    # 重心座標を計算
    deno = (tv2[2] - tv3[2]) * (tv1[0] - tv3[0]) + (tv3[0] - tv2[0]) * (tv1[2] - tv3[2])
    w1 = ((tv2[2] - tv3[2]) * (tp[0] - tv3[0]) + (tv3[0] - tv2[0]) * (tp[2] - tv3[2])) / deno
    w2 = ((tv3[2] - tv1[2]) * (tp[0] - tv3[0]) + (tv1[0] - tv3[0]) * (tp[2] - tv3[2])) / deno
    w3 = 1.0 - w1 - w2

    return w1, w2, w3


def build_embree_scene_from_source(src_triangles):
    """
    ソースメッシュの頂点座標配列 src_triangles を元に、EmbreeX のシーンを構築する。
    
    Args:
        src_triangles (np.ndarray): shape = (num_tri, 3, 3)
            三角形数 = num_tri
            1つの三角形につき頂点が3つ、各頂点は xyz(3次元)

    Returns:
        (EmbreeScene, TriangleMesh) : シーンとメッシュ
    """
    scene = rtcs.EmbreeScene()
    mesh = TriangleMesh(scene, src_triangles)  # これでBVHを構築
    return scene, mesh


class MeshRegistration:
    """メッシュ登録クラス

    異なるトポロジを持つメッシュ間の対応点を見つけるためのクラスです。
    """

    source_mesh: MeshObject
    target_mesh: MeshObject

    def __init__(self, source_mesh, target_mesh):
        """初期化

        Args:
            source_mesh (str|MeshObject): ソースメッシュ
            target_mesh (str|MeshObject): ターゲットメッシュ
        """
        # メッシュオブジェクトに変換
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

        # 対応点結果の格納用
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

        Returns:
            Tuple[np.ndarray, np.ndarray]: Coordinates of correspondence point pairs (source_points, target_points)
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
        source_joint_paths, src_joint_group, src_bone_group = self._get_joint_tree(source_joints)
        target_joint_paths, tar_joint_group, tar_bone_group = self._get_joint_tree(target_joints)

        # For space alignment between source and target
        transform_matrix = None
        original_joint_positions = None

        if align_spaces:
            # print(f"Aligning source bones to target space...")
            transform_matrix = self._calculate_alignment_transform(src_joint_group, tar_joint_group)

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

        # C) Build EmbreeX scene from the source mesh triangles
        #    1) Get the Maya mesh triangles
        print("Getting source mesh triangle information...")
        mesh_fn = self.source_mesh.mesh_fn
        _tri_counts, tri_indices = mesh_fn.getTriangles()
        # num_ids = len(tri_indices)
        src_triangle_indices = np.array(tri_indices, dtype=np.int32)
     
        # E) get mapping points
        print("Calculating mapping points...")
        tar_mapping_points = self.get_mapping_points(
            target_points,
            tar_joint_group,
            tar_bone_group,
            target_weights,
            target_joints
        )
        print(f"Mapping points: {len(tar_mapping_points)}")
     
        # F) Instead of get_correspondence_points(...), call the new embree version
        print(f"Finding correspondences with {sample_number} rays at {sample_degree} degrees...")
     
        raycast_result_array = self.get_correspondence_points(
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
        self.correspondence_points = self.create_optimized_correspondence_points(
            raycast_result_array,
            tar_mapping_points,
            target_points,
            max_points_per_target=1  # 各ターゲット頂点に対して1つの対応点のみ
        )

        # Convert results to numpy arrays
        if len(self.correspondence_points) == 0:
            # If advanced correspondence search fails, try simple skeleton-based method
            print("Advanced correspondence search failed. Trying simple skeleton-based method...")
            self._find_correspondence_using_skeleton(sample_rate, sample_number, sample_degree, weight_decay)

            if len(self.correspondence_points) == 0:
                raise ValueError("No correspondence points found. Check mesh connectivity and skeleton binding.")
        else:
            print(f"Found {len(self.correspondence_points)} correspondence points with advanced method.")

        # Extract point arrays from correspondence points
        source_points = np.array([cp.source_position for cp in self.correspondence_points])
        target_points = np.array([cp.target_position for cp in self.correspondence_points])

        # Restore original coordinates if alignment was used
        if align_spaces and transform_matrix is not None and original_joint_positions is not None:
            print("Restoring source bone positions to original space...")

            # Calculate inverse transform matrix
            # inverse_transform = np.linalg.inv(transform_matrix)

            # First restore original joint positions
            for i, joint in enumerate(src_joint_group):
                joint.position = original_joint_positions[i]
                pos_array = np.array(joint.position).squeeze()
                joint.position = pos
                cmds.xform(joint.path.fullPathName(), ws=True, t=pos_array)

            # # Then transform the correspondence points back to original space
            # for i, cp in enumerate(self.correspondence_points):
            #     # Convert to homogeneous coordinates
            #     homogeneous_point = np.append(cp.source_position, 1.0)
            #     # Apply inverse transform
            #     original_position = np.dot(homogeneous_point, inverse_transform.T)[:3]
            #     # Update the correspondence point
            #     self.correspondence_points[i].source_position = original_position
            # 
            # # Update source_points array with original coordinates
            # source_points = np.array([cp.source_position for cp in self.correspondence_points])

            print("Source points and bones restored to original space.")

        print("Correspondence search completed.")
        print(f"Source points: {source_points.shape}, Target points: {target_points.shape}")
        # source_points = self.source_mesh.get_points()[src_hit_points_indices]

        return source_points, target_points

    @util.timeit
    def get_correspondence_points(
            self,
            tar_mapping_points,
            src_triangles,
            src_triangle_indices,
            sample_number: int,
            sample_degree: float,
            src_joint_group=None,
            tar_joint_group=None,
            src_bone_group=None,
            tar_bone_group=None,
            batch_size: int = 1024,  # レイのバッチ処理サイズ
            max_triangles: int = 0   # 0=制限なし、それ以外は三角形の制限
    ):
        """Intel Embreeを使用した超高速レイキャストによる対応点探索
        
        embreeライブラリを使用して、複数のレイを一度に処理するバッチ処理アプローチで
        メッシュ間の対応点を高速に検索します。
        
        Args:
            tar_mapping_points (list): ターゲットメッシュのマッピング点情報
            src_triangles (np.ndarray): ソースメッシュの頂点座標
            src_triangle_indices (np.ndarray): ソースメッシュの三角形インデックス
            sample_number (int): サンプリングレイの数
            sample_degree (float): サンプリングの角度範囲（度）
            src_joint_group (list, optional): ソースジョイントグループ
            tar_joint_group (list, optional): ターゲットジョイントグループ
            src_bone_group (list, optional): ソース骨グループ
            tar_bone_group (list, optional): ターゲット骨グループ
            batch_size (int, optional): レイのバッチ処理サイズ
            max_triangles (int, optional): 処理する三角形の最大数（0=制限なし）
            
        Returns:
            list: レイキャスト結果の配列
        """
        
        # 入力データをNumPy配列に変換
        src_triangles_np = np.asarray(src_triangles, dtype=np.float32)
        src_triangle_indices_np = np.asarray(src_triangle_indices, dtype=np.int32)
        
        # Get joint information from source and target meshes if not provided
        if src_joint_group is None or tar_joint_group is None or src_bone_group is None or tar_bone_group is None:
            _source_weights, source_joints = self._get_skin_weights(self.source_mesh)
            _target_weights, target_joints = self._get_skin_weights(self.target_mesh)
    
            # Get joint information
            source_joint_paths, src_joint_group, src_bone_group = self._get_joint_tree(source_joints)
            target_joint_paths, tar_joint_group, tar_bone_group = self._get_joint_tree(target_joints)
    
        # Create joint mapping from target to source
        src_joint_index = self._match_joint_trees(tar_joint_group, src_joint_group)
    
        # Get number of source triangles
        src_num_triangles = len(src_triangle_indices) // 3
        
        # 処理する三角形数を制限（オプション）
        if max_triangles > 0 and max_triangles < src_num_triangles:
            src_num_triangles = max_triangles
        
        # プログレスバーの設定
        bar = mel.eval("$tmp = $gMainProgressBar")
        if not cmds.about(batch=True):
            cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Setting up Embree acceleration structure...",
                maxValue=100
            )
            cmds.progressBar(bar, edit=True, step=10)
        
        # Embree用の三角形形式に変換
        # 三角形配列を形成: shape=(src_num_triangles, 3, 3)
        embree_triangles = np.zeros((src_num_triangles, 3, 3), dtype=np.float32)
        
        for i in range(src_num_triangles):
            v0_idx = src_triangle_indices_np[i * 3 + 0]
            v1_idx = src_triangle_indices_np[i * 3 + 1]
            v2_idx = src_triangle_indices_np[i * 3 + 2]
            
            embree_triangles[i, 0] = src_triangles_np[v0_idx]
            embree_triangles[i, 1] = src_triangles_np[v1_idx]
            embree_triangles[i, 2] = src_triangles_np[v2_idx]
        
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=40)
        
        # Embreeシーンとメッシュの作成
        scene = rtcs.EmbreeScene()
        _mesh = TriangleMesh(scene, embree_triangles)
        
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=50)
            cmds.progressBar(bar, edit=True, endProgress=True)
            
            # 新しいプログレスバーを設定（対応点計算用）
            cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Calculating correspondence points with Embree...",
                maxValue=len(tar_mapping_points)
            )
        
        # 結果を格納する配列
        raycast_result_array = [[] for _ in range(len(tar_mapping_points))]
        
        # レイの一時格納用バッファ
        ray_origins = np.zeros((batch_size, 3), dtype=np.float32)
        ray_directions = np.zeros((batch_size, 3), dtype=np.float32)
        ray_data = np.zeros(batch_size, dtype=[
            ("vertex_idx", np.int32),
            ("from_point", np.float32, (3,)),
            ("node_weight", np.float32),
            ("target_distance", np.float32)
        ])
        
        # 各ターゲット頂点を処理
        for current_vert, mapping_result in enumerate(tar_mapping_points):
            if not cmds.about(batch=True):
                cmds.progressBar(bar, edit=True, step=1)
            
            # マッピングポイントが空の場合はスキップ
            if not mapping_result.node_array:
                continue
            
            target_vertex_idx = mapping_result.vertex_index
            # ターゲット頂点の位置を取得
            if hasattr(self, "target_mesh") and self.target_mesh is not None:
                target_vertex_pos = self.target_mesh.get_points()[target_vertex_idx]
            else:
                target_vertex_pos = mapping_result.target_position if hasattr(mapping_result, "target_position") else target_points[target_vertex_idx]
            
            # 各マッピングポイントを処理
            for current_node in mapping_result.node_array:
                # 現在の骨インデックスとウェイト
                current_tar_bone_index = current_node.bone_index
                current_node_weight = current_node.weight
                
                # ウェイトが非常に小さい場合はスキップ
                if current_node_weight < 0.001:
                    continue
                
                # マッピングポイントから頂点へのベクトル
                current_tar_p = current_node.point
                current_tar_pv = target_vertex_pos - current_tar_p
                # 方向ベクトルを正規化
                current_tar_pv_norm = np.linalg.norm(current_tar_pv)
                if current_tar_pv_norm < 1e-10:
                    continue
                current_tar_normal_pv = current_tar_pv / current_tar_pv_norm
                
                # ターゲット骨情報
                tar_start_joint_index = tar_bone_group[current_tar_bone_index].start_joint_index
                tar_end_joint_index = tar_bone_group[current_tar_bone_index].end_joint_index
                
                current_tar_bone_start_point = tar_joint_group[tar_start_joint_index].position
                current_tar_bone_end_point = tar_joint_group[tar_end_joint_index].position
                current_tar_bone_v = current_tar_bone_end_point - current_tar_bone_start_point
                
                # 骨に沿った距離比率を計算
                current_tar_bone_v_norm = np.linalg.norm(current_tar_bone_v)
                if current_tar_bone_v_norm < 1e-10:
                    continue
                tar_distance = np.linalg.norm(current_tar_p - current_tar_bone_start_point) / current_tar_bone_v_norm
                
                # ソース骨情報
                src_start_joint_index = src_joint_index[tar_start_joint_index]
                src_end_joint_index = src_joint_index[tar_end_joint_index]
                
                if src_start_joint_index == -1 or src_end_joint_index == -1:
                    continue
                
                current_src_bone_start_point = src_joint_group[src_start_joint_index].position
                current_src_bone_end_point = src_joint_group[src_end_joint_index].position
                current_src_bone_v = current_src_bone_end_point - current_src_bone_start_point
                
                # ソース側の対応する点（骨上の比率を使用）
                p = current_src_bone_start_point + current_src_bone_v * tar_distance
                p = np.array(p).squeeze()
                
                # 方向ベクトル
                d = np.array(current_tar_normal_pv).squeeze()
                
                # サンプル方向を生成
                sample_directions = rand_cone_vector(d, sample_degree, sample_number)
                
                # Embreeを使用してバッチレイキャスト
                # レイをバッチで処理
                n_rays = len(sample_directions)
                n_batches = (n_rays + batch_size - 1) // batch_size  # 切り上げ除算
                
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, n_rays)
                    current_batch_size = end_idx - start_idx
                    
                    # バッチデータを準備
                    ray_origins[:current_batch_size] = p
                    ray_directions[:current_batch_size] = sample_directions[start_idx:end_idx]
                    ray_data["vertex_idx"][:current_batch_size] = current_vert
                    ray_data["from_point"][:current_batch_size] = p
                    ray_data["node_weight"][:current_batch_size] = current_node_weight
                    ray_data["target_distance"][:current_batch_size] = current_tar_pv_norm
                    
                    # Embreeでレイキャスト実行
                    res = scene.run(ray_origins[:current_batch_size], ray_directions[:current_batch_size], output=1)
                    
                    # ヒットしたレイを処理
                    hit_mask = res["geomID"] >= 0
                    if np.any(hit_mask):
                        # ヒットデータの抽出
                        hit_indices = np.where(hit_mask)[0]
                        primIDs = res["primID"][hit_mask]
                        ts = res["tfar"][hit_mask]
                        us = res["u"][hit_mask]
                        vs = res["v"][hit_mask]
                        
                        # 各ヒットに対して結果を生成
                        for i, hit_idx in enumerate(hit_indices):
                            primID = primIDs[i]
                            t = ts[i]
                            u = us[i]
                            v = vs[i]
                            w = 1.0 - u - v
                            
                            # 三角形の頂点座標
                            v0 = embree_triangles[primID, 0]
                            v1 = embree_triangles[primID, 1]
                            v2 = embree_triangles[primID, 2]
                            
                            # 交点座標を計算（バリセントリック座標）
                            intersection_point = w * v0 + u * v1 + v * v2
                            
                            # 元のデータを取得
                            idx = hit_idx
                            vertex_idx = ray_data["vertex_idx"][idx]
                            from_point = ray_data["from_point"][idx]
                            node_weight = ray_data["node_weight"][idx]
                            target_distance = ray_data["target_distance"][idx]
                            
                            # 結果ノードの作成
                            result_node = {
                                "from_point": from_point,
                                "point": intersection_point,
                                "triangle_index": int(primID),
                                "weight": float(node_weight),
                                "relate_distance": float(target_distance / t)
                            }
                            
                            raycast_result_array[vertex_idx].append(result_node)
        
        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, endProgress=True)
        
        return raycast_result_array

    def create_optimized_correspondence_points(
            self,
            raycast_result_array,
            tar_mapping_points,
            target_points,
            max_points_per_target: int = 1,  # 各ターゲット頂点に対する最大対応点数
            min_weight_threshold: float = 0.01,  # 最小ウェイト閾値
            distance_weight: float = 1.0,     # 距離の重要度
            ray_weight: float = 0.5           # レイ情報の重要度
    ):
        """レイキャスト結果から最適化された対応点を生成する
        
        各ターゲット頂点に対して最も重要な対応点のみを保持し、総数を削減します。
        
        Args:
            raycast_result_array (list): レイキャスト結果の配列
            tar_mapping_points (list): ターゲットマッピングポイント情報
            target_points (np.ndarray): ターゲット頂点の座標
            max_points_per_target (int): 各ターゲット頂点に対する最大対応点数
            min_weight_threshold (float): 最小ウェイト閾値（これ未満の対応点は無視）
            distance_weight (float): 距離スコアの重み係数
            ray_weight (float): レイ情報スコアの重み係数
            
        Returns:
            list: 最適化された対応点リスト
        """
        # 対応点を格納する辞書（キー：ターゲット頂点インデックス）
        correspondence_dict = {}
        
        # レイキャスト結果を処理
        for i, raycast_results in enumerate(raycast_result_array):
            if not raycast_results:
                continue
                
            target_idx = tar_mapping_points[i].vertex_index
            target_pos = target_points[target_idx]
            
            # このターゲット頂点用の候補リスト（スコア付き）
            candidates = []
            
            for raycast in raycast_results:
                # 三角形インデックスの検証
                triangle_idx = raycast["triangle_index"]
                if triangle_idx < 0:
                    continue
                    
                # 交点座標
                src_pos = raycast["point"]
                
                # 基本的な距離とウェイト
                distance = np.linalg.norm(src_pos - target_pos)
                basic_weight = 1.0 / (1.0 + distance)
                
                # レイの関連情報を使用して品質スコアを計算
                ray_quality = 1.0
                if "relate_distance" in raycast:
                    # relate_distanceが小さいほど良い
                    ray_quality = 1.0 / (1.0 + raycast["relate_distance"])
                
                # ノードウェイトも考慮
                node_weight = raycast.get("weight", 1.0)
                
                # 総合スコアの計算
                # 距離ベースのスコア、レイ品質、ノードウェイトを考慮
                total_score = (
                    distance_weight * basic_weight +  # 距離ベースのスコア
                    ray_weight * ray_quality * node_weight  # レイ品質とノードウェイト
                ) / (distance_weight + ray_weight)  # 正規化
                
                # 最小閾値を超える場合のみ候補として追加
                if total_score >= min_weight_threshold:
                    candidates.append({
                        "source_position": src_pos,
                        "target_position": target_pos,
                        "target_index": target_idx,
                        "weight": basic_weight,  # 元のウェイト計算方法を維持
                        "score": total_score,    # ソート用の総合スコア
                        "triangle_index": triangle_idx
                    })
            
            # スコアに基づいて候補をソート（降順）
            candidates.sort(key=lambda x: x["score"], reverse=True)
            
            # 上位N個の候補を保持
            top_candidates = candidates[:max_points_per_target]
            
            # 最終的な対応点を辞書に追加
            if target_idx not in correspondence_dict:
                correspondence_dict[target_idx] = []
                
            correspondence_dict[target_idx].extend(top_candidates)
        
        # 最終的な対応点リストを作成
        optimized_correspondence_points = []
        
        for target_idx, candidates in correspondence_dict.items():
            # 各候補から対応点オブジェクトを作成
            for candidate in candidates:
                correspondence_point = CorrespondencePoint(
                    source_index=-1,  # 正確なインデックスの代わりに座標を使用
                    target_index=candidate["target_index"],
                    source_position=candidate["source_position"],
                    target_position=candidate["target_position"],
                    weight=candidate["weight"]
                )
                optimized_correspondence_points.append(correspondence_point)
        
        return optimized_correspondence_points

    def weight_transform(
            self,
            src_default_points,
            tar_joints_retarget,
            src_default_weight,
            src_deform_weight,
            tar_weight,
            src_triangles,
            tar_mapping_points,
            raycast_result_array,
            weight_decay,
    ):
        """対応点に基づいてウェイトを変換する

        レイキャスト結果に基づいて、ソースメッシュからターゲットメッシュへのウェイト変換を行います。

        Args:
            src_default_points (np.ndarray): ソースメッシュのデフォルト頂点座標
            tar_joints_retarget (list): ターゲットからソースへのジョイントリターゲット情報
            src_default_weight (list): ソースメッシュのデフォルトウェイト
            src_deform_weight (list): ソースメッシュの変形後ウェイト
            tar_weight (list): ターゲットメッシュのウェイト
            src_triangles (np.ndarray): ソースメッシュの三角形インデックス
            tar_mapping_points (list): ターゲットメッシュのマッピングポイント情報
            raycast_result_array (list): レイキャスト結果の配列
            weight_decay (float): 重み減衰係数

        Returns:
            list: 結果ウェイト
        """
        # 結果ウェイトの初期化
        inf_number = len(src_deform_weight[0])
        result_weight = [[0.0 for _ in range(inf_number)] for _ in range(len(tar_weight))]

        # 各ターゲット頂点に対して処理
        for current_vert, mapping_result in enumerate(tar_mapping_points):
            # 現在の頂点インデックスとウェイト
            tar_index = mapping_result.vertex_index
            sum_weight = 0.0
            current_weight = [0.0] * inf_number

            # 各レイキャスト結果に対して処理
            for raycast_node in raycast_result_array[current_vert]:
                triangle_index = raycast_node["triangle_index"]

                # 三角形の3頂点のインデックス
                src_p0_index = src_triangles[triangle_index * 3 + 0]
                src_p1_index = src_triangles[triangle_index * 3 + 1]
                src_p2_index = src_triangles[triangle_index * 3 + 2]

                # 交差点
                best_q = raycast_node["point"]
                weight = raycast_node["weight"]

                # 三角形の頂点
                default_triangle = [
                    src_default_points[src_p0_index],
                    src_default_points[src_p1_index],
                    src_default_points[src_p2_index]
                ]

                # 重心座標の計算
                qtop0, qtop1, qtop2 = triangle_interpolation(
                    default_triangle[0], default_triangle[1], default_triangle[2], best_q
                )

                # ウェイトマスクの計算
                match_count = 0
                match_weight = 0.0

                for inf_index, src_idx in enumerate(tar_joints_retarget):
                    if src_idx == -1:
                        # このジョイントはソースに存在しない
                        continue

                    tar_w = tar_weight[current_vert][inf_index]
                    src_df_w0 = src_default_weight[src_p0_index][src_idx]
                    src_df_w1 = src_default_weight[src_p1_index][src_idx]
                    src_df_w2 = src_default_weight[src_p2_index][src_idx]

                    # 重心座標を使用した補間
                    src_df_qw = src_df_w0 * qtop0 + src_df_w1 * qtop1 + src_df_w2 * qtop2

                    # 平均
                    if tar_w + src_df_qw > 0:
                        match_weight += abs(tar_w - src_df_qw) / (tar_w + src_df_qw)
                        match_count += 1

                # 正規化 + 指数減衰
                match_weight /= max(1, match_count)
                match_weight = (1.0 - match_weight)
                match_weight = match_weight ** weight_decay
                match_weight *= weight

                # ウェイトの計算
                for inf_index in range(inf_number):
                    current_weight[inf_index] += src_deform_weight[src_p0_index][inf_index] * qtop0 * match_weight
                    current_weight[inf_index] += src_deform_weight[src_p1_index][inf_index] * qtop1 * match_weight
                    current_weight[inf_index] += src_deform_weight[src_p2_index][inf_index] * qtop2 * match_weight

            # 合計ウェイトの計算
            sum_weight = sum(current_weight)

            # 正規化
            if sum_weight > 0:
                for inf_index in range(inf_number):
                    result_weight[tar_index][inf_index] = current_weight[inf_index] / sum_weight

        return result_weight

    @util.timeit
    def _find_correspondence_using_skeleton(
            self, 
            sample_rate: float,
            sample_number: int, 
            sample_degree: float,
            weight_decay: float) -> None:
        """スケルトン（骨格）情報を使用して対応点を検索

        Args:
            sample_rate (float): サンプリングする頂点の割合 (0.0-1.0)
            sample_number (int): サンプリングレイの数
            sample_degree (float): サンプリング時の角度範囲（度）
            weight_decay (float): 重み減衰係数
        """
        # メッシュからの情報取得
        source_points = self.source_mesh.get_points()
        target_points = self.target_mesh.get_points()

        # スキニングウェイト情報の取得
        source_weights, source_joints = self._get_skin_weights(self.source_mesh)
        target_weights, target_joints = self._get_skin_weights(self.target_mesh)

        # ジョイント名をキーにしたマッピングを作成
        source_joint_map = {j.split(":")[-1]: i for i, j in enumerate(source_joints)}
        _target_joint_map = {j.split(":")[-1]: i for i, j in enumerate(target_joints)}

        # サンプリングのために頂点数を削減
        if sample_rate < 1.0:
            num_samples = max(10, int(len(target_points) * sample_rate))
            sample_indices = np.linspace(0, len(target_points) - 1, num_samples).astype(int)
        else:
            sample_indices = range(len(target_points))

        # ターゲットメッシュの各頂点について対応点を検索
        self.correspondence_points = []

        for idx in sample_indices:
            target_pos = target_points[idx]

            # この頂点に影響を与えるジョイントを検索
            influential_joints = []
            for joint_idx, weight in enumerate(target_weights[idx]):
                if weight > 0.01:  # 一定以上の影響力を持つジョイントのみ考慮
                    influential_joints.append((joint_idx, weight))

            # 影響力が最も大きいジョイントを使用
            influential_joints.sort(key=lambda x: x[1], reverse=True)

            best_match = None
            min_distance = float("inf")

            # 適切な対応点を見つける
            for joint_idx, weight in influential_joints:
                target_joint_name = target_joints[joint_idx].split(":")[-1]

                # ソースメッシュに同じジョイントが存在するか確認
                if target_joint_name in source_joint_map:
                    source_joint_idx = source_joint_map[target_joint_name]

                    # このジョイントの影響を受けるソースメッシュの頂点を探す
                    candidates = []
                    for src_idx, src_weights in enumerate(source_weights):
                        src_weight = src_weights[source_joint_idx]
                        if src_weight > 0.01:
                            candidates.append((src_idx, src_weight))

                    # 最も近い候補を見つける
                    for src_idx, src_weight in candidates:
                        src_pos = source_points[src_idx]
                        distance = np.linalg.norm(src_pos - target_pos)

                        # 重みによる距離の調整
                        adjusted_distance = distance / (src_weight * weight)

                        if adjusted_distance < min_distance:
                            min_distance = adjusted_distance
                            best_match = (src_idx, src_pos)

            # 一致する点が見つかった場合、対応点リストに追加
            if best_match:
                src_idx, src_pos = best_match
                self.correspondence_points.append(CorrespondencePoint(
                    source_index=src_idx,
                    target_index=idx,
                    source_position=src_pos,
                    target_position=target_pos,
                    weight=1.0 / (1.0 + min_distance)  # 距離に基づく重み付け
                ))

    def _get_joint_tree(self, joint_names):
        # type: (list[str]) -> tuple[list[om.MDagPath], list[JointNode], list[BoneNode]]
        """ジョイントツリーを取得

        Args:
            joint_names (list): ジョイント名のリスト

        Returns:
            tuple: (joint_paths, joint_group, bone_group)
        """
        # ジョイントの DAG パスを取得
        joint_paths = []
        for joint_name in joint_names:
            # ジョイントの存在確認
            if not cmds.objExists(joint_name):
                continue

            # DAG パスの取得
            selection = om.MSelectionList()
            selection.add(joint_name)
            dag_path = selection.getDagPath(0)
            joint_paths.append(dag_path)

        # ジョイントツリーの構築
        joint_group = []
        bone_group = []

        # ルートジョイントを見つける（ルートは親がないジョイント）
        root_joints = []
        for i, path in enumerate(joint_paths):
            if path.length() == 1 or cmds.listRelatives(path.fullPathName(), parent=True, type="joint") is None:
                root_joints.append(path)

        if not root_joints:
            # パスに含まれる "|" の数が最も少ないジョイントをルートとして扱う (階層が浅い)
            min_depth = min(path.fullPathName().count("|") for path in joint_paths)
            root_joints = [path for path in joint_paths if path.fullPathName().count("|") == min_depth]

        # 各ルートジョイントからツリーを構築
        for root_path in root_joints:
            queue = [root_path]
            visited = set()

            while queue:
                current_path = queue.pop(0)
                if current_path.fullPathName() in visited:
                    continue

                visited.add(current_path.fullPathName())

                # ジョイントノードの作成
                name = current_path.fullPathName().split("|")[-1].split(":")[-1].split("|")[-1]
                pos = cmds.xform(current_path.fullPathName(), query=True, translation=True, worldSpace=True)
                position = np.array(pos, dtype=np.float64)

                joint_node = JointNode(
                    path=current_path,
                    index=len(joint_group),
                    detail_name=name,
                    position=position
                )
                joint_group.append(joint_node)

                # 子ジョイントを取得
                children = cmds.listRelatives(current_path.fullPathName(), children=True, type="joint", fullPath=True) or []

                for child in children:
                    # helper ジョイントをスキップ
                    if "helper" in child:
                        continue

                    child_sel = om.MSelectionList()
                    child_sel.add(child)
                    child_path = child_sel.getDagPath(0)

                    # ボーンノードの作成
                    bone_node = BoneNode(
                        start_joint_index=joint_node.index,
                        end_joint_index=len(joint_group)  # 追加される予定の子ジョイントのインデックス
                    )
                    bone_group.append(bone_node)

                    queue.append(child_path)

        return joint_paths, joint_group, bone_group

    def _match_joint_trees(self, tar_joint_group, src_joint_group):
        """Match joint trees

        Args:
            tar_joint_group (list): Target joint group
            src_joint_group (list): Source joint group

        Returns:
            list: Source joint indices
        """
        # Create joint mapping from target to source
        src_joint_index = [-1] * len(tar_joint_group)

        for i, tar_joint in enumerate(tar_joint_group):
            tar_short_name = tar_joint.detail_name.split(":")[-1].split("|")[-1]

            for j, src_joint in enumerate(src_joint_group):
                src_short_name = src_joint.detail_name.split(":")[-1].split("|")[-1]

                if src_short_name == tar_short_name:
                    src_joint_index[i] = j
                    break

        return src_joint_index

    @util.timeit
    def _calculate_alignment_transform_rbf(
        self,
        src_points: np.ndarray,
        tar_points: np.ndarray,
        kernel=RBF.linear,
        radius=1.0
    ):
        """Calculate alignment transform using RBF.

        Use RBF to calculate the alignment transform of source space to target space.

        Args:
            src_points (np.ndarray): Source points (M, 3)
            tar_points (np.ndarray): Target points (M, 3)
            kernel (RBF): Radial basis function kernel
            radius (float): RBF radius

        Returns:
            function: RBF transform function
        """

        # 対応が取れた頂点同士で RBF の重みを計算する
        #   tar座標(M×3) を src座標(M×3) に写すための RBF を学習するイメージ
        weights = calculate_rbf_weight_matrix(
            source_points=src_points,
            target_points=tar_points,
            kernel=kernel,
            radius=radius
        )

        def rbf_transform(query_points: np.ndarray) -> np.ndarray:
            """
            ターゲット空間の座標群を RBF でソース側に寄せる。

            引数:
                query_points: shape=(K,3) の座標群
            戻り値:
                shape=(K,3) の座標群
            """
            # 距離行列 (K,M) を作る
            dist_mat = get_distance_matrix(query_points, tar_points, kernel, radius)

            # RBF の式に則り、[dist_mat, 1, query_points] を行列で組み立てて weights と乗算
            K = query_points.shape[0]
            ones = np.ones((K, 1), dtype=np.float64)
            # dist_mat: (K,M), ones: (K,1), query_points: (K,3) => 結果的に (K, M + 1 + 3)
            h_combined = np.hstack([dist_mat, ones, query_points])

            # weights は (M + 1 + 3, 3) なので掛け合わせると (K,3)
            deformed = h_combined @ weights
            return deformed

        return rbf_transform

    @util.timeit
    def _calculate_alignment_transform(
        self,
        src_joint_group,
        tar_joint_group,
        kernel=RBF.linear,
        radius=1.0
    ):
        """Calculate alignment transform using RBF."""

        # (1) 全ジョイントの座標を取得
        src_points = np.array([joint.position for joint in src_joint_group])  # (Ns,3)
        tar_points = np.array([joint.position for joint in tar_joint_group])  # (Nt,3)

        # (2) 名前一致を探す -> matched_src_indices, matched_tar_indices
        src_matched_flags = np.zeros(len(src_joint_group), dtype=bool)
        tar_matched_flags = np.zeros(len(tar_joint_group), dtype=bool)

        for i, tar_joint in enumerate(tar_joint_group):
            tar_name = tar_joint.detail_name  # 例: "joint1"
            for j, src_joint in enumerate(src_joint_group):
                src_name = src_joint.detail_name
                if src_name == tar_name:
                    tar_matched_flags[i] = True
                    src_matched_flags[j] = True
                    break

        matched_src_indices = np.where(src_matched_flags)[0]
        matched_tar_indices = np.where(tar_matched_flags)[0]

        # (3) マッチするジョイントが3未満なら RBF 不可 => None
        if len(matched_src_indices) < 3:
            print("Not enough matched joints to build RBF. Skipping alignment.")
            return None

        # (4) RBF 変換関数を作る (ソース->ターゲット)
        matched_src_points = src_points[matched_src_indices]  # shape=(M,3)
        matched_tar_points = tar_points[matched_tar_indices]  # shape=(M,3)
        rbf_func = self._calculate_alignment_transform_rbf(
            matched_src_points,
            matched_tar_points,
            kernel=kernel,
            radius=radius
        )

        # (5) ソース全ジョイントを変換
        new_src_points = rbf_func(src_points)  # shape=(Ns,3)

        # (6) matched ジョイントをターゲットの座標に合わせる
        for i, src_id in enumerate(matched_src_indices):
            new_src_points[src_id] = tar_points[matched_tar_indices[i]]

        return new_src_points

    @util.timeit
    def get_mapping_points(
            self,
            target_points,
            target_joint_group,
            target_bone_group,
            target_weights,
            target_joint_names,
            max_distance=0.0
    ):
        """マッピングポイントを取得

        ターゲットメッシュの各頂点から骨格に基づいたマッピングポイントを生成します。

        Args:
            target_points (np.ndarray): ターゲットメッシュの頂点座標
            target_joint_group (list): ターゲットジョイントグループ
            target_bone_group (list): ターゲット骨グループ
            target_weights (list): ターゲットメッシュのウェイト
            target_joint_names (list): ターゲットジョイント名
            max_distance (float): 最大距離（0の場合は制限なし）

        Returns:
            list: マッピング結果の配列
        """
        # ジョイントのウェイトマッピング
        bones_weight_index = [-1] * len(target_joint_group)

        for i, joint in enumerate(target_joint_group):
            for j, joint_name in enumerate(target_joint_names):
                if joint_name.split(":")[-1] == joint.detail_name:
                    bones_weight_index[i] = j
                    break

        # 結果配列を初期化
        mapping_results = []

        # 各頂点に対して処理
        for vert_idx, vertex in enumerate(target_points):
            mapping_result = MappingResult(vertex_index=vert_idx)

            # 各骨に対して処理
            for bone_idx, bone in enumerate(target_bone_group):
                start_joint = target_joint_group[bone.start_joint_index]
                end_joint = target_joint_group[bone.end_joint_index]

                # ウェイトのチェック
                weight_index = bones_weight_index[bone.start_joint_index]
                if weight_index < 0 or target_weights[vert_idx][weight_index] < 1e-5:
                    continue

                # 骨のベクトルを計算
                start_point = start_joint.position
                end_point = end_joint.position
                bone_vector = end_point - start_point
                bone_length = np.linalg.norm(bone_vector)

                if bone_length < 1e-10:
                    continue

                normalize_bone_vector = bone_vector / bone_length

                # 頂点から骨への射影を計算
                w = vertex - start_point
                projection_length = np.dot(w, normalize_bone_vector)
                p = projection_length * normalize_bone_vector + start_point

                # 骨上の投影点が有効かチェック
                check_direction = p - start_point
                left_is_legal = np.dot(check_direction, normalize_bone_vector) > 0
                left_is_legal |= max_distance > np.linalg.norm(check_direction)

                check_direction = p - end_point
                right_is_legal = np.dot(check_direction, normalize_bone_vector) < 0
                right_is_legal |= max_distance > np.linalg.norm(check_direction)

                # 頂点から投影点への距離
                distance = np.linalg.norm(p - vertex)

                if right_is_legal and left_is_legal:
                    # 骨上の有効な投影点
                    mapping_node = MappingNode(
                        point=p,
                        bone_index=bone_idx,
                        distance=distance,
                        weight=target_weights[vert_idx][weight_index]
                    )
                    mapping_result.node_array.append(mapping_node)
                else:
                    # 骨の端点を使用
                    start_distance = np.linalg.norm(start_point - vertex)
                    end_distance = np.linalg.norm(end_point - vertex)

                    if start_distance < end_distance:
                        mapping_node = MappingNode(
                            point=start_point,
                            bone_index=bone_idx,
                            distance=start_distance,
                            weight=target_weights[vert_idx][weight_index]
                        )
                        mapping_result.node_array.append(mapping_node)
                    else:
                        mapping_node = MappingNode(
                            point=end_point,
                            bone_index=bone_idx,
                            distance=end_distance,
                            weight=target_weights[vert_idx][weight_index]
                        )
                        mapping_result.node_array.append(mapping_node)

            mapping_results.append(mapping_result)

        return mapping_results

    @util.timeit
    def get_weight_distance(
        self,
        src_triangles,
        src_num_tri,
        src_weight,         # shape=(num_vertices, num_inf)
        tar_weight,         # shape=(tar_num_vertex, num_inf)
        tar_joints_retarget,
        top_n=100
    ):
        """
        概要:
          - まず "有効なインフルエンス" を抽出
          - ソース三角形の平均ウェイトから (src_num_tri, len(valid_l)) の配列 s_w_valid を作成
          - ターゲット頂点ごとにベクトル演算し、距離を高速に算出 → ソート
    
        Args:
            src_triangles (np.ndarray): ソースメッシュの三角形インデックス, shape=(3*src_num_tri,)
            src_num_tri (int): ソースメッシュの三角形数
            src_weight (list[list[float]]): ソースメッシュの頂点ウェイト [vertex][inf]
            tar_weight (list[list[float]]): ターゲットメッシュの頂点ウェイト [vertex][inf]
            tar_joints_retarget (list[int]): ターゲット→ソースのジョイントインデックス対応 (or -1)
            top_n (int): ウェイト距離の上位何個を取得するか
    
        Returns:
            list[list[TriangleWeightIndex]]:
                tar_best_weight_triangle_map, shape=(tar_num_vertex,),
                各要素が三角形 j 毎の TriangleWeightIndex をソートしたリスト
        """
        # 1) 有効なインフルエンス "valid_l" を抽出
        #    tar_joints_retarget[l] = -1 の要素はスキップ
        num_inf = len(src_weight[0])
        valid_l = [l for l in range(num_inf) if tar_joints_retarget[l] != -1]
    
        # 2) ソース三角形の平均ウェイト (src_triangle_weight) を
        #    (src_num_tri, num_inf) から (src_num_tri, len(valid_l)) へ絞り込む
        #    ただし src_triangle_weight[j][k] で k は "ソースjoint" なので
        #    それを "tar_joints_retarget[l]" で参照する
        src_triangles_np = np.array(src_triangles, dtype=np.int32).reshape(-1, 3)
        src_weight_np = np.array(src_weight, dtype=np.float32)
        tar_weight_np = np.array(tar_weight, dtype=np.float32)
        triangle_weights = src_weight_np[src_triangles_np]  # shape=(src_num_tri,3,num_inf)
        src_triangle_weight = np.mean(triangle_weights, axis=1)  # shape=(src_num_tri, num_inf)
    
        # それを valid_l のみ抜き出す => s_w_valid shape=(src_num_tri, len(valid_l))
        s_w_valid = np.zeros((src_num_tri, len(valid_l)), dtype=np.float32)
        for col, l in enumerate(valid_l):
            # tar_joints_retarget[l] = ソースjointID
            src_jid = tar_joints_retarget[l]
            # ソースjointID が -1 の場合はスキップするが、
            # valid_l には含まれていないのでそのチェック不要
            s_w_valid[:, col] = src_triangle_weight[:, src_jid]
    
        # 3) ターゲット頂点ごとに距離を算出
        tar_num_vertex = len(tar_weight)
        tar_best_weight_triangle_map = []
    
        bar = mel.eval("$tmp = $gMainProgressBar")
        if not cmds.about(batch=True):
            cmds.progressBar(
                    bar,
                    edit=True,
                    beginProgress=True,
                    isInterruptable=False,
                    maxValue=tar_num_vertex,
                    status="Calculating weight-based triangle mapping..."
            )
    
        for i in range(tar_num_vertex):
            t_w_valid = tar_weight_np[i, valid_l]
    
            # ベクトル演算で (src_num_tri,) 個の距離をまとめて計算
            # diff shape=(src_num_tri, len(valid_l))
            diff = s_w_valid - t_w_valid
            dist2 = np.sum(diff**2, axis=1)      # 二乗和 => shape=(src_num_tri,)
            dist = np.sqrt(dist2)                # shape=(src_num_tri,)
    
            # dist をソートして TriangleWeightIndex 化
            idx_sorted = np.argsort(dist)
            if top_n > 0:
                idx_sorted = idx_sorted[:top_n]

            triangle_weight_indices = []
            for j in idx_sorted:
                twi = TriangleWeightIndex(
                    weight_distance=dist[j],
                    triangle_index=j
                )
                triangle_weight_indices.append(twi)

            tar_best_weight_triangle_map.append(triangle_weight_indices)

            if not cmds.about(batch=True):
                cmds.progressBar(bar, edit=True, step=1)

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, endProgress=True)

        return tar_best_weight_triangle_map

    @util.timeit
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

        # インフルエンスインデックス作成
        influence_indices = om.MIntArray([i for i in range(len(influence_objects))])

        # スキンウェイト取得
        weights = fn_skin.getWeights(mesh_obj.dag_path, vert_component, influence_indices)

        # リスト形式に変換
        weights_list = []
        for i in range(num_vertices):
            vertex_weights = []
            for j in range(num_influences):
                weight = weights[i * num_influences + j]
                vertex_weights.append(weight)
            weights_list.append(vertex_weights)

        return weights_list, joint_names

    def _find_skin_cluster(self, mesh_path: om.MDagPath) -> oma.MFnSkinCluster:
        """Find skin cluster for mesh

        Args:
            mesh_path (om.MDagPath): Mesh DAG path

        Returns:
            om.MObject: Skin cluster object (None if not found)
        """
        return util.get_skin_cluster(mesh_path)

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

        # 線を表示するためのグループノード作成
        group_name = cmds.group(empty=True, name="correspondence_visualization")

        # 対応点ごとに線を描画
        for i, cp in enumerate(self.correspondence_points):
            # 線の色を重みに応じて決定（赤-黄-緑）
            color = [1, min(cp.weight * 2, 1), 0]  # 重みが高いほど黄色に

            src_x = float(cp.source_position[0])
            src_y = float(cp.source_position[1])
            src_z = float(cp.source_position[2])

            tar_x = float(cp.target_position[0])
            tar_y = float(cp.target_position[1])
            tar_z = float(cp.target_position[2])

            # 線を描画
            line_name = f"corr_line_{i}"
            curve = cmds.curve(
                degree=1,
                point=[
                    (src_x, src_y, src_z),
                    (tar_x, tar_y, tar_z)
                ],
                name=line_name
            )

            # 線の色と太さを設定
            cmds.setAttr(f"{curve}.overrideEnabled", 1)
            cmds.setAttr(f"{curve}.overrideRGBColors", 1)
            cmds.setAttr(f"{curve}.overrideColorRGB", color[0], color[1], color[2])
            cmds.setAttr(f"{curve}.lineWidth", line_thickness)

            # グループに追加
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
        Tuple[np.ndarray, np.ndarray]: Coordinates of correspondence point pairs (source_points, target_points)
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
