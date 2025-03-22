# -*- coding: utf-8 -*-
"""
Raycast utilities for mesh registration.

This module provides functions for raycasting operations using Embree or fallback methods.
"""
import typing
import numpy as np
from maya import cmds, mel
try:
    from embreex import rtcore_scene as rtcs  # type: ignore
    from embreex.mesh_construction import TriangleMesh  # type: ignore
    EMBREE_AVAILABLE = True
except ImportError:
    EMBREE_AVAILABLE = False
    cmds.warning("embreex library not found. Using standard raycasting instead.")


from . import geometry
from .core import (
    RaycastResult,
)
from .mapping import (
    get_matched_info
)

if typing.TYPE_CHECKING:
    from ..objects import (
        MeshObject
    )
    from .core import (
        MappingResult,
        JointNode,
        BoneNode,
    )


def build_embree_scene_from_source(src_triangles):
    """
    Build an Embree scene from source mesh triangle array.
    
    Args:
        src_triangles (np.ndarray): shape = (num_tri, 3, 3)
            Number of triangles = num_tri
            Each triangle has 3 vertices, each vertex has xyz (3D) coordinates

    Returns:
        tuple: (EmbreeScene, TriangleMesh) - scene and mesh
    """
    if not EMBREE_AVAILABLE:
        raise ImportError("Embree library is not available. Cannot build Embree scene.")
    
    scene = rtcs.EmbreeScene()
    mesh = TriangleMesh(scene, src_triangles)  # This builds the BVH
    return scene, mesh


def perform_raycast(
        src_mesh: "MeshObject",
        tar_mesh: "MeshObject",
        tar_mapping_points: list["MappingResult"],
        src_triangles: np.ndarray,  # (num_verts, 3)
        src_triangle_indices: np.ndarray,  # (num_tri * 3)
        sample_number: int,
        sample_degree: float,
        src_joint_group: list["JointNode"],
        tar_joint_group: list["JointNode"],
        src_bone_group: list["BoneNode"],
        tar_bone_group: list["BoneNode"],
        batch_size: int = 1024,
        max_triangles: int = -1,
) -> list[list[RaycastResult]]:
    """
    Perform raycasting to find correspondence points between meshes.
    
    This is a high-level function that chooses between Embree-based raycasting
    or standard raycasting based on availability.
    
    Args:
        
    Returns:
        list: Raycast result array
    """
    if not EMBREE_AVAILABLE:
        raise ImportError("Embree library is not available. Cannot perform raycasting.")

    # Create joint mapping from target to source
    src_indices, _, _ = get_matched_info(src_joint_group, tar_joint_group)
        
    # 入力データをNumPy配列に変換
    src_triangles_np = np.asarray(src_triangles, dtype=np.float32)
    src_triangle_indices_np = np.asarray(src_triangle_indices, dtype=np.int32)
    
    # Get joint information from source and target meshes if not provided
    # if src_joint_group is None or tar_joint_group is None or src_bone_group is None or tar_bone_group is None:
    #     _source_weights, source_joints = self._get_skin_weights(self.source_mesh)
    #     _target_weights, target_joints = self._get_skin_weights(self.target_mesh)
    # 
    #     # Get joint information
    #     source_joint_paths, src_joint_group, src_bone_group = self._get_joint_tree(source_joints)
    #     target_joint_paths, tar_joint_group, tar_bone_group = self._get_joint_tree(target_joints)
    
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
        target_vertex_pos = tar_mesh.get_points()[target_vertex_idx]
        
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
            src_start_joint_index = src_indices[tar_start_joint_index]
            src_end_joint_index = src_indices[tar_end_joint_index]
            
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
            sample_directions = geometry.rand_cone_vector(d, sample_degree, sample_number)
            
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
                        result_node = RaycastResult(
                            from_point=from_point,
                            point=intersection_point,
                            triangle_index=int(primID),
                            weight=float(node_weight),
                            relate_distance=float(target_distance / t)
                        )
                        
                        raycast_result_array[vertex_idx].append(result_node)
    
    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)
    
    return raycast_result_array
