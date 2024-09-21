# -*- coding: utf-8 -*-
"""
Module for clustering vertices based on skin weights and topological adjacency.
"""
import time
import collections

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import (
    lil_matrix,
)

from maya.api import (
    OpenMaya as om,
)
from maya import (
    cmds,
    mel,
)

from . import util


##############################################################################
# Cluster vertices across multiple meshes
##############################################################################
@util.timeit
def cluster_vertices_by_skin_weight(mesh_paths, precision=3, min_vertices_per_cluster=6):
    # type: (List[om.MDagPath|str], float, int) -> np.ndarray
    """
    Cluster vertices based on their weight similarity across multiple meshes using a custom distance function.

    :param mesh_paths: A list of meshes to cluster
    :param precision: The number of decimal places to round the weights to (default 3)
    :param min_vertices_per_cluster: The minimum number of vertices required to form a cluster (default 6)
    :return: A list of cluster labels for each vertex across all meshes
    """

    # Initialize variables to hold combined data
    all_sparse_weights = []
    vertex_offsets = []
    vertex_offset = 0
    bar = mel.eval("$tmp = $gMainProgressBar")
    start_time = time.time()

    # Process each mesh and collect sparse weights
    for mesh_path in mesh_paths:
        try:
            sparse_weights = util.get_skin_weight_as_sparse_matrix(mesh_path)
        except ValueError:
            mesh = util.get_mesh_fn(mesh_path)
            num_vertices = mesh.numVertices
            sparse_weights = lil_matrix((num_vertices, 0))  # Empty sparse matrix
        num_vertices = sparse_weights.shape[0]
        all_sparse_weights.append(sparse_weights)
        vertex_offsets.append(vertex_offset)
        vertex_offset += num_vertices

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=num_vertices, status=f"Gathering skin weights for {mesh_path}...")

    elapsed_time = time.time() - start_time
    print(f"Elapsed time to gather skin weights: {elapsed_time:.2f} seconds")

    # Combine all sparse weights into one matrix
    total_vertices = vertex_offset
    num_joints = max(sparse.shape[1] for sparse in all_sparse_weights)
    combined_sparse_weights = lil_matrix((total_vertices, num_joints))

    current_vertex = 0
    for sparse_weights in all_sparse_weights:
        num_vertices = sparse_weights.shape[0]
        combined_sparse_weights[current_vertex:current_vertex + num_vertices, :sparse_weights.shape[1]] = sparse_weights
        current_vertex += num_vertices

    elapsed_time2 = time.time() - start_time - elapsed_time
    print(f"Elapsed time to cluster vertices: {elapsed_time2:.2f} seconds")

    clusters = collections.defaultdict(list)

    # Iterate over each vertex in the combined sparse weights
    for vertex_index in range(combined_sparse_weights.shape[0]):
        # Extract the sparse weights for the current vertex
        sparse_row = combined_sparse_weights.getrow(vertex_index)
        if sparse_row.nnz == 0:  # Skip vertices with no weights (fully zero row)
            continue

        # Convert to dense array, round to specified precision, and flatten to 1D
        vertex_weights = np.round(sparse_row.toarray(), precision).flatten()
        rounded_weights = tuple(vertex_weights)
        clusters[rounded_weights].append(vertex_index)

    elapsed_time3 = time.time() - start_time - elapsed_time - elapsed_time2
    print(f"Elapsed time to cluster vertices: {elapsed_time3:.2f} seconds")

    # Initialize labels, where each vertex is labeled with -1 (isolated)
    labels = np.full(combined_sparse_weights.shape[0], -1, dtype=int)
    cluster_id = 0
    for weights, indices in clusters.items():
        if len(indices) > min_vertices_per_cluster:
            labels[indices] = cluster_id  # Assign a cluster ID to all vertices in the cluster
            cluster_id += 1

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, step=vertex_offset, status="Clustering vertices...")

    return labels


@util.timeit
def refine_clusters_by_topology(mesh_paths, labels, tolerance=1e-6):
    # type: (List[om.MDagPath|str], np.ndarray, float) -> np.ndarray
    """
    Refine clusters by checking topological adjacency across multiple meshes.

    :param mesh_paths: A list of meshes for which to refine the clusters
    :param labels: Initial cluster labels based on weight similarity
    :return: Refined cluster labels taking topology into account
    """
    total_vertices = labels.shape[0]
    adjacency_matrix = lil_matrix((total_vertices, total_vertices), dtype=bool)
    all_points = []
    vertex_offsets = []
    vertex_offset = 0

    bar = mel.eval("$tmp = $gMainProgressBar")

    # Collect all points and build adjacency matrix
    for mesh_path in mesh_paths:
        mesh_fn = util.get_mesh_fn(mesh_path)
        num_vertices = mesh_fn.numVertices
        points = util.convert_points_to_numpy(mesh_path)
        all_points.append(points)
        vertex_offsets.append(vertex_offset)

        # Build local adjacency matrix
        vertex_iter = om.MItMeshVertex(mesh_path)
        while not vertex_iter.isDone():
            local_index = vertex_iter.index()
            global_index = local_index + vertex_offset
            connected_vertices = vertex_iter.getConnectedVertices()
            connected_vertices = [idx + vertex_offset for idx in connected_vertices]
            adjacency_matrix[global_index, connected_vertices] = True
            vertex_iter.next()

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=num_vertices, status=f"Collecting to connectivities for {mesh_path}...")

        vertex_offset += num_vertices

    # Combine all points into a single array
    all_points = np.vstack(all_points)

    # Also check for vertices that are in the same position (within the tolerance)
    tree = cKDTree(all_points)
    pairs = tree.query_pairs(r=tolerance)

    for i, j in pairs:
        adjacency_matrix[i, j] = True
        adjacency_matrix[j, i] = True

    # Array to mark visited vertices
    visited = np.full(total_vertices, False)

    # Function to check if all neighbors of a vertex are within the same cluster
    def is_fully_connected_within_cluster(vertex, current_cluster):
        neighbors = adjacency_matrix[vertex].nonzero()[1]
        return np.all(labels[neighbors] == current_cluster)

    # Function to perform breadth-first search (BFS) to find connected components
    def bfs(start_vertex, current_cluster):
        queue = [start_vertex]
        cluster_vertices = []
        dissolve = False  # Flag to determine if the cluster should be dissolved

        while queue:
            vertex = queue.pop(0)
            if visited[vertex]:
                continue

            visited[vertex] = True
            cluster_vertices.append(vertex)

            # Check if all adjacent vertices are in the same cluster
            if not is_fully_connected_within_cluster(vertex, current_cluster):
                dissolve = True  # Mark the cluster for dissolution

            # Get all connected vertices (where adjacency_matrix is True)
            neighbors = adjacency_matrix[vertex].nonzero()[1]
            queue.extend(neighbors[(labels[neighbors] == current_cluster) & (~visited[neighbors])])

        return cluster_vertices, dissolve

    # Refine clusters based on adjacency and dissolve clusters if necessary
    # Initialize refined labels (starting with the existing labels)
    cluster_id = 0
    refined_labels = np.full_like(labels, -1)

    for vertex_index in range(total_vertices):
        if visited[vertex_index] or labels[vertex_index] == -1:
            continue

        # Perform BFS to find all connected vertices in the same cluster
        current_cluster = labels[vertex_index]
        connected_vertices, dissolve = bfs(vertex_index, current_cluster)

        # If the cluster should not be dissolved, assign a new cluster ID
        if not dissolve and len(connected_vertices) > 1:
            refined_labels[connected_vertices] = cluster_id
            cluster_id += 1
        else:
            # Cluster is dissolved, all vertices are marked as -1 (isolated)
            refined_labels[connected_vertices] = -1

        if not cmds.about(batch=True):
            cmds.progressBar(bar, edit=True, step=1, status="Refining clusters...")

    return refined_labels


@util.timeit
def cluster_vertices(mesh_paths):
    # type: (List[om.MDagPath|str]) -> np.ndarray
    """
    Cluster vertices across multiple meshes.

    :param mesh_paths: A list of meshes to cluster
    :return: A list of cluster labels for each vertex across all meshes
    """

    vertex_offset = 0

    for mesh_path in mesh_paths:
        mesh_fn = util.get_mesh_fn(mesh_path)
        num_vertices = mesh_fn.numVertices
        vertex_offset += num_vertices

    bar = mel.eval("$tmp = $gMainProgressBar")
    if not cmds.about(batch=True):
        cmds.progressBar(
                bar,
                edit=True,
                beginProgress=True,
                status="Clustering vertices...",
                maxValue=vertex_offset * 4,
        )

    labels = cluster_vertices_by_skin_weight(mesh_paths)
    refined_labels = refine_clusters_by_topology(mesh_paths, labels)

    if not cmds.about(batch=True):
        cmds.progressBar(bar, edit=True, endProgress=True)

    return refined_labels
