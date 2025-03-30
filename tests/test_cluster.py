from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ymt_mesh_retarget.cluster import cluster_vertices_by_skin_weight, refine_clusters_by_topology
from ymt_mesh_retarget.types import MeshPath


@pytest.fixture
def mock_mesh_paths():
    """Create mock mesh paths for testing."""
    # Since MeshPath is a Union type, we can't instantiate it directly
    # Just use strings which are valid MeshPath types
    mesh1 = "mesh1"
    mesh2 = "mesh2"
    return [mesh1, mesh2]


@pytest.fixture
def mock_sparse_weights():
    """Create mock sparse weight matrices for testing."""
    from scipy.sparse import lil_matrix
    
    # First mesh: 3 vertices, 2 influences
    weights1 = lil_matrix((3, 2))
    weights1[0, 0] = 1.0  # Vertex 0: full influence from joint 0
    weights1[1, 0] = 0.7  # Vertex 1: mixed influence
    weights1[1, 1] = 0.3
    weights1[2, 1] = 1.0  # Vertex 2: full influence from joint 1
    
    # Second mesh: 3 vertices, 2 influences
    weights2 = lil_matrix((3, 2))
    weights2[0, 0] = 0.9  # Vertex 0: similar to mesh1 vertex 0
    weights2[0, 1] = 0.1
    weights2[1, 0] = 0.4  # Vertex 1: similar to mesh1 vertex 1
    weights2[1, 1] = 0.6
    weights2[2, 1] = 0.95  # Vertex 2: similar to mesh1 vertex 2
    
    return [weights1, weights2]


@patch('ymt_mesh_retarget.util.get_skin_weight_as_sparse_matrix')
@patch('maya.api.OpenMaya.MFnMesh')
def test_cluster_vertices_by_skin_weight(mock_MFnMesh, mock_get_weights, mock_mesh_paths, mock_sparse_weights):
    """Test vertex clustering based on skin weights."""
    # Configure mocks
    mock_get_weights.side_effect = mock_sparse_weights
    
    # Mock the mesh function set that would be used to get vertex counts
    mock_mesh_instance = mock_MFnMesh.return_value
    mock_mesh_instance.numVertices = 3  # Set vertex count for each mesh to 3
    
    # Call the function
    with patch('ymt_mesh_retarget.util.get_mesh_fn', return_value=mock_mesh_instance):
        labels = cluster_vertices_by_skin_weight(
            mock_mesh_paths, 
            precision=2,
            min_vertices_per_cluster=1  # Low threshold to ensure we get clusters
        )
    
    # Verify result shape
    assert len(labels) == 6  # Total 6 vertices (3 from each mesh)
    
    # Verify clustering results - vertices with similar weights should be in the same cluster
    assert labels[0] == labels[3]  # mesh1[0] and mesh2[0] should be in same cluster
    assert labels[1] == labels[4]  # mesh1[1] and mesh2[1] should be in same cluster
    assert labels[2] == labels[5]  # mesh1[2] and mesh2[2] should be in same cluster
    
    # Verify all clusters are different
    assert len(set([labels[0], labels[1], labels[2]])) == 3


def test_refine_clusters_by_topology(mock_mesh_paths):
    """Test refining clusters based on topology."""
    # Create mock adjacency matrices (simplified)
    from scipy.sparse import lil_matrix
    
    # First mesh: linear strip of 4 vertices
    adj1 = lil_matrix((4, 4))
    adj1[0, 1] = 1  # Vertex 0 connected to vertex 1
    adj1[1, 0] = 1
    adj1[1, 2] = 1  # Vertex 1 connected to vertex 2
    adj1[2, 1] = 1
    adj1[2, 3] = 1  # Vertex 2 connected to vertex 3
    adj1[3, 2] = 1
    
    # Second mesh: similar structure
    adj2 = lil_matrix((4, 4))
    adj2[0, 1] = 1
    adj2[1, 0] = 1
    adj2[1, 2] = 1
    adj2[2, 1] = 1
    adj2[2, 3] = 1
    adj2[3, 2] = 1
    
    # Instead of mocking get_adjacency_matrix which doesn't exist,
    # mock the necessary Maya components used to build adjacency
    with patch('ymt_mesh_retarget.cluster.lil_matrix', side_effect=[adj1, adj2]):
        # Also mock mesh_fn.getVertices to return edges
        with patch('ymt_mesh_retarget.util.get_mesh_fn') as mock_get_mesh_fn:
            mock_mesh = mock_get_mesh_fn.return_value
            # Each call to getEdges returns a set of connected vertices
            mock_mesh.getEdges.return_value = ([0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2])
            mock_mesh.numVertices = 4
            
            # Initial labels - assume we have 2 clusters (0 and 1)
            # but vertices are not properly grouped by topology
            initial_labels = np.array([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.int32)
            
            # Call the function
            refined_labels = refine_clusters_by_topology(mock_mesh_paths, initial_labels)
    
    # Verify the shape
    assert len(refined_labels) == 8
    
    # The refined labels should respect topology - vertices should be in same cluster
    # as their neighbors if possible
    # Assuming tolerance works as expected:
    assert refined_labels[0] == refined_labels[1]  # Adjacent vertices 0-1 in mesh1
    assert refined_labels[2] == refined_labels[3]  # Adjacent vertices 2-3 in mesh1
    assert refined_labels[4] == refined_labels[5]  # Adjacent vertices 0-1 in mesh2
    assert refined_labels[6] == refined_labels[7]  # Adjacent vertices 2-3 in mesh2


def test_cluster_vertices_no_skin():
    """Test clustering when meshes have no skin weights."""
    # Create mock mesh paths using strings (valid MeshPath types)
    mesh_paths = ["mesh1", "mesh2"]
    
    # Mock util.get_skin_weight_as_sparse_matrix to raise ValueError
    with patch('ymt_mesh_retarget.util.get_skin_weight_as_sparse_matrix') as mock_get_weights:
        mock_get_weights.side_effect = ValueError("No skin cluster found")
        
        # Mock MFnMesh to provide vertex counts
        with patch('maya.api.OpenMaya.MFnMesh') as mock_MFnMesh:
            mock_mesh = mock_MFnMesh.return_value
            mock_mesh.numVertices = 3  # Set vertex count for each mesh
            
            # Mock get_mesh_fn to return our mock mesh
            with patch('ymt_mesh_retarget.util.get_mesh_fn', return_value=mock_mesh):
                # Call the function
                labels = cluster_vertices_by_skin_weight(mesh_paths)
                
                # Verify all vertices are labeled -1 (no cluster)
                assert len(labels) == 6
                assert np.all(labels == -1)  # All vertices should have label -1