# Mesh Registration User's Guide

This guide explains how to use the mesh registration module to find correspondence points between meshes with different topologies. The registration process is essential for transferring attributes like skinning weights, deformations, or animations between different character meshes.

## Introduction

The mesh registration module uses skeletal information to find point correspondences between two meshes with different vertex counts and topologies. It employs advanced raycast-based techniques to establish these correspondences, which can then be used for RBF interpolation.

Key features:
- Skeleton-aware correspondence search
- Ray-casting with Embree acceleration (when available) or BVH fallback
- Parallel processing for improved performance
- Configurable accuracy vs. speed tradeoffs
- Visualization tools for debugging

## Prerequisites

Before using the registration module, ensure:
1. Both meshes have proper skinning (skin clusters)
2. Joint hierarchies have matching joint names
3. Meshes are in appropriate rest poses

## Basic Usage

The simplest way to use the registration module is with the `find_correspondence_pairs` function:

```python
from ymt_mesh_retarget.registration import find_correspondence_pairs

# Find correspondence points between two meshes
source_points, target_points = find_correspondence_pairs(
    source_mesh="sourceCharacterMesh",
    target_mesh="targetCharacterMesh"
)

# Now use these points for attribute transfer
# For example, using RBF interpolation
```

This function handles all the necessary steps, including:
- Analyzing skin weights
- Building joint hierarchies
- Aligning source and target spaces
- Finding optimal correspondence pairs

## Advanced Usage

For more control over the registration process, use the `MeshRegistration` class:

```python
from ymt_mesh_retarget.registration import (
    MeshRegistration,
    get_default_registration_options
)

# Get default options and customize them
options = get_default_registration_options()
options.sample_rate = 0.3  # Process 30% of vertices for faster results
options.sample_number = 16  # Use fewer rays per point
options.num_threads = 8     # Specify thread count

# Create registration object
registration = MeshRegistration(
    source_mesh="sourceCharacterMesh",
    target_mesh="targetCharacterMesh",
    options=options
)

# Find correspondence pairs
source_points, target_points = registration.find_correspondence_pairs()

# Visualize the results
group_node = registration.visualize_correspondences(line_thickness=2)

# Access the correspondence points directly if needed
for cp in registration.correspondence_points:
    print(f"Source: {cp.source_index}, Target: {cp.target_index}, Weight: {cp.weight}")
```

## Performance Optimization

The registration process can be computationally intensive for complex meshes. Here are some tips to optimize performance:

### For Quick Tests

```python
options = get_default_registration_options()
options.sample_rate = 0.1       # Process only 10% of vertices
options.sample_number = 8       # Use just 8 rays per point
options.sample_degree = 30.0    # Narrower ray cone
options.batch_size = 512        # Smaller batches for lower memory usage
```

### For Production Quality

```python
options = get_default_registration_options()
options.sample_rate = 0.8       # Process 80% of vertices
options.sample_number = 64      # More rays for better accuracy
options.sample_degree = 60.0    # Wider ray cone for better coverage
options.max_points_per_target = 4  # Allow more correspondences per target
```

### Hardware Optimization

The registration process automatically detects available CPU cores and uses multithreading. However, you can override these settings:

```python
# Manual thread control
options.num_threads = 4  # Use exactly 4 threads

# Adjust batch size for memory constraints
options.batch_size = 2048  # Larger batches on systems with more RAM
```

## Visualization and Debugging

When things don't work as expected, visualization can help identify issues:

```python
# Automatically visualize after finding correspondences
source_points, target_points = find_correspondence_pairs(
    source_mesh="sourceCharacterMesh",
    target_mesh="targetCharacterMesh",
    visualize=True  # Creates visualization automatically
)

# Or visualize manually with the MeshRegistration class
registration = MeshRegistration("sourceCharacterMesh", "targetCharacterMesh")
source_points, target_points = registration.find_correspondence_pairs()
lines_group = registration.visualize_correspondences(line_thickness=3)

# The visualization uses color coding:
# - Red: Low confidence correspondence
# - Yellow: Medium confidence
# - Green: High confidence
```

## Common Issues and Solutions

### No Correspondence Points Found

If the function raises an error about no correspondence points:

1. Check that both meshes have proper skin clusters
2. Ensure joint hierarchies have matching joint names
3. Try aligning the meshes closer in world space
4. Increase the sample_degree parameter for wider ray casting

```python
options = get_default_registration_options()
options.sample_degree = 90.0  # Much wider angle for difficult cases
options.min_weight_threshold = 0.001  # Allow lower quality matches
```

### Poor Quality Correspondences

If the correspondences don't look accurate:

1. Increase the sample_number for more rays
2. Adjust the weight_decay parameter
3. Try with and without align_spaces=True

```python
options = get_default_registration_options()
options.sample_number = 64  # More rays for better coverage
options.weight_decay = 1.5  # Adjust for more even influence distribution
```

### Slow Performance

If the process is too slow:

1. Reduce sample_rate to process fewer vertices
2. Decrease sample_number to cast fewer rays
3. Ensure use_bvh=True for spatial acceleration
4. Check if Embree is available in your Maya installation

## Advanced Topics

### Custom Correspondence Visualization

You can create custom visualizations for the correspondence points:

```python
from maya import cmds
from maya.api import OpenMaya as om

# After finding correspondences
for i, (src_pt, tar_pt) in enumerate(zip(source_points, target_points)):
    # Create a curve or other visual element
    curve = cmds.curve(p=[src_pt.tolist(), tar_pt.tolist()], d=1)
    
    # Add custom attributes or colors
    shape = cmds.listRelatives(curve, shapes=True)[0]
    cmds.setAttr(f"{shape}.overrideEnabled", 1)
    cmds.setAttr(f"{shape}.overrideRGBColors", 1)
    cmds.setAttr(f"{shape}.overrideColorRGB", 0, 1, 1)  # Cyan color
```

### Registration Between Multiple Meshes

For transferring attributes across multiple meshes:

```python
# First, find correspondences between source and intermediate mesh
source_to_intermediate_src, source_to_intermediate_tar = find_correspondence_pairs(
    source_mesh="sourceMesh",
    target_mesh="intermediateMesh"
)

# Then, find correspondences between intermediate and target mesh
intermediate_to_target_src, intermediate_to_target_tar = find_correspondence_pairs(
    source_mesh="intermediateMesh",
    target_mesh="targetMesh"
)

# Now you have a chain of correspondences that can be used
# for multi-stage attribute transfer
```

## Further Resources

For more information on the underlying techniques:
- "Skeleton-Aware Skin Weight Transfer" paper
- Example scripts in the examples/ directory
- API documentation for more details on specific functions

## Technical Support

If you encounter persistent issues:
1. Try with simplified test meshes to isolate the problem
2. Check the Maya script editor for detailed error messages
3. Contact support with specific examples and Maya scene files