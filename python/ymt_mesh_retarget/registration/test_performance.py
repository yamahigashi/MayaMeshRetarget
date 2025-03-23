"""Test script for performance optimizations in mesh registration.

This script demonstrates the benchmarking tools and compares performance
before and after optimizations.
"""

import time
from typing import Any, cast

from maya import cmds

from ..objects import create_retargetable_object
from ..objects.mesh import MeshObject
from .benchmark import benchmark_raycast_engines_comparison, print_benchmark_results
from .main import MeshRegistration
from .utils import get_default_registration_options


def run_engine_comparison(source_mesh_name: str, num_rays: int = 5000) -> dict[str, Any]:
    """Run a comparison between raycast engines.

    Args:
        source_mesh_name: Name of source mesh to use for testing
        num_rays: Number of rays to cast

    Returns:
        Benchmark results
    """
    source_obj = create_retargetable_object(source_mesh_name)
    source_mesh_obj = cast("MeshObject", source_obj)  # Cast to expected type for benchmark function

    print("Comparing raycast engines...")
    results = benchmark_raycast_engines_comparison(source_mesh_obj, num_rays)
    print_benchmark_results(results, f"Raycast Engine Comparison - {num_rays} rays")

    return results


def run_registration_comparison(
    source_mesh_name: str,
    target_mesh_name: str,
    num_tests: int = 3,
) -> dict[str, dict[str, Any]]:
    """Run a comparison of registration performance with different options.

    Args:
        source_mesh_name: Name of source mesh
        target_mesh_name: Name of target mesh
        num_tests: Number of times to repeat each test

    Returns:
        Benchmark results for different configurations
    """
    source_obj = create_retargetable_object(source_mesh_name)
    target_obj = create_retargetable_object(target_mesh_name)

    # Cast to MeshObject for type compatibility
    source_mesh_obj = cast("MeshObject", source_obj)
    target_mesh_obj = cast("MeshObject", target_obj)

    # Test results
    results: dict[str, dict[str, Any]] = {}

    # Default options with standard raycast
    options_standard = get_default_registration_options()
    options_standard.use_bvh = False
    options_standard.num_threads = 1
    options_standard.sample_number = 16  # Reduce for quicker testing

    # Enhanced options with BVH acceleration
    options_bvh = get_default_registration_options()
    options_bvh.use_bvh = True
    options_bvh.num_threads = 1
    options_bvh.sample_number = 16  # Reduce for quicker testing

    # Enhanced options with BVH acceleration and multithreading
    options_optimal = get_default_registration_options()
    options_optimal.use_bvh = True
    options_optimal.sample_number = 16  # Reduce for quicker testing

    # Run tests
    print("Testing standard raycast without BVH...")
    standard_times: list[float] = []
    for i in range(num_tests):
        registration = MeshRegistration(source_mesh_obj, target_mesh_obj, options=options_standard)
        start_time = time.time()
        source_points, target_points = registration.find_correspondence_pairs()
        elapsed_time = time.time() - start_time
        standard_times.append(elapsed_time)
        print(f"  Test {i + 1}/{num_tests}: {elapsed_time:.2f} seconds")

    results["standard"] = {
        "avg_time": sum(standard_times) / len(standard_times),
        "min_time": min(standard_times),
        "max_time": max(standard_times),
        "num_points": len(source_points),
    }

    print("Testing with BVH acceleration...")
    bvh_times: list[float] = []
    for i in range(num_tests):
        registration = MeshRegistration(source_mesh_obj, target_mesh_obj, options=options_bvh)
        start_time = time.time()
        source_points, target_points = registration.find_correspondence_pairs()
        elapsed_time = time.time() - start_time
        bvh_times.append(elapsed_time)
        print(f"  Test {i + 1}/{num_tests}: {elapsed_time:.2f} seconds")

    results["bvh"] = {
        "avg_time": sum(bvh_times) / len(bvh_times),
        "min_time": min(bvh_times),
        "max_time": max(bvh_times),
        "num_points": len(source_points),
    }

    print("Testing with BVH acceleration and multithreading...")
    optimal_times: list[float] = []
    for i in range(num_tests):
        registration = MeshRegistration(source_mesh_obj, target_mesh_obj, options=options_optimal)
        start_time = time.time()
        source_points, target_points = registration.find_correspondence_pairs()
        elapsed_time = time.time() - start_time
        optimal_times.append(elapsed_time)
        print(f"  Test {i + 1}/{num_tests}: {elapsed_time:.2f} seconds")

    results["optimal"] = {
        "avg_time": sum(optimal_times) / len(optimal_times),
        "min_time": min(optimal_times),
        "max_time": max(optimal_times),
        "num_points": len(source_points),
    }

    # Calculate speedups
    if results["standard"]["avg_time"] > 0:
        results["bvh"]["speedup"] = results["standard"]["avg_time"] / results["bvh"]["avg_time"]
        results["optimal"]["speedup"] = results["standard"]["avg_time"] / results["optimal"]["avg_time"]

    # Print summary
    print("\nPerformance Summary:")
    print(f"Standard: {results['standard']['avg_time']:.2f} seconds")
    print(f"BVH Only: {results['bvh']['avg_time']:.2f} seconds (Speedup: {results['bvh'].get('speedup', 0):.2f}x)")
    print(
        f"BVH + Multithreading: {results['optimal']['avg_time']:.2f} seconds (Speedup: {results['optimal'].get('speedup', 0):.2f}x)",
    )

    return results


if __name__ == "__main__":
    # Check if we have selected meshes
    selection = cmds.ls(selection=True)

    if len(selection) < 1:
        cmds.error(
            "Please select at least a source mesh for engine comparison. For full registration comparison, select both source and target meshes.",
        )

    # Run engine comparison
    source_mesh = selection[0]
    engine_results = run_engine_comparison(source_mesh)

    # If we have two meshes, run registration comparison
    if len(selection) >= 2:
        target_mesh = selection[1]
        registration_results = run_registration_comparison(source_mesh, target_mesh, num_tests=2)
