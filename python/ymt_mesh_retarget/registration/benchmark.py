"""Benchmarking utilities for mesh registration.

This module provides functions for benchmarking the performance of
various parts of the mesh registration process.
"""

import cProfile
import functools
import io
import pstats
import time
from typing import Any, Callable, Optional, TypeVar

import numpy as np
from maya import cmds

from .core import RegistrationOptions
from .raycast import RaycastEngine, get_raycast_engine
from .utils import get_default_registration_options


if typing.TYPE_CHECKING:
    from ..objects import MeshObject


# Type variable for generic function typing
T = TypeVar("T")


def timeit_detailed(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
    """Decorator that times a function execution and returns result with elapsed time.

    Unlike the standard timeit decorator, this returns both the function result
    and the elapsed time as a tuple.

    Args:
        func: Function to time

    Returns:
        Decorated function that returns (result, elapsed_time)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        return result, elapsed_time

    return wrapper


def profile_function(func: Callable[..., T]) -> Callable[..., tuple[T, str]]:
    """Decorator that profiles a function using cProfile.

    Args:
        func: Function to profile

    Returns:
        Decorated function that returns (result, profile_stats_string)
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        profile_output = s.getvalue()

        return result, profile_output

    return wrapper


def benchmark_raycast_engine(
    engine: RaycastEngine,
    num_rays: int = 10000,
    ray_batch_size: int = 1000,
) -> dict[str, float]:
    """Benchmark the performance of a raycast engine.

    Args:
        engine: RaycastEngine instance to benchmark
        num_rays: Number of rays to cast
        ray_batch_size: Size of ray batches

    Returns:
        Dictionary with benchmark metrics
    """
    # Generate random rays for testing
    np.random.seed(42)  # For reproducibility
    origins = np.random.randn(num_rays, 3).astype(np.float32)
    directions = np.random.randn(num_rays, 3).astype(np.float32)

    # Normalize directions
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms

    # Benchmark individual ray casting
    start_time = time.time()
    single_ray_hits = 0
    for i in range(min(100, num_rays)):  # Test only first 100 rays for single casting
        hit = engine.cast_ray(origins[i], directions[i])
        if hit is not None and hit.get("primID", -1) >= 0:
            single_ray_hits += 1
    single_ray_time = time.time() - start_time

    # Benchmark batch ray casting
    start_time = time.time()
    batch_hits = 0
    for i in range(0, num_rays, ray_batch_size):
        end_idx = min(i + ray_batch_size, num_rays)
        batch_origins = origins[i:end_idx]
        batch_directions = directions[i:end_idx]

        results = engine.cast_rays(batch_origins, batch_directions)
        batch_hits += np.sum(results["geomID"] >= 0)
    batch_ray_time = time.time() - start_time

    # Calculate metrics
    metrics = {
        "total_rays": num_rays,
        "single_ray_time": single_ray_time,
        "batch_ray_time": batch_ray_time,
        "rays_per_second_single": 100 / single_ray_time if single_ray_time > 0 else 0,
        "rays_per_second_batch": num_rays / batch_ray_time if batch_ray_time > 0 else 0,
        "hit_rate": batch_hits / num_rays,
        "speedup_factor": (100 / single_ray_time) / (num_rays / batch_ray_time)
        if batch_ray_time > 0 and single_ray_time > 0
        else 0,
    }

    return metrics


def benchmark_registration(
    source_mesh: "MeshObject",
    target_mesh: "MeshObject",
    options: Optional[RegistrationOptions] = None,
    profile: bool = False,
) -> dict[str, Any]:
    """Benchmark the full registration process.

    Args:
        source_mesh: Source mesh object
        target_mesh: Target mesh object
        options: Registration options (uses defaults if None)
        profile: Whether to run cProfile profiling

    Returns:
        Dictionary with benchmark metrics and timing information
    """
    from .main import MeshRegistration

    # Use default options if none provided
    if options is None:
        options = get_default_registration_options()

    # Create registration object
    registration = MeshRegistration(source_mesh, target_mesh, options=options)

    # Time/profile the correspondence finding
    if profile:
        find_corr_func = profile_function(registration.find_correspondence_pairs)
        (source_points, target_points), profile_output = find_corr_func()

        # Parse profile output to extract key metrics
        profile_metrics = _parse_profile_output(profile_output)

        # Combine with basic timing
        metrics = {
            "num_correspondence_points": len(source_points),
            "profile_data": profile_metrics,
        }
    else:
        find_corr_func = timeit_detailed(registration.find_correspondence_pairs)
        (source_points, target_points), elapsed_time = find_corr_func()

        metrics = {
            "total_time_seconds": elapsed_time,
            "num_correspondence_points": len(source_points),
            "points_per_second": len(source_points) / elapsed_time if elapsed_time > 0 else 0,
            "source_mesh_vertices": source_mesh.mesh_fn.numVertices,
            "target_mesh_vertices": target_mesh.mesh_fn.numVertices,
        }

    return metrics


def benchmark_raycast_engines_comparison(
    source_mesh: "MeshObject",
    num_rays: int = 10000,
    force_standard: bool = False,
) -> dict[str, dict[str, float]]:
    """Compare performance between Embree and standard raycast engines.

    Args:
        source_mesh: Source mesh with triangles to test against
        num_rays: Number of rays to cast
        force_standard: Whether to force using standard engine even if Embree is available

    Returns:
        Dictionary with benchmark results for each engine type
    """
    # Get mesh data
    mesh_fn = source_mesh.mesh_fn
    points = np.array(mesh_fn.getPoints())
    _tri_counts, tri_indices = mesh_fn.getTriangles()
    triangle_indices = np.array(tri_indices, dtype=np.int32)

    # Create standard engine
    standard_engine = get_raycast_engine(points, triangle_indices, force_standard=True)
    standard_metrics = benchmark_raycast_engine(standard_engine, num_rays)
    standard_engine.cleanup()

    results = {
        "standard_engine": standard_metrics,
    }

    # Create Embree engine if available and requested
    if not force_standard:
        try:
            embree_engine = get_raycast_engine(points, triangle_indices, force_standard=False)
            if not isinstance(embree_engine, type(standard_engine)):  # Different engine types
                embree_metrics = benchmark_raycast_engine(embree_engine, num_rays)
                embree_engine.cleanup()
                results["embree_engine"] = embree_metrics

                # Add comparison metrics
                results["comparison"] = {
                    "embree_vs_standard_speedup": (
                        embree_metrics["rays_per_second_batch"] / standard_metrics["rays_per_second_batch"]
                    )
                    if standard_metrics["rays_per_second_batch"] > 0
                    else 0,
                }
        except ImportError:
            # Embree not available
            pass

    return results


def _parse_profile_output(profile_output: str) -> dict[str, Any]:
    """Parse cProfile output to extract key performance metrics.

    Args:
        profile_output: String output from cProfile

    Returns:
        Dictionary with parsed performance metrics
    """
    metrics = {
        "top_functions": [],
    }

    lines = profile_output.strip().split("\n")

    # Skip header lines
    for i, line in enumerate(lines):
        if line.startswith("ncalls"):
            start_line = i + 1
            break
    else:
        return metrics

    # Parse function call data
    for i in range(start_line, min(start_line + 20, len(lines))):
        parts = lines[i].strip().split()
        if len(parts) >= 6:
            func_name = " ".join(parts[5:])
            metrics["top_functions"].append(
                {
                    "name": func_name,
                    "cumulative_time": float(parts[3]),
                    "per_call_time": float(parts[4]),
                    "num_calls": parts[0].split("/")[0],
                },
            )

    return metrics


def print_benchmark_results(results: dict[str, Any], title: str = "Benchmark Results") -> None:
    """Print benchmark results in a readable format.

    Args:
        results: Benchmark results dictionary
        title: Title for the results section
    """
    print("=" * 80)
    print(f"{title:^80}")
    print("=" * 80)

    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                if isinstance(subvalue, (int, float)):
                    if "time" in subkey:
                        print(f"  {subkey}: {subvalue:.4f} seconds")
                    elif "per_second" in subkey:
                        print(f"  {subkey}: {subvalue:.2f}")
                    elif "factor" in subkey or "speedup" in subkey:
                        print(f"  {subkey}: {subvalue:.2f}x")
                    else:
                        print(f"  {subkey}: {subvalue}")
                else:
                    print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    from ..objects import create_retargetable_object

    # Get selected meshes
    selection = cmds.ls(selection=True)
    if len(selection) != 2:
        raise ValueError("Select source and target meshes")

    source_obj = create_retargetable_object(selection[0])
    target_obj = create_retargetable_object(selection[1])

    # Run benchmark
    options = get_default_registration_options()
    options.sample_number = 8  # Reduce for quicker testing

    results = benchmark_registration(source_obj, target_obj, options)
    print_benchmark_results(results, "Registration Benchmark")

    # Compare raycast engines
    engine_results = benchmark_raycast_engines_comparison(source_obj, 5000)
    print_benchmark_results(engine_results, "Raycast Engine Comparison")
