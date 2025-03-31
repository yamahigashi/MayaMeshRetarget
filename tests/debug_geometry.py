import numpy as np

from ymt_mesh_retarget.registration.geometry import ray_triangle_intersection


def debug_ray_triangle_intersection():
    """Debug the ray-triangle intersection function."""
    # Triangle vertices
    v0 = np.array([0, 0, 0], dtype=np.float64)
    v1 = np.array([1, 0, 0], dtype=np.float64)
    v2 = np.array([0, 1, 0], dtype=np.float64)

    # Ray that should intersect the triangle
    orig = np.array([0.2, 0.2, 1], dtype=np.float64)
    dir_vec = np.array([0, 0, -1], dtype=np.float64)

    # Run ray-triangle intersection test
    hit, point, t = ray_triangle_intersection(orig, dir_vec, v0, v1, v2)

    # Print results
    print(f"Hit: {hit}")
    if point is not None:
        print(f"Intersection point: {point}")
    print(f"Parameter t: {t}")

    # Print intermediate calculations
    # Scaling
    center_tri = (v0 + v1 + v2) / 3.0
    v0_scaled = (v0 - center_tri) * 1.0 + center_tri
    v1_scaled = (v1 - center_tri) * 1.0 + center_tri
    v2_scaled = (v2 - center_tri) * 1.0 + center_tri

    print(f"Center of triangle: {center_tri}")
    print(f"v0 scaled: {v0_scaled}")
    print(f"v1 scaled: {v1_scaled}")
    print(f"v2 scaled: {v2_scaled}")

    # Triangle edge vectors
    e1 = v1_scaled - v0_scaled
    e2 = v2_scaled - v0_scaled

    print(f"Edge 1: {e1}")
    print(f"Edge 2: {e2}")

    # Normal vector
    n = np.cross(e1, e2)
    ndd = np.dot(dir_vec, n)

    print(f"Normal: {n}")
    print(f"Dot product (ray dir, normal): {ndd}")

    EPSILON = 1e-12

    # Möller–Trumbore algorithm
    h = np.cross(dir_vec, e2)
    a = np.dot(e1, h)

    print(f"h: {h}")
    print(f"a: {a}")

    if -EPSILON < a < EPSILON:
        print("Ray is parallel to triangle")

    f = 1.0 / a
    s = orig - v0_scaled
    u = f * np.dot(s, h)

    print(f"f: {f}")
    print(f"s: {s}")
    print(f"u: {u}")

    if u < 0.0 or u > 1.0:
        print("u out of bounds")

    q = np.cross(s, e1)
    v = f * np.dot(dir_vec, q)

    print(f"q: {q}")
    print(f"v: {v}")

    if v < 0.0 or u + v > 1.0:
        print("v out of bounds or u+v > 1")

    t = f * np.dot(e2, q)

    print(f"t: {t}")

    if t > EPSILON:
        intersection_point = orig + dir_vec * t
        print(f"Final intersection point: {intersection_point}")

if __name__ == "__main__":
    debug_ray_triangle_intersection()
