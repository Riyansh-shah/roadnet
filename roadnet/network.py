"""Road cost of straight segments and the road network built on a set of points.

Given the houses plus some junction points, the cheapest way to join them all with straight roads is a minimum
spanning tree (MST). Hill climbing only decides where the junctions go; the MST decides which roads to build.
"""
import numpy as np

from .world import LAKE_COST


def segment_cost(p, q, lakes):
    """Cost of straight roads p -> q (arrays broadcast to (..., 2)): length, plus (LAKE_COST - 1) x the part in a lake.

    The part inside a circle comes from solving |p + t (q - p) - c| = r for t, then clipping t to [0, 1].
    """
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    length = np.hypot(d[..., 0], d[..., 1])
    cost = length.copy()
    a = np.maximum(length ** 2, 1e-12)
    for cx, cy, r in lakes:
        fx, fy = p[..., 0] - cx, p[..., 1] - cy
        b = 2 * (fx * d[..., 0] + fy * d[..., 1])
        c = fx * fx + fy * fy - r * r
        root = np.sqrt(np.maximum(b * b - 4 * a * c, 0.0))
        t1 = np.clip((-b - root) / (2 * a), 0, 1)
        t2 = np.clip((-b + root) / (2 * a), 0, 1)
        cost += (LAKE_COST - 1) * (t2 - t1) * length
    return cost


def cost_matrix(points, lakes):
    """Cost of the straight road between every pair of points."""
    return segment_cost(points[:, None, :], points[None, :, :], lakes)


def mst(cost):
    """Prim's algorithm on a full cost matrix. Returns (total cost, edges as an (m-1, 2) int array)."""
    m = len(cost)
    if m < 2:
        return 0.0, np.zeros((0, 2), int)
    in_tree = np.zeros(m, bool)
    in_tree[0] = True
    best = cost[0].copy()
    parent = np.zeros(m, int)
    edges = np.zeros((m - 1, 2), int)
    total = 0.0
    for k in range(m - 1):
        j = int(np.argmin(np.where(in_tree, np.inf, best)))
        edges[k] = parent[j], j
        total += best[j]
        in_tree[j] = True
        closer = cost[j] < best
        best = np.where(closer, cost[j], best)
        parent = np.where(closer, j, parent)
    return float(total), edges


def baseline(world):
    """The network without junctions: the MST over the houses alone. This is what you'd build with no search."""
    return mst(cost_matrix(world.houses, world.lakes))


def degrees(edges, m):
    return np.bincount(edges.ravel(), minlength=m)


def junction_angles(points, edges, node):
    """Angles (degrees) between consecutive roads leaving `node`, going round it. A perfect junction has 120, 120, 120."""
    others = np.concatenate([edges[edges[:, 0] == node, 1], edges[edges[:, 1] == node, 0]])
    if len(others) < 2:
        return []
    v = points[others] - points[node]
    heading = np.sort(np.degrees(np.arctan2(v[:, 1], v[:, 0])))
    gaps = np.diff(np.append(heading, heading[0] + 360))
    return [float(g) for g in gaps]

