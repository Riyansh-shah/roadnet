"""A* on an 8-connected grid with per-cell costs (no corner cutting), plus Dijkstra for cost matrices.

Moving into cell v costs (1 or sqrt 2) * cost[v]. Every finite cost is >= 1, so the octile distance to the
nearest goal is an admissible heuristic. Several goals are allowed (e.g. "any exit").
"""
import heapq
import math

import numpy as np

INF = math.inf
SQ2 = math.sqrt(2)
MOVES = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))


def _search(cost, start, goals=None):
    H, W = cost.shape
    flat = cost.ravel().tolist()
    s = start[0] * W + start[1]
    gset = {g[0] * W + g[1] for g in goals} if goals else set()
    glist = list(goals) if goals else []

    def h(r, c):
        if not glist:
            return 0.0
        best = INF
        for gr, gc in glist:
            dr, dc = abs(r - gr), abs(c - gc)
            v = dr + dc + (SQ2 - 2) * min(dr, dc)
            if v < best:
                best = v
        return best

    g = {s: 0.0}
    parent = {s: -1}
    heap = [(h(*start), 0.0, s)]
    closed = set()
    while heap:
        _, gu, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u in gset:
            path = []
            while u != -1:
                path.append(divmod(u, W))
                u = parent[u]
            return path[::-1], gu, g
        r, c = divmod(u, W)
        for dr, dc, L in MOVES:
            rr, cc = r + dr, c + dc
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            v = rr * W + cc
            cv = flat[v]
            if cv == INF or v in closed:
                continue
            if dr and dc and (flat[r * W + cc] == INF or flat[rr * W + c] == INF):
                continue
            ng = gu + L * cv
            if ng < g.get(v, INF):
                g[v] = ng
                parent[v] = u
                heapq.heappush(heap, (ng + h(rr, cc), ng, v))
    return None, INF, g


def astar(cost, start, goals):
    """Cheapest path from start to any goal. Returns (list of (r, c) incl. start, cost) or (None, inf)."""
    path, c, _ = _search(cost, tuple(start), [tuple(x) for x in goals])
    return path, c


def dijkstra(cost, src):
    """Cost of the cheapest path from src to every cell (inf where unreachable)."""
    _, _, g = _search(cost, tuple(src))
    H, W = cost.shape
    d = np.full(H * W, INF)
    d[list(g.keys())] = list(g.values())
    return d.reshape(H, W)
