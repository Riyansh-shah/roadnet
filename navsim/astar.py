"""A* on an 8-connected grid with per-cell costs (no corner cutting), plus Dijkstra for cost matrices.

Moving into cell v costs (1 or sqrt 2) * cost[v]. Heuristic:

    h(n) = c_min * octile(n, nearest goal),   octile(dr, dc) = max(dr, dc) + (sqrt 2 - 1) * min(dr, dc)

where c_min is the cheapest finite cell cost on the grid. Every move into a cell costs at least c_min per unit of length,
and octile distance is the shortest 8-connected length, so h never overestimates (admissible) and drops by at most the
move cost along any edge (consistent). With the fuzzy cost every cell costs about 2.4 or more, so c_min * octile is a
much tighter estimate than octile alone and A* expands far fewer nodes, while still returning the optimal path.
"""
import heapq
import math

import numpy as np

INF = math.inf
SQ2 = math.sqrt(2)
MOVES = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2))


def octile(dr, dc):
    dr, dc = abs(dr), abs(dc)
    return max(dr, dc) + (SQ2 - 1) * min(dr, dc)


def _search(cost, start, goals=None, hscale=1.0, stats=None):
    H, W = cost.shape
    flat = cost.ravel().tolist()
    s = start[0] * W + start[1]
    gset = {g[0] * W + g[1] for g in goals} if goals else set()
    glist = list(goals) if goals and hscale > 0 else []

    def h(r, c):
        if not glist:
            return 0.0
        return hscale * min(octile(r - gr, c - gc) for gr, gc in glist)

    g = {s: 0.0}
    parent = {s: -1}
    heap = [(h(*start), 0.0, s)]
    closed = set()
    found = None
    while heap:
        _, gu, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u in gset:
            found = u
            break
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
    if stats is not None:
        stats["expanded"] = stats.get("expanded", 0) + len(closed)
    if found is None:
        return None, INF, g
    path, u = [], found
    while u != -1:
        path.append(divmod(u, W))
        u = parent[u]
    return path[::-1], g[found], g


def c_min(cost):
    finite = cost[np.isfinite(cost)]
    return float(finite.min()) if finite.size else 1.0


def astar(cost, start, goals, heuristic="tight", stats=None):
    """Cheapest path from start to any goal. Returns (list of (r, c) incl. start, cost) or (None, inf).
    heuristic: "tight" = c_min * octile (default), "octile" = octile alone, "none" = Dijkstra (h = 0)."""
    hscale = {"tight": c_min(cost), "octile": 1.0, "none": 0.0}[heuristic]
    path, c, _ = _search(cost, tuple(start), [tuple(x) for x in goals], hscale, stats)
    return path, c


def dijkstra(cost, src):
    """Cost of the cheapest path from src to every cell (inf where unreachable)."""
    _, _, g = _search(cost, tuple(src), hscale=0.0)
    H, W = cost.shape
    d = np.full(H * W, INF)
    d[list(g.keys())] = list(g.values())
    return d.reshape(H, W)
