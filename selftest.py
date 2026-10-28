"""Checks that the road network code and the climbers give the right answers.   python selftest.py"""
import math
import sys

import numpy as np

from roadnet.climb import CLIMBERS, Problem, run
from roadnet.network import baseline, cost_matrix, junction_angles, mst, segment_cost
from roadnet.world import random_world, square, triangle

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {len(results):2d}. {name}" + (f"  ({detail})" if detail else ""))


# 1-3: road costs
lake = np.array([[500.0, 500.0, 100.0]])
c = float(segment_cost([0, 500], [1000, 500], lake))
check("a road through a lake's centre costs L + 2 x 2r", abs(c - 1400) < 1e-6, f"{c:.1f} m, expected 1400")
c = float(segment_cost([0, 0], [1000, 0], lake))
check("a road that misses the lake costs its length", abs(c - 1000) < 1e-6, f"{c:.1f} m")
c = float(segment_cost([500, 450], [500, 550], lake))
check("a road entirely inside the lake costs 3x", abs(c - 300) < 1e-6, f"{c:.1f} m")

# 4: MST on a hand-worked case: points on a line 0, 100, 300, plus one 400 above the middle -> 100 + 200 + 400
P = np.array([[0, 0], [100, 0], [300, 0], [100, 400]], float)
total, edges = mst(cost_matrix(P, np.zeros((0, 3))))
pairs = sorted(tuple(sorted(e)) for e in edges.tolist())
check("MST of a hand-worked 4-point case", abs(total - 700) < 1e-9 and pairs == [(0, 1), (1, 2), (1, 3)], f"{total:.0f} m")

# 5-6: baselines of the presets
check("square baseline is 3 sides = 3000 m", abs(baseline(square())[0] - 3000) < 1e-6)
check("triangle baseline is 2 sides = 2000 m", abs(baseline(triangle())[0] - 2000) < 1e-6)

# 7+: every climber on the presets reaches the known best network
for name, climber in CLIMBERS.items():
    w = triangle()
    b = run(climber(w, seed=1)).best
    centre = w.houses.mean(axis=0)
    off = float(np.hypot(*(b.junctions[0] - centre))) if len(b.junctions) == 1 else math.inf
    check(f"{name}: triangle reaches sqrt(3) x side, junction at the centre",
          abs(b.cost - w.optimum) < 0.5 and off < 2, f"{b.cost:.2f} m vs {w.optimum:.2f}, {off:.1f} m from centre")

    w = square()
    b = run(climber(w, seed=1)).best
    pts = Problem(w).points(b.junctions)
    angles = [a for j in range(w.n, len(pts)) for a in junction_angles(pts, b.edges, j)]
    check(f"{name}: square reaches the H, (1 + sqrt 3) x side, 120 deg corners",
          abs(b.cost - w.optimum) < 1 and len(b.junctions) == 2 and all(abs(a - 120) < 2 for a in angles),
          f"{b.cost:.2f} m vs {w.optimum:.2f}, angles {', '.join(f'{a:.1f}' for a in angles)}")

# never worse than the baseline, with and without lakes; deterministic for a fixed seed
for name, climber in CLIMBERS.items():
    worse = 0
    for seed in range(10):
        for lakes in (False, True):
            w = random_world(10, seed, lakes)
            worse += run(climber(w, seed=seed)).best.cost > baseline(w)[0] + 1e-6
    check(f"{name}: never worse than the no-junction baseline (20 maps)", worse == 0, f"{worse} worse")
    w = random_world(9, 5, True)
    a, b = run(climber(w, seed=3)).best, run(climber(w, seed=3)).best
    check(f"{name}: same seed, same result", a.cost == b.cost and np.array_equal(a.junctions, b.junctions))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
