"""Sanity checks for every component. Run: python selftest.py"""
import math

import numpy as np

from navsim.astar import astar, dijkstra
from navsim.belief import BeliefMap
from navsim.building import make_building
from navsim.config import Config
from navsim.fuzzy import cost_grid, infer, penalty
from navsim.ga import BIG, decode, fitness, ga_allocate, greedy_allocate
from navsim.mission import Mission
from navsim.sensor import Obs

cfg = Config()
rng = np.random.default_rng(0)


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    assert ok, name


# A* returns the same cost as Dijkstra on random grids
worst = 0.0
for _ in range(30):
    cost = rng.uniform(1, 10, (15, 20))
    cost[rng.random(cost.shape) < 0.2] = np.inf
    free = np.argwhere(np.isfinite(cost))
    s, g = map(tuple, free[rng.choice(len(free), 2, replace=False)])
    _, c = astar(cost, s, [g])
    d = dijkstra(cost, s)[g]
    worst = max(worst, 0 if (math.isinf(c) and math.isinf(d)) else abs(c - d))
check("A* cost equals Dijkstra", worst < 1e-9, f"(max diff {worst:.2e})")

fewer, same = 0, True
for _ in range(30):
    cost = rng.uniform(2.4, 20, (15, 20))
    cost[rng.random(cost.shape) < 0.2] = np.inf
    free = np.argwhere(np.isfinite(cost))
    s, g = map(tuple, free[rng.choice(len(free), 2, replace=False)])
    st_t, st_o = {}, {}
    _, ct = astar(cost, s, [g], "tight", st_t)
    _, co = astar(cost, s, [g], "octile", st_o)
    same &= (math.isinf(ct) and math.isinf(co)) or abs(ct - co) < 1e-9
    fewer += st_t["expanded"] <= st_o["expanded"]
check("tight heuristic c_min*octile keeps the optimal cost and expands no more nodes", same and fewer == 30,
      f"({fewer}/30 queries)")

cost = np.ones((5, 20))
path, c = astar(cost, (2, 5), [(2, 0), (2, 19)])
check("multi-goal A* picks the nearer exit", path[-1] == (2, 0), f"(cost {c:.1f})")

# fuzzy system
lo, hi = float(infer(0, 0, 0, 1)), float(infer(1, 1, 0, 1))
check("penalty grows with hazard and blockage", lo < hi, f"({lo:.2f} < {hi:.2f})")
unit, civ = float(infer(0.5, 0, 0, 1)), float(infer(0.5, 0, 0, 0.3))
check("civilian penalised more in smoke than a protected unit", civ > unit, f"({civ:.2f} > {unit:.2f})")
stale, fresh = float(infer(0.5, 0, 1, 1)), float(infer(0.5, 0, 0, 1))
check("stale smoke reading penalised more than a fresh one", stale > fresh, f"({stale:.2f} > {fresh:.2f})")
x = rng.random((4, 3000))
err = np.abs(penalty(*x) - infer(*x))
check("lookup table matches the direct FIS", np.percentile(err, 99) < 0.02 and err.mean() < 0.005,
      f"(mean {err.mean():.4f}, p99 {np.percentile(err, 99):.3f}, max {err.max():.3f})")

# Bayes
b = make_building(10, np.random.default_rng(1), cfg)
bel = BeliefMap(b, cfg)
bel_d = BeliefMap(b, Config(debris=True))         # blockage filter, tested with debris switched on
cells = np.argwhere(b.area_id == 3)
o = Obs(cells[:, 0], cells[:, 1], np.full(len(cells), 1.0), np.zeros(len(cells), bool), np.zeros(len(cells)), False)
m0 = bel.area_mass()[3]
for _ in range(3):
    bel.update(o)
m1 = bel.area_mass()[3]
check("misses drain an area's P(person)", m1 < 0.05 * m0, f"({m0:.3f} -> {m1:.4f})")
check("P(person) stays normalised", abs(bel.pp.sum() - 1) < 1e-9)
door = tuple(np.argwhere(b.door)[0])
o = Obs(np.array([door[0]]), np.array([door[1]]), np.array([2.0]), np.array([True]), np.array([0.0]), False)
for _ in range(2):
    bel_d.update(o)
check("two 'blocked' readings make a door impassable", bel_d.p_block[door] > cfg.impassable_p,
      f"(p = {bel_d.p_block[door]:.2f})")
check("believed-blocked door is impassable in the cost grid", math.isinf(cost_grid(bel_d, b.solid, cfg)[door]))

# GA
M, K = 9, 3
D = rng.uniform(5, 40, (K + M, M))
P = rng.dirichlet(np.ones(M))
dwell, t0 = np.full(M, 8.0), np.array([0.0, 2.0, 4.0])


def slow(chrom):
    total = 0.0
    for k, route in enumerate(decode(chrom, M, K)):
        t, node = t0[k], k
        for a in route:
            t += D[node, a]
            total += P[a] * t
            t += dwell[a]
            node = K + a
    return total


pop = np.array([rng.permutation(M + K - 1) for _ in range(50)])
diff = np.abs(fitness(pop, D, P, dwell, t0, K) - [slow(c) for c in pop]).max()
check("vectorised GA fitness matches a plain decode", diff < 1e-9, f"(max diff {diff:.1e})")

wins = 0
for seed in range(20):
    mi = Mission(seed, "FULL")
    areas, D, P, dwell, t0, searchers = mi.allocation_problem(cost_grid(mi.belief, mi.b.solid, mi.cfg))
    k = len(searchers)
    _, f_ga, _ = ga_allocate(D, P, dwell, t0, k, np.random.default_rng(seed), mi.cfg)
    from navsim.ga import routes_objective
    f_gr = routes_objective(greedy_allocate(D, P, dwell, t0, k), D, P, dwell, t0, k, np.random.default_rng(0))
    wins += f_ga <= f_gr + 1e-9
check("GA at least as good as greedy on t = 0 problems", wins >= 16, f"({wins}/20)")

lower_ok, eff_ok = True, True
for n in range(8):
    mi = Mission(n, "FULL")
    er, ec = mi.b.entrance
    pr, pc = mi.world.person
    lower_ok &= mi.oracle_rescue >= 2 * (max(abs(er - pr), abs(ec - pc)) - 1) - 1
    res = mi.run()
    eff_ok &= res["outcome"] != "rescued" or res["t_rescue"] >= mi.oracle_rescue
check("oracle rescue time is physically possible and never beaten by the drones", lower_ok and eff_ok)

r = Mission(0, "FULL").run()
check("an episode runs end to end", r["outcome"] in ("rescued", "casualty", "timeout"), f"({r['outcome']}, t = {r['ticks']})")
print("all checks passed")
