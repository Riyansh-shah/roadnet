"""Batch experiments -> results/*.csv. Calibration used seeds 0-39, so all experiments use seeds from 100 up.

E1  leave-one-out ablation: 5 methods x seeds 100-199, N = 10 rooms, K = 3 units
E2  scaling: FULL vs noGA x K in {1,2,3,4,6} x N in {6,10,16} x seeds 100-149
E3  allocation quality: the allocation problem the team faces at t = 30 s, GA vs greedy vs random,
    for (N=10, K=3) and (N=16, K=6), seeds 100-199; plus GA convergence curves

    python experiments.py --jobs 8                (all)
    python experiments.py --only e1 --jobs 8
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from navsim.config import METHODS
from navsim.fuzzy import cost_grid
from navsim.ga import ga_allocate, greedy_allocate, random_objective, routes_objective
from navsim.mission import Mission, run_episode

OUT = "results"


def _episode(job):
    return run_episode(*job)


def _allocation(job):
    seed, n_rooms, k = job
    m = Mission(seed, "FULL", n_rooms, k)
    while m.t < 30 and m.t_find is None and not m.done:
        m.step()
    if m.t_find is not None or m.done:
        return None, []
    areas, D, P, dwell, t0, searchers = m.allocation_problem(cost_grid(m.belief, m.b.solid, m.cfg))
    K = len(searchers)
    if len(areas) < 2 or K == 0:
        return None, []
    rng = np.random.default_rng(seed)
    t = time.perf_counter()
    _, f_ga, curve = ga_allocate(D, P, dwell, t0, K, rng, m.cfg)
    ga_ms = 1000 * (time.perf_counter() - t)
    t = time.perf_counter()
    f_gr = routes_objective(greedy_allocate(D, P, dwell, t0, K), D, P, dwell, t0, K, rng)
    gr_ms = 1000 * (time.perf_counter() - t)
    f_rand = random_objective(D, P, dwell, t0, K, rng)
    row = dict(seed=seed, n_rooms=n_rooms, k_units=K, n_areas=len(areas), ga=round(f_ga, 3), greedy=round(f_gr, 3),
               random=round(f_rand, 3), ga_gain_pct=round(100 * (f_gr - f_ga) / f_gr, 2),
               ga_ms=round(ga_ms, 1), greedy_ms=round(gr_ms, 2), generations=len(curve) - 1)
    curve_rows = [dict(seed=seed, n_rooms=n_rooms, k_units=K, gen=g, best=round(b, 3), mean=round(mu, 3),
                       greedy=round(f_gr, 3)) for g, (b, mu) in enumerate(curve)]
    return row, curve_rows


def write(name, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def run_all(ex, fn, jobs, label):
    t, out = time.time(), []
    for i, r in enumerate(ex.map(fn, jobs, chunksize=2), 1):
        out.append(r)
        if i % 50 == 0 or i == len(jobs):
            print(f"  {label} {i}/{len(jobs)}  ({time.time() - t:.0f} s)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", default="e1,e2,e3")
    a = ap.parse_args()
    todo = a.only.split(",")
    with ProcessPoolExecutor(a.jobs) as ex:
        if "e1" in todo:
            jobs = [(s, m, 10, 3) for m in METHODS for s in range(100, 200)]
            write("e1.csv", run_all(ex, _episode, jobs, "E1"))
        if "e2" in todo:
            jobs = [(s, m, n, k) for n in (6, 10, 16) for k in (1, 2, 3, 4, 6) for m in ("FULL", "noGA")
                    for s in range(100, 150)]
            write("e2.csv", run_all(ex, _episode, jobs, "E2"))
        if "e3" in todo:
            jobs = [(s, n, k) for n, k in ((10, 3), (16, 6)) for s in range(100, 200)]
            res = run_all(ex, _allocation, jobs, "E3")
            write("e3.csv", [r for r, _ in res if r is not None])
            write("ga_curve.csv", [c for _, cs in res for c in cs])


if __name__ == "__main__":
    main()
