"""The real-time mission loop: K units enter through one door, search with a shared belief, find the person and
escort them to the safest exit.

Each tick: the world evolves -> every unit senses -> the shared belief is updated -> the fuzzy cost grid is rebuilt
-> areas are (re)allocated if needed -> every unit picks a goal, (re)plans with A* and moves one cell.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .astar import astar, dijkstra
from .belief import BeliefMap, RawMap
from .building import make_building
from .config import METHODS, Config
from .fuzzy import cost_grid
from .ga import ga_allocate, greedy_allocate
from .sensor import observe
from .world import World


@dataclass
class Unit:
    id: int
    state: str = "QUEUE"      # QUEUE -> SEARCH -> APPROACH -> ESCORT -> OUT (finder); others SEARCH -> EXIT -> OUT
    pos: tuple = None
    route: list = field(default_factory=list)   # areas still to search, in order
    area: int = -1            # area being searched now
    area_m0: float = 0.0      # its unnormalised P(person) mass when the unit took it
    area_t0: int = -1         # tick the unit entered it
    path: list = field(default_factory=list)
    path_goal: tuple = None
    plan_t: int = -10 ** 9
    trail: list = field(default_factory=list)
    dist: float = 0.0
    exposure: float = 0.0
    bumps: int = 0
    stuck: int = 0            # consecutive ticks without a route


class Mission:
    def __init__(self, seed, method="FULL", n_rooms=10, k_units=3, cfg=None):
        self.cfg = cfg = cfg or Config()
        self.seed, self.method, self.m = seed, method, METHODS[method]
        ws, ss, gs = np.random.SeedSequence(seed).spawn(3)
        rng_w = np.random.default_rng(ws)          # building + world: identical for every method
        self.b = make_building(n_rooms, rng_w, cfg)
        self.world = World(self.b, cfg, rng_w)
        self.srng, self.grng = np.random.default_rng(ss), np.random.default_rng(gs)
        self.belief = BeliefMap(self.b, cfg) if self.m["bayes"] else RawMap(self.b, cfg)
        self.units = [Unit(i) for i in range(k_units)]
        self.exits = [tuple(e) for e in self.b.exits]
        self.t = 0
        self.t_find = self.t_rescue = self.outcome = self.escort = self.health_find = None
        self.dirty, self.last_alloc, self.n_bel_blocked = True, -10 ** 9, 0
        self.ga_calls = self.replans = 0
        self.ga_ms, self.tick_ms = [], []
        self.last_curve = None
        self.cost = None

    @property
    def done(self):
        return self.outcome is not None

    def active(self):
        return [u for u in self.units if u.state not in ("QUEUE", "OUT")]

    # ------------------------------------------------------------------ main loop
    def step(self):
        if self.done:
            return
        cfg, b, w, bel = self.cfg, self.b, self.world, self.belief
        w.step(occupied={u.pos for u in self.active()})
        t_start = time.perf_counter()

        for u in self.units:
            if u.state == "QUEUE" and self.t >= u.id * cfg.entry_gap:
                u.state, u.pos = "SEARCH", b.entrance
                u.trail.append(u.pos)

        bel.predict()
        self.in_view = np.zeros(b.shape, bool)
        for u in self.active():
            o = observe(w, u.pos, self.srng, cfg)
            bel.update(o)
            self.in_view[o.r, o.c] = True
            if o.detected and self.t_find is None:
                self.t_find, self.health_find = self.t, w.health

        nb = int((bel.p_block > 0.5).sum())
        if nb != self.n_bel_blocked:
            self.n_bel_blocked, self.dirty = nb, True
        self.cost = cost = cost_grid(bel, b.solid, cfg, 1.0, self.m["fuzzy"])

        if self.t_find is not None and self.escort is None:
            self._start_escort(cost)
        if self.t_find is None and (self.dirty or self.t - self.last_alloc >= cfg.realloc_every):
            self._allocate(cost)
        for u in self.active():
            self._act(u, cost)

        self.tick_ms.append(1000 * (time.perf_counter() - t_start))
        self.t += 1
        if self.outcome is None:
            if w.health <= 0:
                self.outcome = "casualty"
            elif self.t >= cfg.t_max:
                self.outcome = "timeout"

    def run(self):
        while not self.done:
            self.step()
        return self.metrics()

    # ------------------------------------------------------------------ allocation (GA / greedy / sweep)
    def allocation_problem(self, cost):
        """Areas worth searching, the cost matrix D (searchers + areas, areas), P, dwell, t0, searchers."""
        cfg, b, bel = self.cfg, self.b, self.belief
        searchers = [u for u in self.units if u.state in ("QUEUE", "SEARCH")]
        mass = bel.area_mass()
        areas = np.flatnonzero(mass > (cfg.mass_eps if bel.bayes else 0))
        if not len(areas):
            areas = np.flatnonzero(mass > 0)
        K = len(searchers)
        starts = [u.pos if u.state == "SEARCH" else b.entrance for u in searchers]
        t0 = np.array([0 if u.state == "SEARCH" else u.id * cfg.entry_gap - self.t for u in searchers], float)
        cache = {}

        def row(p):
            if p not in cache:
                cache[p] = dijkstra(cost, p)[tuple(b.viewpoints[areas].T)]
            return cache[p]

        D = np.array([row(p) for p in starts] + [row(tuple(b.viewpoints[a])) for a in areas])
        keep = np.flatnonzero(np.isfinite(D[:K]).any(axis=0))
        D = D[np.r_[np.arange(K), K + keep]][:, keep]
        areas = areas[keep]
        return areas, D, mass[areas], np.full(len(areas), cfg.dwell), np.maximum(t0, 0), searchers

    def _allocate(self, cost):
        self.last_alloc, self.dirty = self.t, False
        areas, D, P, dwell, t0, searchers = self.allocation_problem(cost)
        K = len(searchers)
        if not K or not len(areas):
            return
        if self.m["alloc"] == "ga":
            local = {int(a): i for i, a in enumerate(areas)}
            seed = [[local[a] for a in ([u.area] if u.area >= 0 else []) + u.route if a in local] for u in searchers]
            t = time.perf_counter()
            routes, _, self.last_curve = ga_allocate(D, P, dwell, t0, K, self.grng, self.cfg, seed)
            self.ga_ms.append(1000 * (time.perf_counter() - t))
            self.ga_calls += 1
        else:
            routes = greedy_allocate(D, P, dwell, t0, K, "ratio" if self.m["alloc"] == "greedy" else "nearest")
        for u, r in zip(searchers, routes):
            r = [int(areas[i]) for i in r]
            if u.area >= 0 and r and r[0] == u.area:
                u.route = r[1:]
            else:
                u.area, u.route = -1, r

    # ------------------------------------------------------------------ per-unit behaviour
    def _mass_un(self, a):
        return self.belief.area_mass(normalised=False)[a]

    def _area_done(self, u):
        if u.area_t0 >= 0 and self.t - u.area_t0 > self.cfg.dwell_max:
            return True
        if self.belief.bayes:
            return self._mass_un(u.area) < self.cfg.area_done_frac * u.area_m0
        return self.belief.coverage()[u.area] >= self.cfg.cover_frac

    def _finish_area(self, u):
        if not self.belief.bayes:        # coverage search never comes back to an area it gave up on
            self.belief.looked[self.b.area_id == u.area] = True
        u.area = -1

    def _search_goal(self, u, cost):
        b, bel = self.b, self.belief
        while True:
            if u.area < 0:
                if not u.route:
                    self.dirty = True
                    return None
                u.area = u.route.pop(0)
                u.area_m0, u.area_t0 = self._mass_un(u.area), -1
            if not self._area_done(u):
                break
            self._finish_area(u)
        if b.area_id[u.pos] == u.area and u.area_t0 < 0:
            u.area_t0 = self.t
        if u.area_t0 < 0:
            return tuple(int(v) for v in b.viewpoints[u.area])
        # inside the area. Bayes: the most probable cell (nearer cells break ties).
        # Coverage: the nearest cell not looked at yet.
        rr, cc = np.nonzero((b.area_id == u.area) & np.isfinite(cost))
        d = np.hypot(rr - u.pos[0], cc - u.pos[1])
        if bel.bayes:
            score = bel.pp_un[rr, cc] / (1 + 0.1 * d)
        else:
            score = np.where(bel.looked[rr, cc], -np.inf, -d)
        if not len(rr) or not np.isfinite(score.max()):
            return tuple(int(v) for v in b.viewpoints[u.area])
        i = int(np.argmax(score))
        return int(rr[i]), int(cc[i])

    def _start_escort(self, cost):
        w = self.world
        cand = [u for u in self.active() if u.state == "SEARCH"]
        best = min(cand, key=lambda u: (astar(cost, u.pos, [w.person])[1],
                                        math.dist(u.pos, w.person)))
        best.state, best.route, best.path = "APPROACH", [], []
        self.escort = best
        for u in cand:
            if u is not best:
                u.state, u.route, u.path = "EXIT", [], []

    def _act(self, u, cost):
        cfg, w, b = self.cfg, self.world, self.b
        if u.state == "SEARCH":
            goal = self._search_goal(u, cost)
            if goal is None:
                return
            goals = [goal]
        if u.state == "APPROACH":
            pr, pc = w.person
            if max(abs(u.pos[0] - pr), abs(u.pos[1] - pc)) <= 1:
                u.state = "ESCORT"
            else:
                goals = [(r, c) for r in range(pr - 1, pr + 2) for c in range(pc - 1, pc + 2)
                         if not b.solid[r, c]]
        if u.state in ("ESCORT", "EXIT"):
            if u.pos in self.exits:
                if u.state == "ESCORT":
                    self.outcome, self.t_rescue = "rescued", self.t
                u.state = "OUT"
                return
            if u.state == "ESCORT":
                cost = cost_grid(self.belief, b.solid, cfg, w.health, self.m["fuzzy"])
            goals = self.exits
        others = [v.pos for v in self.active() if v is not u]
        if others:
            cost = cost.copy()
            for p in others:
                cost[p] += cfg.unit_cost
        prev = u.pos
        status = self._move(u, cost, goals)
        if status == "noroute":
            u.stuck += 1
            self.belief.relax(self.in_view)   # forget blockages out of view, replan next tick
            if u.state == "SEARCH" and u.stuck >= 2:
                self._finish_area(u)          # still unreachable after relaxing
        else:
            u.stuck = 0
        if status == "moved" and u.state == "ESCORT":
            w.person = prev

    def _move(self, u, cost, goals):
        key = tuple(goals)
        if u.pos in goals:
            return "arrived"
        stale = (len(u.path) < 2 or u.path[0] != u.pos or u.path_goal != key
                 or self.t - u.plan_t >= self.cfg.replan_every
                 or any(not np.isfinite(cost[c]) for c in u.path[1:]))
        if stale:
            path, _ = astar(cost, u.pos, goals)
            self.replans += 1
            u.plan_t, u.path_goal, u.path = self.t, key, path or []
        if len(u.path) < 2:
            return "noroute"
        nxt = u.path[1]
        if self.world.blocked[nxt]:
            u.bumps += 1
            self.belief.mark_blocked(nxt)
            u.path = []
            return "bump"
        u.dist += math.dist(u.pos, nxt)
        u.pos, u.path = nxt, u.path[1:]
        u.trail.append(nxt)
        u.exposure += self.world.h[nxt]
        return "moved"

    # ------------------------------------------------------------------ results
    def metrics(self):
        us, w, nan = self.units, self.world, float("nan")
        return dict(
            seed=self.seed, method=self.method, n_rooms=self.b.n_rooms, k_units=len(us),
            outcome=self.outcome, success=int(self.outcome == "rescued"),
            t_find=nan if self.t_find is None else self.t_find,
            t_rescue=nan if self.t_rescue is None else self.t_rescue,
            health=round(float(w.health), 4),
            escort_loss=nan if self.t_rescue is None else round(float(self.health_find - w.health), 4),
            escort_ticks=nan if self.t_rescue is None else self.t_rescue - self.t_find,
            unit_exposure=round(float(sum(u.exposure for u in us)), 3),
            unit_dist=round(sum(u.dist for u in us), 2),
            bumps=sum(u.bumps for u in us), replans=self.replans, ga_calls=self.ga_calls,
            tick_ms_mean=round(float(np.mean(self.tick_ms)), 3),
            tick_ms_p95=round(float(np.percentile(self.tick_ms, 95)), 3),
            ga_ms_mean=round(float(np.mean(self.ga_ms)), 2) if self.ga_ms else nan,
            blocks=w.n_blocks, ticks=self.t,
        )


def run_episode(seed, method="FULL", n_rooms=10, k_units=3, cfg=None):
    return Mission(seed, method, n_rooms, k_units, cfg).run()
