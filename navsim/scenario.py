"""Scenario generation that only produces winnable missions.

A scenario is kept only if an omniscient rescuer could save the person even when it starts late. That rescuer knows
the fire, the debris and where the person is, and moves one cell per tick through cells that are neither burning nor
blocked. Otherwise the next attempt for the same scenario number is tried, so scenario #n is always the same
building, fire and person.

The rescuer that starts at t = 0 also gives the reference time used by the efficiency metric:
efficiency = oracle rescue time / actual rescue time (100% = as fast as an all-knowing rescuer).
"""
import copy
from dataclasses import dataclass

import numpy as np

from .building import make_building
from .world import World


def _dilate(m, diagonal):
    """One move (staying put allowed): 4-connected, or 8-connected when diagonal is True."""
    p = np.pad(m, 1)
    out = m | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    if diagonal:
        out |= p[:-2, :-2] | p[:-2, 2:] | p[2:, :-2] | p[2:, 2:]
    return out


def oracle(world, t_max, runs=((0, True),)):
    """Omniscient rescue on a copy of the world. Each run is (start delay, diagonal moves allowed).
    Returns {run: (t_find, t_rescue, person health)} for the runs that succeed.
    A run finds the person once it stands next to them, like a drone, and then leads them back to the entrance."""
    w = copy.deepcopy(world)
    b = w.b
    reach = {r: np.zeros(b.shape, bool) for r in runs}
    found, out = {}, {}
    for t in range(t_max):
        w.step()
        free = ~b.solid & ~w.blocked & ~w.burning
        near = np.zeros(b.shape, bool)
        near[w.person] = True
        near = _dilate(near, True)
        for r in runs:
            d, diag = r
            if r in out or t < d:
                continue
            if t == d:
                reach[r][b.entrance] = True
            reach[r] = _dilate(reach[r], diag) & free
            if r not in found:
                if (reach[r] & near).any():           # next to the person: from here on, lead them out
                    found[r] = t + 1                  # stamped at the end of the tick, like the mission
                    reach[r] = reach[r] & near
            elif reach[r][b.entrance]:
                out[r] = (found[r], t + 1, w.health)
        if w.health <= 0 or len(out) == len(runs):
            break
    return out


@dataclass
class Scenario:
    building: object
    world: object
    sensor_seed: object
    ga_seed: object
    oracle_find: int
    oracle_rescue: int
    attempts: int


def make_scenario(number, n_rooms, cfg, max_attempts=50):
    """Scenario #number: the first attempt whose late-starting oracle still rescues the person with margin."""
    for attempt in range(max_attempts):
        ws, ss, gs = np.random.SeedSequence([number, attempt]).spawn(3)
        rng = np.random.default_rng(ws)
        b = make_building(n_rooms, rng, cfg)
        w = World(b, cfg, rng)
        best, late = (0, True), (cfg.oracle_delay, False)    # reference: earliest possible; guarantee: late, 4-way
        res = oracle(w, cfg.t_max, runs=(best, late))
        ok = res.get(late)
        if ok and best in res and ok[2] >= cfg.oracle_min_health and ok[1] <= cfg.oracle_max_time:
            find0, rescue0, _ = res[best]
            return Scenario(b, w, ss, gs, find0, rescue0, attempt + 1)
    raise RuntimeError(f"no winnable scenario found for #{number} with {n_rooms} rooms")
