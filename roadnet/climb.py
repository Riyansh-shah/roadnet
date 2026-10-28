"""Hill climbing over junction positions.

State      a set of junction points (at most n - 2 for n houses).
Score      the cost of the MST over houses + junctions (lower is better).
Neighbour  one small random change: nudge a junction, add a junction, or remove a junction.

Every climber is a generator that yields a Snapshot after each evaluation, so the app can animate it, and every climber
gets the same evaluation budget so the race is fair.
"""
from dataclasses import dataclass

import numpy as np

from .network import cost_matrix, degrees, mst, segment_cost
from .world import SIZE

STEP0 = 60.0        # first nudge size (m)
MIN_STEP = 1.0      # below this the climber has converged
PATIENCE = 60       # failed tries before the nudge size halves
P_ADD, P_REMOVE = 0.15, 0.15


def budget_for(n):
    """Evaluations each climber may spend on a map with n houses."""
    return 300 * n


@dataclass
class State:
    junctions: np.ndarray   # (k, 2)
    cost: float
    edges: np.ndarray       # MST edges; nodes 0..n-1 are houses, n.. are junctions


@dataclass
class Snapshot:
    current: State          # what the climber is looking at now
    best: State             # best seen so far
    evals: int
    status: str
    restarts: int = 0
    done: bool = False


class Problem:
    """Scores junction sets for one map and counts evaluations."""

    def __init__(self, world):
        self.houses, self.lakes, self.n = world.houses, world.lakes, world.n
        self.hh = cost_matrix(self.houses, self.lakes)
        self.evals = 0

    def points(self, junctions):
        return np.vstack([self.houses, junctions])

    def evaluate(self, junctions):
        """Cost of the network using these junctions. Dead-end junctions (only one road) only add road, so drop them."""
        J = np.asarray(junctions, float).reshape(-1, 2)
        n = self.n
        while True:
            self.evals += 1
            k = len(J)
            C = np.empty((n + k, n + k))
            C[:n, :n] = self.hh
            if k:
                hj = segment_cost(self.houses[:, None], J[None], self.lakes)
                C[:n, n:], C[n:, :n] = hj, hj.T
                C[n:, n:] = segment_cost(J[:, None], J[None], self.lakes)
            cost, edges = mst(C)
            dead = np.flatnonzero(degrees(edges, n + k)[n:] <= 1)
            if not len(dead):
                return State(J, cost, edges)
            J = np.delete(J, dead, axis=0)


def add_point(problem, state, rng):
    """A promising new junction: the centre of a node and two of its road neighbours (where a 'V' can be shortened)."""
    P = problem.points(state.junctions)
    deg = degrees(state.edges, len(P))
    v = rng.choice(np.flatnonzero(deg >= 2))
    nbrs = np.concatenate([state.edges[state.edges[:, 0] == v, 1], state.edges[state.edges[:, 1] == v, 0]])
    a, b = rng.choice(nbrs, 2, replace=False)
    return (P[v] + P[a] + P[b]) / 3


def neighbour(problem, state, step, rng):
    """One random change to the junctions, scored."""
    J, k = state.junctions, len(state.junctions)
    r = rng.random()
    if k == 0 or (r < P_ADD and k < problem.n - 2):
        J2 = np.vstack([J, add_point(problem, state, rng)])
    elif r < P_ADD + P_REMOVE:
        J2 = np.delete(J, rng.integers(k), axis=0)
    else:
        J2 = J.copy()
        i = rng.integers(k)
        J2[i] = np.clip(J2[i] + rng.normal(0, step, 2), 0, SIZE)
    return problem.evaluate(J2)


def climb(problem, state, rng, budget):
    """Plain hill climbing from `state`: take a random neighbour only if it is shorter.

    Yields (state, step, stuck). When `patience` tries in a row fail, the nudge size halves; once it is below MIN_STEP no
    small change helps any more, which means the climber is at a local optimum ("stuck").
    """
    step, fails = STEP0, 0
    while problem.evals < budget:
        cand = neighbour(problem, state, step, rng)
        if cand.cost < state.cost - 1e-6:
            state, fails = cand, 0
        else:
            fails += 1
            if fails >= PATIENCE:
                step, fails = step / 2, 0
                if step < MIN_STEP:
                    yield state, step, True
                    return
        yield state, step, False


def simple_hc(world, budget=None, seed=0):
    """Simple hill climbing from the no-junction network. Stops at the first local optimum it reaches."""
    problem = Problem(world)
    budget = budget or budget_for(world.n)
    state = problem.evaluate(np.zeros((0, 2)))
    if problem.n < 3:
        yield Snapshot(state, state, problem.evals, "nothing to improve", done=True)
        return
    rng = np.random.default_rng(seed)
    for state, step, stuck in climb(problem, state, rng, budget):
        if stuck:
            yield Snapshot(state, state, problem.evals, "stuck: local optimum", done=True)
            return
        yield Snapshot(state, state, problem.evals, f"climbing, step {step:.0f} m")
    yield Snapshot(state, state, problem.evals, "budget used", done=True)


def random_start(problem, rng):
    """A random set of junctions: each sits in the middle of three random houses, shaken a little."""
    k = int(rng.integers(1, problem.n - 1))
    trios = np.array([rng.choice(problem.n, 3, replace=False) for _ in range(k)])
    J = problem.houses[trios].mean(axis=1) + rng.normal(0, STEP0, (k, 2))
    return problem.evaluate(np.clip(J, 0, SIZE))


def random_restart_hc(world, budget=None, seed=0):
    """Hill climbing that restarts from a random set of junctions each time it gets stuck, keeping the best network.

    Restart 1 starts from the no-junction network, exactly like simple hill climbing; the extra climbs use the budget
    simple hill climbing leaves unused when it stops.
    """
    problem = Problem(world)
    budget = budget or budget_for(world.n)
    state = best = problem.evaluate(np.zeros((0, 2)))
    if problem.n < 3:
        yield Snapshot(state, state, problem.evals, "nothing to improve", done=True)
        return
    rng = np.random.default_rng(seed)
    restart = 1
    while problem.evals < budget:
        for state, step, stuck in climb(problem, state, rng, budget):
            if state.cost < best.cost:
                best = state
            yield Snapshot(state, best, problem.evals, f"climb {restart}, " + ("stuck, restarting" if stuck else
                           f"step {step:.0f} m"), restarts=restart - 1)
        if problem.evals < budget:
            restart += 1
            state = random_start(problem, rng)
    yield Snapshot(best, best, problem.evals, f"best of {restart} climbs", restarts=restart - 1, done=True)


CLIMBERS = {"Simple hill climbing": simple_hc, "Random-restart hill climbing": random_restart_hc}


def run(gen):
    """Run a climber to the end and return its last snapshot."""
    snap = None
    for snap in gen:
        pass
    return snap
