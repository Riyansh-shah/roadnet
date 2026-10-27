"""Allocating search areas to K units (a multiple-TSP).

Problem: D is (K + M, M). Row k < K is unit k's start, row K + a is area a's viewpoint; column a is area a.
P[a] is the probability that the person is in area a, dwell[a] the time spent searching it, t0[k] when unit k
becomes available. Objective = expected arrival time at the person's area: sum_a P[a] * T_arrive(a).

GA encoding: a permutation of M area tokens (0..M-1) and K-1 delimiter tokens (M..M+K-2). Reading left to right,
each delimiter hands over to the next unit. Order crossover and inversion mutation keep it a permutation.
"""
import numpy as np

BIG = 1e4   # stands in for an unreachable leg


def fitness(pop, D, P, dwell, t0, K):
    """Objective of every chromosome in pop (n, M+K-1), vectorised over the population."""
    M = len(P)
    X = np.atleast_2d(pop)
    isdel = X >= M
    unit = np.cumsum(isdel, axis=1) - isdel                 # delimiters strictly before each position
    prev = np.concatenate([np.full((len(X), 1), M), X[:, :-1]], axis=1)
    prev_area = prev < M
    a = np.where(isdel, 0, X)
    frm = np.where(prev_area, K + np.where(prev_area, prev, 0), np.minimum(unit, K - 1))
    leg = np.where(isdel, 0.0, D[frm, a] + dwell[a])
    tot = np.cumsum(leg, axis=1)
    off = np.maximum.accumulate(np.where(isdel, tot, 0.0), axis=1)   # time used by earlier units
    arrive = tot - off - dwell[a] + t0[np.minimum(unit, K - 1)]
    return np.where(isdel, 0.0, P[a] * arrive).sum(axis=1)


def decode(chrom, M, K):
    routes, k = [[] for _ in range(K)], 0
    for g in chrom:
        if g >= M:
            k += 1
        else:
            routes[k].append(int(g))
    return routes


def encode(routes, M, K, rng):
    """Routes (local area indices) -> chromosome. Areas missing from the routes are inserted at random."""
    routes = [[a for a in r if a < M] for r in routes] + [[] for _ in range(K - len(routes))]
    have = {a for r in routes for a in r}
    for a in rng.permutation(M):
        if a not in have:
            r = routes[rng.integers(K)]
            r.insert(rng.integers(len(r) + 1), int(a))
    chrom = []
    for k, r in enumerate(routes[:K]):
        chrom += r
        if k < K - 1:
            chrom.append(M + k)
    return np.array(chrom)


def _ox(p1, p2, rng):
    L = len(p1)
    i, j = sorted(rng.choice(L, 2, replace=False))
    child = np.empty_like(p1)
    seg = p1[i:j + 1]
    rest = p2[~np.isin(p2, seg)]
    child[i:j + 1] = seg
    child[:i] = rest[:i]
    child[j + 1:] = rest[i:]
    return child


def ga_allocate(D, P, dwell, t0, K, rng, cfg, seed_routes=None):
    """Returns (routes, objective, curve) where curve is a list of (best, mean) per generation."""
    M = len(P)
    D = np.where(np.isfinite(D), D, BIG)
    L = M + K - 1
    if M == 0:
        return [[] for _ in range(K)], 0.0, []
    pop = np.array([rng.permutation(L) for _ in range(cfg.pop)])
    if seed_routes is not None:
        pop[0] = encode(seed_routes, M, K, rng)
    fit = fitness(pop, D, P, dwell, t0, K)
    best_i = int(np.argmin(fit))
    best, best_f = pop[best_i].copy(), fit[best_i]
    curve, stall = [(best_f, fit.mean())], 0
    if L < 2:
        return decode(best, M, K), float(best_f), curve
    n = cfg.pop
    for _ in range(cfg.gens):
        elite = pop[np.argsort(fit)[:cfg.elite]]
        cand = rng.integers(n, size=(2 * (n - cfg.elite), cfg.tourn))
        parents = pop[cand[np.arange(len(cand)), np.argmin(fit[cand], axis=1)]]
        kids = []
        for p1, p2 in zip(parents[0::2], parents[1::2]):
            c = _ox(p1, p2, rng) if rng.random() < cfg.p_cx else p1.copy()
            if rng.random() < cfg.p_mut:
                i, j = sorted(rng.choice(L, 2, replace=False))
                c[i:j + 1] = c[i:j + 1][::-1]
            if rng.random() < cfg.p_mut:
                i, j = rng.choice(L, 2, replace=False)
                c[i], c[j] = c[j], c[i]
            kids.append(c)
        pop = np.vstack([elite, kids])
        fit = fitness(pop, D, P, dwell, t0, K)
        i = int(np.argmin(fit))
        if fit[i] < best_f - 1e-9:
            best, best_f, stall = pop[i].copy(), fit[i], 0
        else:
            stall += 1
        curve.append((best_f, fit.mean()))
        if stall >= cfg.stall:
            break
    return decode(best, M, K), float(best_f), curve


def greedy_allocate(D, P, dwell, t0, K, score="ratio"):
    """The unit that is free first takes the best remaining area: highest P/(travel + dwell) for "ratio",
    nearest for "nearest" (the sweep baseline)."""
    M = len(P)
    D = np.where(np.isfinite(D), D, BIG)
    node, time = list(range(K)), np.array(t0, float)
    routes, left = [[] for _ in range(K)], list(range(M))
    while left:
        k = int(np.argmin(time))
        d = D[node[k], left]
        j = int(np.argmax(P[left] / (d + dwell[left])) if score == "ratio" else np.argmin(d))
        a = left.pop(j)
        routes[k].append(a)
        time[k] += D[node[k], a] + dwell[a]
        node[k] = K + a
    return routes


def routes_objective(routes, D, P, dwell, t0, K, rng):
    D = np.where(np.isfinite(D), D, BIG)
    return float(fitness(encode(routes, len(P), K, rng), D, P, dwell, t0, K)[0])


def random_objective(D, P, dwell, t0, K, rng, n=200):
    D = np.where(np.isfinite(D), D, BIG)
    pop = np.array([rng.permutation(len(P) + K - 1) for _ in range(n)])
    return float(fitness(pop, D, P, dwell, t0, K).mean())
