"""Short-range noisy sensor: line of sight within a disc, readings of blockage and hazard, person detection."""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


def bresenham(r0, c0, r1, c1):
    cells = []
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    r, c = r0, c0
    while (r, c) != (r1, c1):
        cells.append((r, c))
        e2 = 2 * err
        if e2 >= -dr:
            err -= dr
            c += sc
        if e2 <= dc:
            err += dc
            r += sr
    return cells[1:]  # intermediate cells only


@lru_cache(maxsize=None)
def rays(radius):
    """Disc offsets, their distances, and the padded intermediate cells of each ray (padding = own cell)."""
    offs = [(dr, dc) for dr in range(-radius, radius + 1) for dc in range(-radius, radius + 1)
            if dr * dr + dc * dc <= radius * radius]
    inter = [bresenham(0, 0, dr, dc) for dr, dc in offs]
    L = max(1, max(len(x) for x in inter))
    IR = np.zeros((len(offs), L), int)
    IC = np.zeros((len(offs), L), int)
    for i, x in enumerate(inter):
        for j, (r, c) in enumerate(x):
            IR[i, j], IC[i, j] = r, c
    offs = np.array(offs)
    return offs, np.hypot(offs[:, 0], offs[:, 1]), IR, IC


def visible(opaque, pos, radius):
    """Cells in the disc whose ray is not stopped by an opaque cell. Returns rows, cols, distances."""
    offs, dist, IR, IC = rays(radius)
    p = np.pad(opaque, radius, constant_values=True)
    r, c = pos[0] + radius, pos[1] + radius
    vis = ~p[r + IR, c + IC].any(axis=1)
    rr, cc = pos[0] + offs[vis, 0], pos[1] + offs[vis, 1]
    H, W = opaque.shape
    inb = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    return rr[inb], cc[inb], dist[vis][inb]


@dataclass
class Obs:
    r: np.ndarray
    c: np.ndarray
    d: np.ndarray
    blk: np.ndarray       # bool reading "blocked"
    z: np.ndarray         # hazard reading
    detected: bool


def observe(world, pos, rng, cfg):
    b = world.b
    rr, cc, d = visible(b.solid | world.blocked, pos, cfg.radius)
    keep = ~b.solid[rr, cc]
    rr, cc, d = rr[keep], cc[keep], d[keep]
    h = world.h[rr, cc]
    p_hit = cfg.p_hit0 - cfg.p_hit_smoke * h
    u = rng.random(len(rr))
    blk = np.where(world.blocked[rr, cc], u < p_hit, u < cfg.p_false) if cfg.debris else np.zeros(len(rr), bool)
    z = np.clip(h + rng.normal(0.0, cfg.hazard_sd, len(rr)), 0.0, 1.0)
    detected = False
    hit = np.flatnonzero((rr == world.person[0]) & (cc == world.person[1]))
    if hit.size:
        i = hit[0]
        pd = cfg.pd0 * np.exp(-d[i] / cfg.pd_range) * (1 - 0.5 * h[i])
        detected = bool(rng.random() < pd)
    return Obs(rr, cc, d, blk, z, detected)
