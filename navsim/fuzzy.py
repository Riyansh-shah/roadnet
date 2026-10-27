"""Mamdani fuzzy inference system for the traversal cost of a cell (written from scratch).

Inputs  : Hazard (belief smoke/heat), Blocked (belief P(blocked)), Uncertainty (how stale the hazard estimate is),
          Resilience of whoever crosses the cell (1 for a unit in protective gear; 0.5 x health for the civilian,
          who has no breathing apparatus).
Output  : Penalty in [0, 1]; cell cost = 1 + w * Penalty.
Engine  : min for AND, clipping implication, max aggregation, centroid defuzzification.
For real-time use the FIS is compiled once into a 4-D lookup table that is interpolated with numpy.
"""
import itertools
from functools import lru_cache

import numpy as np


def trap(x, a, b, c, d):
    x = np.asarray(x, float)
    up = np.ones_like(x) if b == a else np.clip((x - a) / (b - a), 0, 1)
    down = np.ones_like(x) if d == c else np.clip((d - x) / (d - c), 0, 1)
    return np.minimum(up, down)


HAZARD = {"Low": (0, 0, 0.15, 0.35), "Med": (0.2, 0.45, 0.45, 0.7), "High": (0.55, 0.8, 1, 1)}
BLOCKED = {"Low": (0, 0, 0.1, 0.3), "Med": (0.2, 0.45, 0.45, 0.7), "High": (0.55, 0.8, 1, 1)}
UNCERT = {"Known": (0, 0, 0.3, 0.7), "Unknown": (0.3, 0.7, 1, 1)}
RESIL = {"Low": (0, 0, 0.3, 0.7), "High": (0.3, 0.7, 1, 1)}
PENALTY = {"VLow": (0, 0, 0.05, 0.2), "Low": (0.05, 0.2, 0.2, 0.4), "Med": (0.25, 0.45, 0.45, 0.65),
           "High": (0.5, 0.7, 0.7, 0.85), "VHigh": (0.75, 0.9, 1, 1)}
INPUTS = {"Hazard": HAZARD, "Blocked": BLOCKED, "Uncertainty": UNCERT, "Resilience": RESIL}

# (Hazard, Blocked, Uncertainty) -> Penalty; columns are Blocked = Low, Med, High.
# Units wear protective gear: smoke alone only costs Low/Med, a possibly blocked door costs a lot (a dead end
# wastes time), and unexplored cells are free to enter. A stale smoke reading is assumed to have got worse.
_BASE = {
    "Known": {"Low": ("VLow", "Med", "VHigh"),
              "Med": ("Low", "High", "VHigh"),
              "High": ("Med", "High", "VHigh")},
    "Unknown": {"Low": ("VLow", "Med", "VHigh"),
                "Med": ("Med", "High", "VHigh"),
                "High": ("High", "VHigh", "VHigh")},
}
RULES = [({"Hazard": hz, "Blocked": bl, "Uncertainty": un}, out)
         for un, row in _BASE.items() for hz, outs in row.items() for bl, out in zip(("Low", "Med", "High"), outs)]
# The civilian being escorted has no protection: smoke matters much more.
RULES += [({"Resilience": "Low", "Hazard": "Med"}, "High"),
          ({"Resilience": "Low", "Hazard": "High"}, "VHigh")]

Y = np.linspace(0, 1, 101)
_OUT_MF = {k: trap(Y, *v) for k, v in PENALTY.items()}


def infer(hazard, blocked, unc, resil):
    """Direct Mamdani inference on arrays of equal shape (or scalars). Returns the defuzzified penalty."""
    x = {"Hazard": np.asarray(hazard, float), "Blocked": np.asarray(blocked, float),
         "Uncertainty": np.asarray(unc, float), "Resilience": np.asarray(resil, float)}
    shape = np.broadcast(*x.values()).shape
    x = {k: np.broadcast_to(v, shape).ravel() for k, v in x.items()}
    mu = {(name, term): trap(x[name], *p) for name, terms in INPUTS.items() for term, p in terms.items()}
    agg = np.zeros((x["Hazard"].size, Y.size))
    for ante, out in RULES:
        w = np.min([mu[(k, v)] for k, v in ante.items()], axis=0)
        np.maximum(agg, np.minimum(w[:, None], _OUT_MF[out][None, :]), out=agg)
    s = agg.sum(axis=1)
    pen = np.where(s > 0, (agg * Y).sum(axis=1) / np.maximum(s, 1e-12), 0.0)
    return pen.reshape(shape)


LUT_SHAPE = (21, 21, 6, 6)   # hazard, blocked, uncertainty, resilience


@lru_cache(maxsize=1)
def lut():
    axes = [np.linspace(0, 1, n) for n in LUT_SHAPE]
    grid = np.meshgrid(*axes, indexing="ij")
    return infer(*grid)


def penalty(hazard, blocked, unc, resil):
    """Multilinear interpolation in the compiled lookup table (same result as infer, much faster)."""
    T = lut()
    coords = [np.clip(np.asarray(v, float), 0, 1) * (n - 1) for v, n in zip((hazard, blocked, unc, resil), T.shape)]
    shape = np.broadcast(*coords).shape
    coords = [np.broadcast_to(c, shape) for c in coords]
    i0 = [np.minimum(np.floor(c).astype(int), n - 2) for c, n in zip(coords, T.shape)]
    fr = [c - i for c, i in zip(coords, i0)]
    out = np.zeros(shape)
    for corner in itertools.product((0, 1), repeat=4):
        w = np.ones(shape)
        for f, bit in zip(fr, corner):
            w = w * (f if bit else 1 - f)
        out += w * T[tuple(i + bit for i, bit in zip(i0, corner))]
    return out


def cost_grid(belief, solid, cfg, resil=1.0, fuzzy=True):
    """Per-cell traversal cost. inf = impassable (wall, believed blocked, believed on fire)."""
    if fuzzy:
        cost = 1 + cfg.w_penalty * penalty(belief.hazard, belief.p_block, belief.unc, resil)
    else:
        cost = np.ones(solid.shape)
    cost[solid] = np.inf
    cost[(belief.p_block > cfg.impassable_p) | (belief.hazard > cfg.impassable_h)] = np.inf
    return cost
