"""All constants in one place. One cell = 0.5 m, one tick = 1 s, a unit moves one cell per tick.

Fire, blockage and health constants were calibrated on seeds 0-19 only (experiments use other seeds too).
"""
from dataclasses import dataclass


@dataclass
class Config:
    # building
    room_w: int = 8            # room interior, cells
    room_h: int = 6
    corr_w: int = 3            # corridor width, cells
    p_connect: float = 0.5     # connecting door between neighbouring rooms

    # world (ground truth)
    t_pre: int = 60            # fire burns this long before the units arrive
    p_ignite: float = 0.03     # per tick, for a cell next to a burning cell
    diffusion: float = 0.2     # smoke/heat diffusion rate (stable below 0.25)
    decay: float = 0.003
    p_door_blocked0: float = 0.15
    p_block_rate: float = 0.004  # per site per tick, times local hazard
    health_rate: float = 0.006   # person health lost per tick at hazard 1
    t_max: int = 900

    # sensor
    radius: int = 4
    p_hit0: float = 0.9        # P(read blocked | blocked) in clear air
    p_hit_smoke: float = 0.3   # ...minus this times hazard
    p_false: float = 0.1       # P(read blocked | free)
    hazard_sd: float = 0.15
    pd0: float = 0.95          # person detection at distance 0, clear air
    pd_range: float = 5.0

    # belief (Bayes)
    prior_door: float = 0.15
    prior_cell: float = 0.02
    lo_clamp: float = 4.0
    lo_drift: float = 0.01     # log-odds drift back to the prior per tick
    haz_var0: float = 0.25
    haz_q: float = 0.002       # hazard process noise per tick

    # fuzzy cost
    w_penalty: float = 20.0    # cost = 1 + w * penalty
    impassable_p: float = 0.9
    impassable_h: float = 0.9
    unit_cost: float = 5.0     # extra cost on cells holding another unit

    # GA allocation
    pop: int = 60
    gens: int = 100
    stall: int = 25
    p_cx: float = 0.9
    p_mut: float = 0.2
    elite: int = 2
    tourn: int = 3
    dwell: float = 8.0         # expected search time per area, ticks
    mass_eps: float = 0.005    # areas below this P(person) are not allocated

    # mission
    entry_gap: int = 2         # ticks between units passing the entrance
    realloc_every: int = 25
    replan_every: int = 5
    area_done_frac: float = 0.15   # area searched when its mass falls below this fraction
    dwell_max: int = 40
    cover_frac: float = 0.95   # coverage baseline: area cleared when this share of its cells was looked at


# Leave-one-out ablation. Without Bayes the GA still runs, on the uncovered share of each area instead of P(person).
METHODS = {
    "FULL":    dict(fuzzy=True,  bayes=True,  alloc="ga"),
    "noFuzzy": dict(fuzzy=False, bayes=True,  alloc="ga"),       # plain A* (distance only)
    "noBayes": dict(fuzzy=True,  bayes=False, alloc="ga"),       # last reading + coverage search
    "noGA":    dict(fuzzy=True,  bayes=True,  alloc="greedy"),   # greedy P/cost allocation
    "B0":      dict(fuzzy=False, bayes=False, alloc="sweep"),    # none of the three
}
