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
    n_exits: int = 1           # 1: the entrance is the only way out; 3: also an east-end and a south side exit

    # world (ground truth)
    t_pre: int = 0             # the drones arrive as the fire starts
    fire_start: int = 2        # the fire starts as a fire_start x fire_start block of cells (2 x 2 = 4 cells)
    p_ignite: float = 0.03     # per tick, for a cell next to a burning cell
    diffusion: float = 0.2     # smoke/heat diffusion rate (stable below 0.25)
    decay: float = 0.003
    debris: bool = False       # falling debris that blocks doors/corridor sections (off: only fire and smoke)
    p_door_blocked0: float = 0.15  # with debris: share of doors blocked at the start
    p_block_rate: float = 0.004  # with debris: per site per tick, times local hazard
    health_rate: float = 0.006   # person health lost per tick at hazard 1
    t_max: int = 900
    # winnable-scenario check (navsim/scenario.py): an all-knowing rescuer starting this late must still succeed
    oracle_delay: int = 120
    oracle_min_health: float = 0.3
    oracle_max_time: int = 450

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
    realloc_gap: int = 5       # at least this many ticks between event-triggered reallocations
    replan_every: int = 5
    area_done_frac: float = 0.15   # area searched when its mass falls below this fraction
    dwell_max: int = 40
    unreach_ticks: int = 30    # an area with no route is skipped this long
    cover_frac: float = 0.95   # coverage baseline: area cleared when this share of its cells was looked at


# The final system is FULL: Bayesian search map + GA allocation + A* (octile heuristic). The other entries exist
# only for the evaluation that justifies it: each one removes a component, and "withFuzzy" adds the fuzzy cost
# that was built, tested and left out because it gave no measurable gain.
METHODS = {
    "FULL":      dict(fuzzy=False, bayes=True,  alloc="ga"),
    "withFuzzy": dict(fuzzy=True,  bayes=True,  alloc="ga"),       # + Mamdani fuzzy cell cost (tested, rejected)
    "noBayes":   dict(fuzzy=False, bayes=False, alloc="ga"),       # last reading only + coverage search
    "noGA":      dict(fuzzy=False, bayes=True,  alloc="greedy"),   # greedy P/cost allocation
    "B0":        dict(fuzzy=False, bayes=False, alloc="sweep"),    # none: coverage search, nearest room first
    # the non-AI approach shown in the app: room-by-room sweep, nearest room first, plain shortest paths (Dijkstra)
    "NONAI":     dict(fuzzy=False, bayes=False, alloc="sweep", planner="dijkstra"),
}
VISIBILITY = {"Clear": 0.95, "Smoky": 0.7, "Dense smoke": 0.5}   # chance one look spots the person up close (pd0)
