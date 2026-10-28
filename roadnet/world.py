"""The map: houses that must all be connected by road, and lakes where road costs more.

Everything is in metres. The map is SIZE x SIZE. A lake is a circle (cx, cy, r); a metre of road built across a
lake (a causeway) costs LAKE_COST metres of land road, so good networks bend around lakes.
"""
import math
from dataclasses import dataclass

import numpy as np

SIZE = 1200.0
LAKE_COST = 3.0
MIN_HOUSES, MAX_HOUSES = 3, 20


@dataclass
class World:
    houses: np.ndarray                                              # (n, 2) house positions
    lakes: np.ndarray | None = None                                 # (L, 3) circles cx, cy, r
    name: str = "custom"
    optimum: float | None = None                                    # known shortest network, for the presets

    def __post_init__(self):
        self.houses = np.asarray(self.houses, dtype=float).reshape(-1, 2)
        self.lakes = np.zeros((0, 3)) if self.lakes is None else np.asarray(self.lakes, dtype=float).reshape(-1, 3)

    @property
    def n(self):
        return len(self.houses)

    def in_lake(self, p, margin=0.0):
        if not len(self.lakes):
            return False
        d = np.hypot(self.lakes[:, 0] - p[0], self.lakes[:, 1] - p[1])
        return bool(np.any(d < self.lakes[:, 2] + margin))

    def copy(self):
        return World(self.houses.copy(), self.lakes.copy(), self.name, self.optimum)


def triangle(side=1000.0):
    """Equilateral triangle. Best network: one junction in the middle, 3 roads at 120 deg, length sqrt(3) * side."""
    h = side * math.sqrt(3) / 2
    x0, y0 = (SIZE - side) / 2, (SIZE - h) / 2
    houses = [(x0, y0), (x0 + side, y0), (x0 + side / 2, y0 + h)]
    return World(houses, name="triangle", optimum=math.sqrt(3) * side)


def square(side=1000.0):
    """Square. The X through the centre is 2*sqrt(2)*side; the best is an H with two junctions, (1 + sqrt(3)) * side."""
    a, b = (SIZE - side) / 2, (SIZE + side) / 2
    houses = [(a, a), (b, a), (b, b), (a, b)]
    return World(houses, name="square", optimum=(1 + math.sqrt(3)) * side)


def random_lakes(rng, count):
    lakes = []
    while len(lakes) < count:
        r = rng.uniform(90, 190)
        c = rng.uniform(250, SIZE - 250, size=2)
        if all(math.hypot(c[0] - x, c[1] - y) > r + rr + 60 for x, y, rr in lakes):
            lakes.append((c[0], c[1], r))
    return np.array(lakes)


def random_world(n=10, seed=0, lakes=False):
    """n houses spread over the map, not in a lake and not on top of each other. Same seed -> same map."""
    rng = np.random.default_rng(seed)
    world = World(np.zeros((0, 2)), random_lakes(rng, int(rng.integers(2, 4))) if lakes else None, name=f"random #{seed}")
    houses = []
    while len(houses) < n:
        p = rng.uniform(80, SIZE - 80, size=2)
        if world.in_lake(p, margin=40):
            continue
        if any(math.hypot(p[0] - q[0], p[1] - q[1]) < 110 for q in houses):
            continue
        houses.append(p)
    world.houses = np.array(houses)
    return world


def add_lakes(world, seed=0):
    """The same houses with lakes added in the gaps between them (lakes never cover a house)."""
    rng = np.random.default_rng(seed + 7919)
    out = world.copy()
    for attempt in range(300):
        lakes = random_lakes(rng, 3 - attempt // 100)
        d = np.hypot(world.houses[:, None, 0] - lakes[None, :, 0], world.houses[:, None, 1] - lakes[None, :, 1])
        if np.all(d > lakes[None, :, 2] + 40):
            out.lakes = lakes
            return out
    return out
