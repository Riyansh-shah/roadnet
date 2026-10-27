"""Floor plan: one corridor with N rooms on both sides and three exits.

Arrays are indexed [row, col]. Row 0 is the north outer wall. The floor plan (walls, furniture, door positions)
is known to the units; whether a door is passable is not.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np

FURN_SHAPES = [(1, 2), (2, 1), (2, 2), (1, 3), (2, 3)]


@dataclass
class Building:
    solid: np.ndarray       # (H, W) bool: walls + furniture
    furn: np.ndarray        # (H, W) bool: furniture (subset of solid)
    door: np.ndarray        # (H, W) bool: internal doors, may become blocked
    room_id: np.ndarray     # (H, W) int, -1 outside rooms
    area_id: np.ndarray     # (H, W) int: rooms 0..N-1, corridor segments N.., -1 elsewhere
    viewpoints: np.ndarray  # (A, 2) one central free cell per area
    exits: list             # [(r, c)]; exits[0] is the entrance
    cut_cols: list          # corridor columns where debris can block the corridor
    corr_rows: tuple        # (first, last) corridor row
    n_rooms: int

    @property
    def entrance(self):
        return self.exits[0]

    @property
    def shape(self):
        return self.solid.shape

    @property
    def n_areas(self):
        return len(self.viewpoints)

    def area_name(self, a):
        return f"R{a}" if a < self.n_rooms else f"C{a - self.n_rooms}"


def reachable(open_, src):
    """4-connected flood fill from src over open cells."""
    H, W = open_.shape
    seen = np.zeros_like(open_)
    seen[src] = True
    q = deque([src])
    while q:
        r, c = q.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < H and 0 <= cc < W and open_[rr, cc] and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def make_building(n_rooms, rng, cfg):
    assert n_rooms >= 2
    rw, rh, cw = cfg.room_w, cfg.room_h, cfg.corr_w
    nn, ns = (n_rooms + 1) // 2, n_rooms // 2
    W = nn * (rw + 1) + 1
    H = 2 * rh + cw + 4
    cr0 = rh + 2
    cr1 = cr0 + cw - 1
    south_r0 = cr1 + 2

    solid = np.ones((H, W), bool)
    room_id = -np.ones((H, W), int)
    door = np.zeros((H, W), bool)
    furn = np.zeros((H, W), bool)
    solid[cr0:cr1 + 1, 1:W - 1] = False

    rooms = [(1, 1 + i * (rw + 1), "N") for i in range(nn)] + \
            [(south_r0, 1 + i * (rw + 1), "S") for i in range(ns)]
    for k, (r0, c0, _) in enumerate(rooms):
        solid[r0:r0 + rh, c0:c0 + rw] = False
        room_id[r0:r0 + rh, c0:c0 + rw] = k

    # one corridor door per room, away from the room corners
    for r0, c0, side in rooms:
        c = c0 + rng.integers(1, rw - 1)
        r = r0 + rh if side == "N" else r0 - 1
        solid[r, c] = False
        door[r, c] = True
    # connecting doors between neighbouring rooms on the same side
    for side in "NS":
        same = [k for k, room in enumerate(rooms) if room[2] == side]
        for a, b in zip(same, same[1:]):
            if rng.random() < cfg.p_connect:
                r0, c0, _ = rooms[b]
                r, c = r0 + rng.integers(1, rh - 1), c0 - 1
                solid[r, c] = False
                door[r, c] = True

    # furniture stays off the room's outer ring, so every room stays walkable around its edge
    for r0, c0, _ in rooms:
        for _ in range(rng.integers(1, 3)):
            fh, fw = FURN_SHAPES[rng.integers(len(FURN_SHAPES))]
            r = r0 + 1 + rng.integers(0, rh - 2 - fh + 1)
            c = c0 + 1 + rng.integers(0, rw - 2 - fw + 1)
            furn[r:r + fh, c:c + fw] = True
    solid |= furn

    mid = cr0 + cw // 2
    exits = [(mid, 0), (mid, W - 1)]
    if ns:
        r0, c0, _ = rooms[nn + ns // 2]
        exits.append((r0 + rh, c0 + rw // 2))
    for e in exits:
        solid[e] = False

    # seal any pocket the furniture cut off
    pocket = ~solid & ~reachable(~solid, exits[0])
    furn |= pocket
    solid |= pocket

    area_id = room_id.copy()
    for i in range(nn):
        c0 = 1 + i * (rw + 1)
        area_id[cr0:cr1 + 1, c0:min(c0 + rw + 1, W - 1)] = n_rooms + i
    area_id[solid] = -1

    viewpoints = []
    for a in range(n_rooms + nn):
        cells = np.argwhere(area_id == a)
        centre = cells.mean(axis=0)
        viewpoints.append(cells[np.argmin(((cells - centre) ** 2).sum(axis=1))])

    cut_cols = [i * (rw + 1) for i in range(1, nn)]
    return Building(solid, furn, door, room_id, area_id, np.array(viewpoints), exits, cut_cols,
                    (cr0, cr1), n_rooms)
