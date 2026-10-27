"""Ground truth: spreading fire and smoke, doors and corridor sections that get blocked, the person's health."""
import numpy as np

from .building import reachable


def neighbours(a, fill):
    """The four neighbours of every cell (north, south, west, east), padded with fill."""
    p = np.pad(a, 1, constant_values=fill)
    return p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]


class World:
    def __init__(self, b, cfg, rng):
        self.b, self.cfg, self.rng = b, cfg, rng
        self.h = np.zeros(b.shape)          # smoke/heat hazard in [0, 1]
        self.burning = np.zeros(b.shape, bool)
        self.blocked = np.zeros(b.shape, bool)
        self.exit_mask = np.zeros(b.shape, bool)
        for e in b.exits:
            self.exit_mask[e] = True
        self.t = 0
        self.n_blocks = 0

        self.fire_room = int(rng.integers(b.n_rooms))
        fire_cells = np.argwhere((b.room_id == self.fire_room) & ~b.solid)
        self.burning[tuple(fire_cells[rng.integers(len(fire_cells))])] = True

        ok = (b.area_id >= 0) & (b.room_id != self.fire_room)
        cand = np.argwhere(ok)
        self.person = tuple(int(v) for v in cand[rng.integers(len(cand))])
        self.health = 1.0

        r0, r1 = b.corr_rows
        self.sites = [[tuple(d)] for d in np.argwhere(b.door)] + \
                     [[(r, c) for r in range(r0, r1 + 1)] for c in b.cut_cols]
        for site in self.sites:
            if len(site) == 1 and rng.random() < cfg.p_door_blocked0:
                self._try_block(site, ())
        self.n_blocks = 0
        for _ in range(cfg.t_pre):
            self.step()

    def _try_block(self, site, occupied):
        """Block a site unless a unit stands there or it would cut the person off from the entrance."""
        if any(c in occupied or c == self.person for c in site):
            return False
        for c in site:
            self.blocked[c] = True
        if not reachable(~self.b.solid & ~self.blocked, self.b.entrance)[self.person]:
            for c in site:
                self.blocked[c] = False
            return False
        self.n_blocks += 1
        return True

    def step(self, occupied=()):
        cfg, rng = self.cfg, self.rng
        open_ = ~self.b.solid & ~self.blocked

        nb_burn = sum(n.astype(int) for n in neighbours(self.burning, False))
        ignite = open_ & ~self.burning & ~self.exit_mask & (nb_burn > 0)
        self.burning |= ignite & (rng.random(self.h.shape) < cfg.p_ignite)

        h = self.h
        lap = np.zeros_like(h)
        for hn, on in zip(neighbours(h, 0.0), neighbours(open_, False)):
            lap += on * (hn - h)
        h = (h + cfg.diffusion * lap * open_) * (1 - cfg.decay)
        h[self.burning] = 1.0
        h[self.exit_mask] = 0.0
        h[~open_] = 0.0
        self.h = h

        u = rng.random(len(self.sites))
        for site, x in zip(self.sites, u):
            if not self.blocked[site[0]]:
                hs = max(h[c] for c in site)
                if x < cfg.p_block_rate * hs:
                    self._try_block(site, occupied)

        self.health = max(0.0, self.health - cfg.health_rate * h[self.person])
        self.t += 1
