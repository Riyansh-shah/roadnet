"""What the team believes, shared by all units.

BeliefMap (Bayes):
  * P(person) per cell: Bayesian search theory. A look at cell c that finds nobody multiplies its prior by
    (1 - p_d(c)), and the map is renormalised. Cells missed in smoke keep part of their probability.
  * P(blocked) per cell: binary Bayes filter in log-odds. Its predict step is the world's transition model:
    a door or corridor section can become blocked (more likely in smoke) but never clears.
  * Hazard per cell: 1-D Kalman filter (Gaussian mean/variance), variance grows while a cell is unseen.
RawMap (ablation without Bayes): last reading only, and room-by-room "searched" flags.
"""
import numpy as np


def logit(p):
    return np.log(p / (1 - p))


def _prior_block(b, cfg):
    p = np.full(b.shape, cfg.prior_cell)
    p[b.door] = cfg.prior_door
    for e in b.exits:
        p[e] = 1e-3
    return p


class BeliefMap:
    bayes = True

    def __init__(self, b, cfg):
        self.b, self.cfg = b, cfg
        self.lo0 = logit(_prior_block(b, cfg))
        self.lo = self.lo0.copy()
        self.confirmed = np.zeros(b.shape, bool)     # bumped into: certain, and blockages never clear
        self.site = b.door.copy()                    # cells that can become blocked
        r0, r1 = b.corr_rows
        self.site[r0:r1 + 1, b.cut_cols] = True
        self.mu = np.zeros(b.shape)
        self.var = np.full(b.shape, cfg.haz_var0)
        self.searchable = b.area_id >= 0
        self.pp_un = self.searchable / self.searchable.sum()   # unnormalised posterior (= prior x misses)
        self._scale = 1.0                                       # undoes the rare underflow rescaling below
        self._area_idx = b.area_id[self.searchable]

    @property
    def p_block(self):
        return np.where(self.confirmed, 1.0, 1 / (1 + np.exp(-self.lo)))

    @property
    def hazard(self):
        return self.mu

    @property
    def unc(self):
        return np.sqrt(self.var / self.cfg.haz_var0)

    @property
    def pp(self):
        return self.pp_un / self.pp_un.sum()

    def predict(self):
        cfg = self.cfg
        p = self.p_block
        p = p + (1 - p) * cfg.p_block_rate * self.mu * self.site     # P(free -> blocked) this tick
        self.lo = np.clip(logit(np.clip(p, 1e-6, 1 - 1e-6)), -cfg.lo_clamp, cfg.lo_clamp)
        np.minimum(self.var + cfg.haz_q, cfg.haz_var0, out=self.var)

    def update(self, o):
        cfg, r, c = self.cfg, o.r, o.c
        p_hit = cfg.p_hit0 - cfg.p_hit_smoke * self.mu[r, c]
        inc = np.where(o.blk, np.log(p_hit / cfg.p_false), np.log((1 - p_hit) / (1 - cfg.p_false)))
        self.lo[r, c] = np.clip(self.lo[r, c] + inc, -cfg.lo_clamp, cfg.lo_clamp)

        k = self.var[r, c] / (self.var[r, c] + cfg.hazard_sd ** 2)
        self.mu[r, c] += k * (o.z - self.mu[r, c])
        self.var[r, c] *= 1 - k

        if not o.detected:
            pd = cfg.pd0 * np.exp(-o.d / cfg.pd_range) * (1 - 0.5 * self.mu[r, c])
            self.pp_un[r, c] *= 1 - pd
            s = self.pp_un.sum()
            if s < 1e-200:
                self.pp_un /= s
                self._scale *= s

    def search_progress(self):
        """POS: probability that the search so far would have detected the person (1 - remaining prior mass)."""
        return 1.0 - float(self.pp_un.sum()) * self._scale

    def mark_blocked(self, cell):
        self.confirmed[cell] = True

    def relax(self, keep):
        """No route left: forget believed (not confirmed) blockages outside `keep` (cells in view)."""
        sel = ~keep & (self.lo > self.lo0)
        self.lo[sel] = self.lo0[sel]

    def area_mass(self, normalised=True):
        m = np.bincount(self._area_idx, weights=self.pp_un[self.searchable], minlength=self.b.n_areas)
        return m / m.sum() if normalised else m


class RawMap:
    """Coverage search: a cell looked at once counts as cleared, whatever the smoke did to detection."""
    bayes = False

    def __init__(self, b, cfg):
        self.b, self.cfg = b, cfg
        self.blk = _prior_block(b, cfg)
        self.haz = np.zeros(b.shape)
        self.seen = np.zeros(b.shape, bool)
        self.looked = np.zeros(b.shape, bool)    # looked at since the current sweep started
        self.confirmed = np.zeros(b.shape, bool)
        self.searchable = b.area_id >= 0
        self._area_idx = b.area_id[self.searchable]
        self._area_n = np.bincount(self._area_idx, minlength=b.n_areas)

    p_block = property(lambda self: np.where(self.confirmed, 1.0, self.blk))
    hazard = property(lambda self: self.haz)
    unc = property(lambda self: (~self.seen).astype(float))

    def predict(self):
        pass

    def update(self, o):
        self.blk[o.r, o.c] = o.blk
        self.haz[o.r, o.c] = o.z
        self.seen[o.r, o.c] = True
        self.looked[o.r, o.c] = True

    def mark_blocked(self, cell):
        self.confirmed[cell] = True
        self.seen[cell] = True

    def relax(self, keep):
        """No route left: forget blocked readings outside `keep` (cells in view) and try again."""
        prior = _prior_block(self.b, self.cfg)
        self.blk[~keep] = np.minimum(self.blk, prior)[~keep]

    def search_progress(self):
        """Share of the building looked at in the current sweep."""
        return float(self.looked[self.searchable].mean())

    def coverage(self):
        return np.bincount(self._area_idx, weights=self.looked[self.searchable],
                           minlength=self.b.n_areas) / self._area_n

    def area_mass(self, normalised=True):
        left = 1 - self.coverage()
        left[left < 1 - self.cfg.cover_frac] = 0
        if left.sum() == 0:              # everything covered without a find: sweep again
            self.looked[:] = False
            left = np.ones(self.b.n_areas)
        return left / left.sum()
