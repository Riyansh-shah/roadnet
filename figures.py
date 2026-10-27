"""Figures (figs/*.png) and summary tables (results/tables.md) from the experiment CSVs.

    python figures.py
"""
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from navsim import fuzzy
from navsim.mission import Mission
from navsim.render import (AXIS, GRID, INK, INK2, METHOD_COLOR, METHOD_ORDER, MUTED, PALETTE, PROB, draw_map,
                           style_axes)

FIGS, RES = "figs", "results"
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK, "figure.dpi": 100,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})
LABEL = {"FULL": "Full system", "noFuzzy": "without fuzzy A*", "noBayes": "without Bayes",
         "noGA": "without GA", "B0": "none of the three"}
rng = np.random.default_rng(0)


def load(name):
    with open(os.path.join(RES, name)) as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0]:
        try:
            out[k] = np.array([float(r[k]) for r in rows])
        except ValueError:
            out[k] = np.array([r[k] for r in rows])
    return out


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    means = rng.choice(x, (n, len(x))).mean(axis=1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def mcnemar_p(b, c):
    """Exact two-sided McNemar test on the discordant pair counts."""
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name))
    plt.close(fig)
    print("wrote", os.path.join(FIGS, name))


# ---------------------------------------------------------------- 1 + 3: one mission, chosen by a fixed rule
def story_seed(e1):
    """First test seed where FULL rescued the person after a real search (found at t >= 30 s)."""
    sel = (e1["method"] == "FULL") & (e1["success"] == 1) & (e1["t_find"] >= 30)
    return int(e1["seed"][sel].min())


def fig_mission(seed):
    m = Mission(seed, "FULL")
    snaps, beliefs = {}, {}
    while not m.done:
        if m.t in (8,):
            snaps["entry"] = _copy_view(m)
        m.step()
        if m.t_find is not None and "found" not in snaps:
            snaps["found"] = _copy_view(m)
    snaps["rescued"] = _copy_view(m)
    titles = {"entry": "t = 8 s: drones enter one by one and fan out on their GA routes",
              "found": f"t = {m.t_find} s: person detected; the nearest drone goes to them, the others leave",
              "rescued": f"t = {m.t_rescue} s: person led back out through the entrance (health {m.world.health:.2f})"}
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4))
    for ax, key in zip(axes, ("entry", "found", "rescued")):
        draw_map(ax, snaps[key], "truth", legend=(key == "rescued"))
        ax.set_title(titles[key], loc="left", fontweight="normal", color=INK2)
    fig.suptitle(f"Example mission (FULL, seed {seed}, 10 rooms, 3 drones): orange = smoke/fire", x=0.02, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig1_mission.png")
    return m


def _copy_view(m):
    """A frozen copy of what draw_map needs."""
    import copy
    return copy.deepcopy(m)


def fig_bayes(seed):
    m = Mission(seed, "FULL")
    views = [_copy_view(m)]
    find = None
    probe = Mission(seed, "FULL")
    while probe.t_find is None and not probe.done:
        probe.step()
    find = probe.t_find
    times = [0, max(1, find // 2), max(2, find - 1)]
    for t in times[1:]:
        while m.t < t:
            m.step()
        views.append(_copy_view(m))
    vmax = max(v.belief.pp.max() for v in views)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4))
    names = ["prior: uniform over every free cell", "misses drain P(person) from searched areas",
             "one step before the find: mass is left only where nobody could look (the burning room, corners)"]
    for ax, v, t, nm in zip(axes, views, times, names):
        draw_map(ax, v, "belief", vmax=vmax, routes=False)
        ax.set_title(f"t = {t} s: {nm}", loc="left", fontweight="normal", color=INK2)
    sm = plt.cm.ScalarMappable(cmap=PROB, norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("P(person in cell)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Bayesian search map shared by the drones (seed {seed}); red x = believed blocked",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig3_bayes.png")


# ---------------------------------------------------------------- 2: fuzzy system
def fig_fuzzy():
    fig = plt.figure(figsize=(10, 5.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.5], hspace=0.45, wspace=0.35)
    x = np.linspace(0, 1, 201)
    sets = list(fuzzy.INPUTS.items()) + [("Penalty (output)", fuzzy.PENALTY)]
    for i, (name, terms) in enumerate(sets):
        ax = fig.add_subplot(gs[0, i])
        for j, (term, p) in enumerate(terms.items()):
            y = fuzzy.trap(x, *p)
            ax.plot(x, y, color=PALETTE[j], lw=2)
            k = int(np.argmax(y))
            ax.text(x[k], 1.05 + 0.13 * (j % 2), term, color=INK2, fontsize=7, ha="center", va="bottom")
        ax.set_ylim(0, 1.32)
        ax.set_yticks([0, 0.5, 1])
        ax.set_title(name, loc="left")
        style_axes(ax)
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("membership")
    hz, bl = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101), indexing="xy")
    sub = gs[1, :].subgridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)
    surf = []
    for i, (res, who) in enumerate(((1.0, "drone, resilience 1"), (0.45, "escorted civilian, resilience 0.45"))):
        ax = fig.add_subplot(sub[0, i])
        surf.append(ax)
        pen = fuzzy.infer(hz, bl, 0.0, res)
        im = ax.imshow(pen, origin="lower", extent=(0, 1, 0, 1), cmap=PROB, vmin=0, vmax=1, aspect="auto")
        cs = ax.contour(hz, bl, pen, levels=[0.2, 0.4, 0.6, 0.8], colors="white", linewidths=0.8)
        ax.clabel(cs, fontsize=7, fmt="%.1f")
        ax.set_xlabel("believed hazard")
        if i == 0:
            ax.set_ylabel("believed P(blocked)")
        ax.set_title(f"Penalty: {who}", loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
    cb = fig.colorbar(im, cax=fig.add_subplot(sub[0, 2]))
    cb.set_label("penalty  (cell cost = 1 + 20 x penalty)", color=INK2)
    cb.outline.set_visible(False)
    fig.suptitle(f"Mamdani fuzzy cost: 4 inputs, {len(fuzzy.RULES)} rules, centroid defuzzification",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    save(fig, "fig2_fuzzy.png")


# ---------------------------------------------------------------- 4: E1 ablation
def fig_ablation(e1):
    methods = METHOD_ORDER
    seeds = np.unique(e1["seed"])
    ok_all = [s for s in seeds if all(e1["success"][(e1["seed"] == s) & (e1["method"] == m)][0] for m in methods)]
    panels = [("rescued alive (%)", lambda m, sel: (100 * e1["success"][sel].mean(), None)),
              ("efficiency vs best possible (%)", "eff_pct"),
              ("time to find (s)*", "t_find"), ("smoke dose during escort*", "escort_dose")]
    e1["eff_pct"] = 100 * e1["efficiency"]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 2.8), sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, (title, what) in zip(axes, panels):
        for yi, m in zip(y, methods):
            sel = e1["method"] == m
            if callable(what):
                v, err = what(m, sel)
                lo = hi = v
            else:
                common = sel & np.isin(e1["seed"], ok_all) if title.endswith("*") else sel
                v, lo, hi = boot_ci(e1[what][common])
            ax.barh(yi, v, height=0.62, color=METHOD_COLOR[m])
            if hi > lo:
                ax.errorbar(v, yi, xerr=[[v - lo], [hi - v]], color=INK, lw=1, capsize=2)
            ax.text(hi + 0.02 * ax.get_xlim()[1] if hi > lo else v, yi, f" {v:.3g}" if v < 10 else f" {v:.0f}",
                    va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left")
        style_axes(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=GRID, lw=0.6)
        ax.margins(x=0.25)
    axes[0].set_yticks(y, [LABEL[m] for m in methods])
    fig.suptitle(f"Why each component is needed: the full system vs simpler versions (100 winnable scenarios, "
                 f"10 rooms, 3 drones). * = over the {len(ok_all)} scenarios every version rescued; bars = mean, "
                 f"whiskers = 95% CI", x=0.02, ha="left", fontsize=9.5, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig4_ablation.png")
    return ok_all


# ---------------------------------------------------------------- 5: GA
def fig_ga(e3, curve):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.4, 1]})
    for (n, k), col, va in zip(((10, 3), (16, 6)), (PALETTE[0], PALETTE[6]), ("top", "bottom")):
        sel = (curve["n_rooms"] == n) & (curve["k_units"] == k)
        G = int(curve["gen"][sel].max())
        mat = np.full((len(np.unique(curve["seed"][sel])), G + 1), np.nan)
        for i, s in enumerate(np.unique(curve["seed"][sel])):
            ss = sel & (curve["seed"] == s)
            r = curve["best"][ss] / curve["greedy"][ss]
            mat[i, :len(r)] = r
            mat[i, len(r):] = r[-1]          # stopped early: the best stays
        mu = mat.mean(axis=0)
        lo, hi = np.percentile(mat, [25, 75], axis=0)
        a.fill_between(np.arange(G + 1), lo, hi, color=col, alpha=0.15, lw=0)
        a.plot(mu, color=col, lw=2)
        a.text(G, mu[-1], f"  N={n}, K={k}", color=INK2, va=va, fontsize=8)
    a.axhline(1, color=MUTED, lw=1, ls=(0, (4, 3)))
    a.text(0, 1.005, "greedy", color=MUTED, fontsize=8, va="bottom")
    a.set_xlabel("generation")
    a.set_ylabel("best objective / greedy")
    a.set_title("GA convergence (mean, band = middle 50% of seeds)", loc="left")
    a.margins(x=0.12)
    style_axes(a)
    cfgs = ((10, 3), (16, 6))
    w = 0.26
    for j, (key, col, lab) in enumerate((("ga", PALETTE[0], "GA"), ("greedy", PALETTE[1], "greedy"),
                                         ("random", MUTED, "random"))):
        for i, (n, k) in enumerate(cfgs):
            sel = (e3["n_rooms"] == n) & (e3["k_units"] == k)
            v = (e3[key][sel] / e3["greedy"][sel]).mean()
            b.bar(i + (j - 1) * w, v, width=w - 0.03, color=col, label=lab if i == 0 else None)
            b.text(i + (j - 1) * w, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
    b.set_xticks(range(len(cfgs)), [f"N={n} rooms, K={k} drones" for n, k in cfgs])
    b.set_ylabel("expected find time / greedy")
    b.set_title("Allocation objective at t = 30 s (lower is better)", loc="left")
    b.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    b.set_ylim(0, 1.3 * b.get_ylim()[1])
    style_axes(b)
    fig.tight_layout()
    save(fig, "fig5_ga.png")


# ---------------------------------------------------------------- 6: scaling + real time
def fig_scaling(e2):
    Ns, Ks = (6, 10, 16), (1, 2, 3, 4, 6)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), sharex=True)
    for c, n in enumerate(Ns):
        ends = {}
        for m in ("FULL", "noGA"):
            sr, tf = [], []
            for k in Ks:
                sel = (e2["n_rooms"] == n) & (e2["k_units"] == k) & (e2["method"] == m)
                sr.append(100 * e2["success"][sel].mean())
                tf.append(np.nanmean(e2["t_find"][sel & (e2["success"] == 1)]))
            for r, vals in ((0, sr), (1, tf)):
                ax = axes[r, c]
                ax.plot(Ks, vals, color=METHOD_COLOR[m], lw=2, marker="o", ms=4,
                        label=LABEL[m] if (r, c) == (0, 0) else None)
                ends.setdefault(r, {})[m] = vals[-1]
        if c == len(Ns) - 1:                      # direct labels, the higher line labelled above
            for r, e in ends.items():
                hi = max(e, key=e.get)
                for m, v in e.items():
                    axes[r, c].text(Ks[-1], v, "  " + m, color=INK2, fontsize=8,
                                    va="bottom" if m == hi else "top")
        axes[0, c].set_title(f"{n} rooms", loc="left")
        axes[1, c].set_xlabel("drones K")
    axes[0, 0].set_ylabel("rescued alive (%)")
    axes[1, 0].set_ylabel("time to find (s), rescued runs")
    for ax in axes.ravel():
        style_axes(ax)
        ax.set_xticks(Ks)
        ax.margins(x=0.15)
    for ax in axes[0]:
        ax.set_ylim(50, 100)                      # one honest scale: 1 seed = 2 points
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("More drones find the person sooner, with diminishing returns (50 seeds per point)",
                 x=0.02, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_scaling.png")

    fig, ax = plt.subplots(figsize=(6, 3))
    for i, n in enumerate(Ns):
        sel = (e2["n_rooms"] == n) & (e2["method"] == "FULL")
        mean = [e2["tick_ms_mean"][sel & (e2["k_units"] == k)].mean() for k in Ks]
        p95 = [e2["tick_ms_p95"][sel & (e2["k_units"] == k)].mean() for k in Ks]
        ax.plot(Ks, p95, color=PALETTE[i], lw=2, marker="o", ms=4)
        ax.plot(Ks, mean, color=PALETTE[i], lw=1.2, ls=(0, (3, 2)))
        ax.text(Ks[-1], p95[-1], f"  {n} rooms", color=INK2, fontsize=8, va="center")
    ax.axhline(1000, color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(1, 1000, "real-time budget: 1 tick = 1000 ms", color=MUTED, fontsize=8, va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(Ks)
    ax.set_xlabel("drones K")
    ax.set_ylabel("compute per tick (ms, log)")
    ax.set_title("Decision time per tick, FULL: p95 (solid) and mean (dashed)", loc="left")
    ax.margins(x=0.2)
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig7_realtime.png")


# ---------------------------------------------------------------- 8: heuristic
def heuristic_rows(e2):
    """Nodes expanded on the same sampled A* queries, per building size (full system), relative to no heuristic."""
    rows = []
    for n in (6, 10, 16):
        sel = (e2["n_rooms"] == n) & (e2["method"] == "FULL")
        t, o, d = (e2[k][sel].sum() for k in ("astar_nodes", "octile_nodes", "dijkstra_nodes"))
        rows.append((n, 100 * t / d, 100 * o / d, int(d)))
    return rows


def fig_heuristic(e2):
    rows = heuristic_rows(e2)
    fig, ax = plt.subplots(figsize=(6, 2.8))
    w = 0.26
    for j, (lab, col, idx) in enumerate((("no heuristic (Dijkstra)", MUTED, None), ("h = octile", PALETTE[1], 2),
                                         ("h = c_min x octile (ours)", PALETTE[0], 1))):
        for i, r in enumerate(rows):
            v = 100.0 if idx is None else r[idx]
            ax.bar(i + (j - 1) * w, v, width=w - 0.03, color=col, label=lab if i == 0 else None)
            ax.text(i + (j - 1) * w, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(rows)), [f"{r[0]} rooms" for r in rows])
    ax.set_ylabel("nodes expanded (% of Dijkstra)")
    ax.set_ylim(0, 125)
    ax.set_title("A* heuristic: same optimal paths, far fewer expanded nodes", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    style_axes(ax)
    fig.tight_layout()
    save(fig, "fig8_heuristic.png")


# ---------------------------------------------------------------- tables
def fmt(x, lo=None, hi=None, d=1):
    if np.isnan(x):
        return "-"
    return f"{x:.{d}f}" if lo is None else f"{x:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def tables(e1, ok_all, e2, e3):
    L = ["# Results\n",
         "Every scenario is winnable: an all-knowing rescuer that starts 120 s late can still save the person "
         "(`navsim/scenario.py`). Test scenarios are #100 and up; calibration used #0-39.\n",
         "**Success** = the person reaches the exit alive. **Efficiency** = best possible rescue time / actual rescue "
         "time (0 if the person is lost); the best possible time is that of an all-knowing rescuer starting at t = 0. "
         "Values are means with [95% bootstrap CI].\n"]
    f = e1["method"] == "FULL"
    ok = f & (e1["success"] == 1)
    L += ["## Headline: the full system (10 rooms, 3 drones, scenarios #100-199)\n",
          "| success | efficiency | time to find | best possible find | time to rescue | best possible rescue | "
          "person health at exit |", "|---|---|---|---|---|---|---|",
          f"| **{100 * e1['success'][f].mean():.0f}%** ({int(e1['success'][f].sum())}/{int(f.sum())}) | "
          f"**{fmt(*boot_ci(100 * e1['efficiency'][f]), d=0)} %** | {fmt(*boot_ci(e1['t_find'][ok]))} s | "
          f"{e1['oracle_find'][f].mean():.1f} s | {fmt(*boot_ci(e1['t_rescue'][ok]))} s | "
          f"{e1['oracle_rescue'][f].mean():.1f} s | {e1['health'][ok].mean():.2f} |"]
    L += ["", "## Why each component is needed (same 100 scenarios)\n",
          f"Times and escort smoke dose are over the {len(ok_all)} scenarios every version rescued.\n",
          "| version | success % | efficiency % | time to find (s) | time to rescue (s) | escort smoke dose |",
          "|---|---|---|---|---|---|"]
    for m in METHOD_ORDER:
        sel = e1["method"] == m
        com = sel & np.isin(e1["seed"], ok_all)
        L.append(f"| {LABEL[m]} | {100 * e1['success'][sel].mean():.0f} | "
                 f"{fmt(*boot_ci(100 * e1['efficiency'][sel]), d=0)} | {fmt(*boot_ci(e1['t_find'][com]))} | "
                 f"{fmt(*boot_ci(e1['t_rescue'][com]))} | {fmt(*boot_ci(e1['escort_dose'][com]), d=3)} |")
    L += ["", "Paired with the full system on the same scenarios (other minus full; a CI that excludes 0 is a clear "
          "effect; McNemar p is for the scenarios only one of the two rescued):\n",
          "| version | only full rescued | only this rescued | McNemar p | efficiency difference (points) | "
          "extra time to find (s) | extra escort smoke dose |", "|---|---|---|---|---|---|---|"]
    full = {int(s): (ok_, ef, tf, d) for s, ok_, ef, tf, d, m in
            zip(e1["seed"], e1["success"], e1["efficiency"], e1["t_find"], e1["escort_dose"], e1["method"])
            if m == "FULL"}
    for m in METHOD_ORDER[1:]:
        only_f = only_m = 0
        de, dt, dd = [], [], []
        for s, ok_, ef, tf, d, mm in zip(e1["seed"], e1["success"], e1["efficiency"], e1["t_find"],
                                         e1["escort_dose"], e1["method"]):
            if mm != m:
                continue
            fok, fef, ftf, fd = full[int(s)]
            only_f += int(fok and not ok_)
            only_m += int(ok_ and not fok)
            de.append(100 * (ef - fef))
            if ok_ and fok:
                dt.append(tf - ftf)
                dd.append(d - fd)
        p = mcnemar_p(only_f, only_m)
        ps = "<0.01" if p < 0.01 else f"{p:.2f}"
        L.append(f"| {LABEL[m]} | {only_f} | {only_m} | {ps} | {fmt(*boot_ci(de))} | {fmt(*boot_ci(dt))} | "
                 f"{fmt(*boot_ci(dd), d=3)} |")
    L += ["", "## Scaling (scenarios #100-149)\n", "Times are over the rescued runs.\n",
          "| rooms | drones | allocation | success % | efficiency % | time to find (s) | ms/tick p95 |",
          "|---|---|---|---|---|---|---|"]
    for n in (6, 10, 16):
        for k in (1, 2, 3, 4, 6):
            for m in ("FULL", "noGA"):
                sel = (e2["n_rooms"] == n) & (e2["k_units"] == k) & (e2["method"] == m)
                ok_ = sel & (e2["success"] == 1)
                L.append(f"| {n} | {k} | {'GA' if m == 'FULL' else 'greedy'} | {100 * e2['success'][sel].mean():.0f} | "
                         f"{100 * e2['efficiency'][sel].mean():.0f} | {np.nanmean(e2['t_find'][ok_]):.1f} | "
                         f"{e2['tick_ms_p95'][sel].mean():.1f} |")
    L += ["", "## GA allocation quality at t = 30 s (expected time to find; lower is better)\n",
          "| rooms | drones | problems | GA better than greedy | GA gain over greedy % | random vs greedy % | "
          "GA ms | greedy ms |", "|---|---|---|---|---|---|---|---|"]
    for n, k in ((10, 3), (16, 6)):
        sel = (e3["n_rooms"] == n) & (e3["k_units"] == k)
        g, gr, rd = e3["ga"][sel], e3["greedy"][sel], e3["random"][sel]
        L.append(f"| {n} | {k} | {sel.sum()} | {100 * (g < gr - 1e-9).mean():.0f}% | "
                 f"{fmt(*boot_ci(100 * (gr - g) / gr))} | {100 * ((rd - gr) / gr).mean():+.0f} | "
                 f"{e3['ga_ms'][sel].mean():.0f} | {e3['greedy_ms'][sel].mean():.1f} |")
    L += ["", "## A* heuristic (same sampled queries, full system)\n",
          "h(n) = c_min x octile distance, c_min = cheapest cell cost on the grid (admissible and consistent; "
          "paths stay optimal).\n",
          "| rooms | nodes with no heuristic (Dijkstra) | h = octile | h = c_min x octile |", "|---|---|---|---|"]
    for n, tight, octl, d in heuristic_rows(e2):
        L.append(f"| {n} | {d:,} (100%) | {octl:.0f}% | **{tight:.0f}%** |")
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", os.path.join(RES, "tables.md"))


def main():
    e1 = load("e1.csv")
    fig_fuzzy()
    ok_all = fig_ablation(e1)
    seed = story_seed(e1)
    fig_mission(seed)
    fig_bayes(seed)
    have = {f: os.path.exists(os.path.join(RES, f)) for f in ("e2.csv", "e3.csv", "ga_curve.csv")}
    e3 = curve = e2 = None
    if have["e3.csv"]:
        e3, curve = load("e3.csv"), load("ga_curve.csv")
        fig_ga(e3, curve)
    if have["e2.csv"]:
        e2 = load("e2.csv")
        fig_scaling(e2)
        fig_heuristic(e2)
    if e2 is not None and e3 is not None:
        tables(e1, ok_all, e2, e3)


if __name__ == "__main__":
    main()
