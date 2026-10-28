"""RoadNet app: three hill climbers race to connect the same houses with the least road.

    python demo.py                        interactive window (random map, 12 houses)
    python demo.py --preset square        start on a preset: triangle | square | random
    python demo.py --houses 15 --lakes    random map with 15 houses and lakes

Each map shows one climber. They all start from the road network with no junctions (the minimum spanning tree over
the houses) and get the same number of evaluations. The chart below tracks the best road each has found.
"""
import argparse
import sys
import time

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.widgets import Button

from roadnet.climb import CLIMBERS, budget_for
from roadnet.draw import MapView
from roadnet.network import baseline
from roadnet.world import random_world, square, triangle

BG, CARD, EDGE = "#1a1a19", "#242422", "#383835"
TEXT, TEXT2, MUTED = "#f0efec", "#c3c2b7", "#898781"
BTN, BTN_H, BLUE, BLUE_H = "#2c2c2a", "#383835", "#3987e5", "#2a78d6"
GRID = "#2c2c2a"
SERIES = ["#3987e5", "#d95926", "#199e70"]       # one colour per climber (validated for the dark surface)
SPEEDS = [1, 3, 10, 30, 100, 300]                # evaluations per frame, per climber
FRAME_MS = 40

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": TEXT,
                     "axes.edgecolor": EDGE, "toolbar": "None"})


class Race:
    """The three climbers running on one map."""

    def __init__(self, world, seed=0):
        self.world = world
        self.base_cost, self.base_edges = baseline(world)
        self.budget = budget_for(world.n)
        self.gens = [climber(world, self.budget, seed) for climber in CLIMBERS.values()]
        self.snaps = [None] * len(self.gens)
        self.history = [[(0, self.base_cost)] for _ in self.gens]

    @property
    def started(self):
        return any(s is not None for s in self.snaps)

    @property
    def finished(self):
        return all(s is not None and s.done for s in self.snaps)

    def advance(self, evals):
        for i, gen in enumerate(self.gens):
            snap = self.snaps[i]
            if snap is not None and snap.done:
                continue
            target = (snap.evals if snap else 0) + evals
            for snap in gen:
                if snap.done or snap.evals >= target:
                    break
            self.snaps[i] = snap
            self.history[i].append((snap.evals, snap.best.cost))


class App:
    def __init__(self, world, fullscreen=False):
        self.fig = plt.figure(figsize=(16, 9), facecolor=BG)
        self.fig.canvas.manager.set_window_title("RoadNet")
        if fullscreen:
            self.fig.canvas.manager.full_screen_toggle()
        self.running = False
        self.speed = 2
        self._build_header()
        self._build_maps()
        self._build_chart()
        self._build_controls()
        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self.tick)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.set_world(world)

    # ---------------------------------------------------------------- layout
    def _build_header(self):
        self.fig.text(0.025, 0.962, "RoadNet", fontsize=19, weight="bold", va="center")
        self.fig.text(0.098, 0.960, "connect every house with the least road, by hill climbing", fontsize=12,
                      color=TEXT2, va="center")
        self.header = self.fig.text(0.975, 0.960, "", fontsize=10.5, color=TEXT2, ha="right", va="center")

    def _build_maps(self):
        self.views, self.counters, self.details = [], [], []
        w, h, y0 = 0.29, 0.29 * 16 / 9, 0.31
        for i, name in enumerate(CLIMBERS):
            x0 = 0.025 + i * (w + 0.04)
            self.fig.add_artist(Line2D([x0, x0 + 0.022], [0.915, 0.915], color=SERIES[i], lw=3,
                                       solid_capstyle="round"))
            self.fig.text(x0 + 0.03, 0.915, name, fontsize=12, weight="bold", va="center")
            self.counters.append(self.fig.text(x0, 0.877, "", fontsize=19, weight="bold", va="center"))
            self.details.append(self.fig.text(x0, 0.842, "", fontsize=9, color=MUTED, va="center"))
            self.views.append(MapView(self.fig.add_axes([x0, y0, w, h])))

    def _build_chart(self):
        self.fig.text(0.025, 0.272, "Best road found so far", fontsize=11, weight="bold", va="center")
        self.fig.text(0.16, 0.272, "metres, against evaluations used (each climber gets the same budget)", fontsize=9,
                      color=MUTED, va="center")
        ax = self.chart = self.fig.add_axes([0.065, 0.06, 0.47, 0.18])
        ax.set_facecolor(BG)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        self.base_line = ax.axhline(0, color=MUTED, lw=1)
        self.best_line = ax.axhline(0, color=TEXT2, lw=1, ls=(0, (1, 2.5)))
        self.base_label = ax.text(0, 0, " no junctions", color=MUTED, fontsize=8, va="bottom")
        self.best_label = ax.text(0, 0, " known best", color=TEXT2, fontsize=8, va="top")
        self.lines = [ax.plot([], [], color=c, lw=2, solid_joinstyle="round", solid_capstyle="round", zorder=3)[0]
                      for c in SERIES]
        self.ends = [ax.plot([], [], "o", ms=8, color=c, mec=BG, mew=2, zorder=4)[0] for c in SERIES]

    def _button(self, rect, label, callback, primary=False):
        ax = self.fig.add_axes(rect)
        b = Button(ax, label, color=BLUE if primary else BTN, hovercolor=BLUE_H if primary else BTN_H)
        b.label.set_color(TEXT)
        b.label.set_fontsize(9.5)
        for s in ax.spines.values():
            s.set_color(EDGE)
        b.on_clicked(lambda _e: callback())
        return b

    def _row_label(self, y, text):
        self.fig.text(0.585, y + 0.02, text, fontsize=9, color=MUTED, va="center")

    def _build_controls(self):
        x, bh = 0.625, 0.04
        self._row_label(0.145, "Run")
        self.b_run = self._button([x, 0.145, 0.08, bh], "▶  Solve", self.toggle_run, primary=True)
        self.b_step = self._button([x + 0.085, 0.145, 0.06, bh], "Step", self.step)
        self.b_reset = self._button([x + 0.15, 0.145, 0.06, bh], "Reset", self.reset)

    # ---------------------------------------------------------------- state
    def set_world(self, world):
        self.world = world
        self.pause()
        self.race = Race(world)
        for v in self.views:
            v.set_world(world)
        best = f"   ·   known best {world.optimum:,.0f} m" if world.optimum else ""
        self.header.set_text(f"{world.n} houses   ·   no-junction road {self.race.base_cost:,.0f} m{best}   ·   "
                             f"{self.race.budget:,} evaluations per climber")
        self.refresh()

    def reset(self):
        self.set_world(self.world)

    def toggle_run(self):
        if self.running:
            self.pause()
        elif not self.race.finished and self.world.n >= 3:
            self.running = True
            self.b_run.label.set_text("❚❚  Pause")
            self.timer.start()
        self.fig.canvas.draw_idle()

    def pause(self):
        self.running = False
        self.timer.stop()
        if hasattr(self, "b_run"):
            self.b_run.label.set_text("▶  Solve")

    def step(self):
        self.pause()
        if not self.race.finished:
            self.race.advance(1)
            self.refresh()

    def tick(self):
        self.race.advance(SPEEDS[self.speed])
        self.refresh()
        if self.race.finished:
            self.pause()
            self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- drawing
    def refresh(self):
        race = self.race
        empty = np.zeros((0, 2))
        for i, view in enumerate(self.views):
            snap = race.snaps[i]
            if self.world.n < 3:
                view.show(empty, race.base_edges)
                self.counters[i].set_text("add at least 3 houses")
                self.details[i].set_text("click on a map to place houses")
                continue
            if snap is None:
                view.show(empty, race.base_edges)
                self.counters[i].set_text(f"{race.base_cost:,.0f} m")
                self.details[i].set_text("no junctions yet   ·   press Solve")
                continue
            cur = snap.current
            view.show(cur.junctions, cur.edges)
            saved = 100 * (1 - snap.best.cost / race.base_cost)
            self.counters[i].set_text(f"{cur.cost:,.0f} m")
            self.details[i].set_text(f"best {snap.best.cost:,.0f} m (saves {saved:.1f}%)   ·   "
                                     f"{len(cur.junctions)} junctions   ·   {snap.evals:,} evals   ·   {snap.status}")
        self._refresh_chart()
        self.fig.canvas.draw_idle()

    def _refresh_chart(self):
        race, ax = self.race, self.chart
        lows = [race.base_cost]
        for i, line in enumerate(self.lines):
            h = np.array(race.history[i])
            line.set_data(h[:, 0], h[:, 1])
            self.ends[i].set_data(h[-1:, 0], h[-1:, 1])
            lows.append(h[:, 1].min())
        opt = self.world.optimum
        if opt:
            lows.append(opt)
        lo, hi = min(lows), race.base_cost
        pad = max(hi - lo, 1.0) * 0.12
        ax.set_xlim(0, race.budget * 1.02)
        ax.set_ylim(lo - pad, hi + pad)
        self.base_line.set_ydata([hi, hi])
        self.base_label.set_position((0, hi))
        self.best_line.set_visible(bool(opt))
        self.best_label.set_visible(bool(opt))
        if opt:
            self.best_line.set_ydata([opt, opt])
            self.best_label.set_position((0, opt))

    def on_key(self, event):
        actions = {" ": self.toggle_run, "right": self.step, "r": self.reset}
        if event.key in actions:
            actions[event.key]()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=["random", "square", "triangle"], default="random")
    p.add_argument("--houses", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lakes", action="store_true")
    p.add_argument("--fullscreen", action="store_true")
    a = p.parse_args()
    world = {"square": square, "triangle": triangle}.get(a.preset, lambda: random_world(a.houses, a.seed, a.lakes))()
    App(world, fullscreen=a.fullscreen)
    plt.show()


if __name__ == "__main__":
    main()
