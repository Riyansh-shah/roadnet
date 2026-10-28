"""RoadNet app: three hill climbers race to connect the same houses with the least road.

    python demo.py                        interactive window (random map, 12 houses)
    python demo.py --preset square        start on a preset: triangle | square | random
    python demo.py --houses 15 --lakes    random map with 15 houses and lakes

Each map shows one climber. They all start from the road network with no junctions (the minimum spanning tree over
the houses) and get the same number of evaluations. The chart below tracks the best road each has found.

Mouse, on any map: click = add a house, drag = move it, right-click = delete it.
Keys: space solve/pause · → step · r reset · n new random map · l lakes · a 120° angles · i how it works ·
+ / − speed · q quit.
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
from roadnet.world import MAX_HOUSES, MIN_HOUSES, World, add_lakes, random_world, square, triangle

BG, CARD, EDGE = "#1a1a19", "#242422", "#383835"
TEXT, TEXT2, MUTED = "#f0efec", "#c3c2b7", "#898781"
BTN, BTN_H, BLUE, BLUE_H = "#2c2c2a", "#383835", "#3987e5", "#2a78d6"
GRID = "#2c2c2a"
SERIES = ["#3987e5", "#d95926", "#199e70"]       # one colour per climber (validated for the dark surface)
SPEEDS = [1, 3, 10, 30, 100, 300]                # evaluations per frame, per climber
FRAME_MS = 40
GRAB = 45                                        # metres: how close a click must be to grab a house
HINT = "Mouse, on any map:  click = add a house   ·   drag = move it   ·   right-click = delete it"

PRESETS = {"triangle": triangle, "square": square}

INFO = [
    ("The problem", """Every house must be connected to every other house by road. A metre of road
costs 1, and a metre across a lake costs 3. We may add junctions: extra points
where roads meet. Where should they go to make the total road shortest?
This is the Steiner tree problem. It is NP-hard, so instead of trying every
possibility we search with hill climbing."""),
    ("Search formulation", """State: the positions of the junctions (at most N − 2 for N houses).
Cost: total road of the cheapest network joining the houses and junctions,
which is a minimum spanning tree (Prim's algorithm).
Start: no junctions, the plain spanning tree over the houses. This is the
baseline without search.
Moves: nudge a junction  ·  add one in the middle of a 'V' of roads  ·  remove one."""),
    ("The three climbers (same number of evaluations each)", """Simple hill climbing tries a random move and keeps it only if the road gets
shorter. When no nearby move helps, it is stuck at a local optimum and stops.
Random-restart hill climbing starts again from random junctions whenever it
is stuck, and keeps the best network it ever found.
Simulated annealing also accepts a longer road with probability e^(−Δ/T), so it
can climb out of a local optimum. T cools down, so it ends as hill climbing."""),
    ("How we know it works", """Triangle: the best network has one junction with three 120° angles,
√3 × side = 1,732 m, which saves 13.4%.
Square: the X costs 2,828 m, but the H with two junctions costs 2,732 m.
On flat land, junctions settle at 120° (press "120° angles"). Junctions save at
most about 13.4% over the no-junction network (Steiner ratio √3/2).
Measured: road length, % saved, evaluations used, junctions placed."""),
]

plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": TEXT,
                     "axes.edgecolor": EDGE, "toolbar": "None"})
for _k in ("home", "back", "forward", "pan", "zoom", "save", "grid", "grid_minor", "xscale", "yscale"):
    plt.rcParams[f"keymap.{_k}"] = []      # our own keys only (matplotlib's defaults clash with them)


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
    def __init__(self, world, fullscreen=False, houses=12, seed=0):
        self.n_houses, self.seed = houses, seed
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
        self._build_info()
        self.timer = self.fig.canvas.new_timer(interval=FRAME_MS)
        self.timer.add_callback(self.tick)
        self.drag = None
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
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
        bh, g = 0.04, 0.005
        self.buttons = []

        def row(y, label, items):
            """items: (width, label, callback, primary) for a button, or (width, None) for a value readout."""
            self._row_label(y, label)
            x, made = 0.62, []
            for width, text, *rest in items:
                if text is None:
                    made.append(self.fig.text(x + width / 2, y + bh / 2, "", fontsize=9.5, ha="center", va="center"))
                else:
                    made.append(self._button([x, y, width, bh], text, rest[0], primary=bool(rest[1:])))
                    self.buttons.append(made[-1])
                x += width + g
            return made

        _, _, _, _, self.houses_text, _, self.b_lakes = row(0.215, "Map", [
            (0.055, "Triangle", lambda: self.preset("triangle")), (0.055, "Square", lambda: self.preset("square")),
            (0.055, "Random", self.new_random), (0.022, "−", lambda: self.change_houses(-1)), (0.05, None),
            (0.022, "+", lambda: self.change_houses(+1)), (0.068, "Lakes: off", self.toggle_lakes)])
        self.b_run, self.b_step, self.b_reset, _, self.speed_text, _ = row(0.145, "Run", [
            (0.08, "▶  Solve", self.toggle_run, True), (0.055, "Step", self.step), (0.055, "Reset", self.reset),
            (0.022, "−", lambda: self.change_speed(-1)), (0.06, None), (0.022, "+", lambda: self.change_speed(+1))])
        self.b_angles, self.b_info, self.b_clear = row(0.075, "View", [
            (0.105, "120° angles: off", self.toggle_angles), (0.1, "How it works", self.toggle_info),
            (0.07, "Clear map", self.clear)])
        self.hint = self.fig.text(0.585, 0.035, HINT, fontsize=9, color=MUTED, va="center")
        self.houses_text.set_text(f"{self.n_houses} houses")
        self.speed_text.set_text(f"speed ×{SPEEDS[self.speed]}")

    def _build_info(self):
        ax = self.info = self.fig.add_axes([0.02, 0.3, 0.96, 0.645])
        ax.set_facecolor(CARD)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(EDGE)
        ax.text(0.02, 0.94, "How RoadNet works", fontsize=16, weight="bold", va="center", transform=ax.transAxes)
        ax.text(0.98, 0.94, "click anywhere on this panel, or press i, to close", fontsize=9, color=MUTED,
                ha="right", va="center", transform=ax.transAxes)
        for k, (title, body) in enumerate(INFO):
            x, y = 0.02 + 0.49 * (k % 2), 0.84 - 0.42 * (k // 2)
            ax.text(x, y, title, fontsize=12, weight="bold", color=BLUE, va="top", transform=ax.transAxes)
            ax.text(x, y - 0.065, body, fontsize=10.5, color=TEXT2, va="top", linespacing=1.55,
                    transform=ax.transAxes)
        ax.set_zorder(10)             # above the figure-level titles and counters
        ax.set_visible(False)

    def toggle_info(self):
        self.info.set_visible(not self.info.get_visible())
        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------------- state
    def set_world(self, world):
        self.world = world
        self.pause()
        self.race = Race(world)
        for v in self.views:
            v.set_world(world)
        if hasattr(self, "b_lakes"):
            self.b_lakes.label.set_text(f"Lakes: {'on' if len(world.lakes) else 'off'}")
        best = f"   ·   known best {world.optimum:,.0f} m" if world.optimum else ""
        self.header.set_text(f"{world.n} houses   ·   no-junction road {self.race.base_cost:,.0f} m{best}   ·   "
                             f"{self.race.budget:,} evaluations per climber")
        self.refresh()

    def reset(self):
        self.set_world(self.world)

    def preset(self, name):
        self.set_world(PRESETS[name]())
        self.hint.set_text(HINT)

    def new_random(self, seed=None):
        self.seed = self.seed + 1 if seed is None else seed
        self.set_world(random_world(self.n_houses, self.seed, lakes=bool(len(self.world.lakes))))
        self.hint.set_text(f"Random map #{self.seed}.   " + HINT)

    def change_houses(self, delta):
        self.n_houses = int(np.clip(self.n_houses + delta, MIN_HOUSES, MAX_HOUSES))
        self.houses_text.set_text(f"{self.n_houses} houses")
        self.new_random(self.seed)

    def toggle_lakes(self):
        w = self.world
        if len(w.lakes):
            w = PRESETS[w.name]() if w.name in PRESETS else World(w.houses, None, w.name)
        else:
            w = add_lakes(w, self.seed)
            w.optimum = None
        self.set_world(w)

    def clear(self):
        self.edit(np.zeros((0, 2)), "Click on a map to place houses (at least 3).")

    def change_speed(self, delta):
        self.speed = int(np.clip(self.speed + delta, 0, len(SPEEDS) - 1))
        self.speed_text.set_text(f"speed ×{SPEEDS[self.speed]}")
        self.fig.canvas.draw_idle()

    def toggle_angles(self):
        on = not self.views[0].show_angles
        for v in self.views:
            v.show_angles = on
        self.b_angles.label.set_text(f"120° angles: {'on' if on else 'off'}")
        self.refresh()

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
        ax.set_xlim(0, max(race.budget, 100) * 1.02)
        ax.set_ylim(lo - pad, hi + pad)
        self.base_line.set_ydata([hi, hi])
        self.base_label.set_position((0, hi))
        self.best_line.set_visible(bool(opt))
        self.best_label.set_visible(bool(opt))
        if opt:
            self.best_line.set_ydata([opt, opt])
            self.best_label.set_position((0, opt))

    # ---------------------------------------------------------------- mouse editing
    def _map_point(self, event):
        if event.xdata is None or event.inaxes not in [v.ax for v in self.views]:
            return None
        return np.array([event.xdata, event.ydata])

    def _house_at(self, p):
        if not self.world.n:
            return None
        d = np.hypot(*(self.world.houses - p).T)
        return int(np.argmin(d)) if d.min() < GRAB else None

    def edit(self, houses, message=HINT):
        self.set_world(World(houses, self.world.lakes, name="custom"))
        self.hint.set_text(message)

    def on_press(self, event):
        if self.info.get_visible():
            if event.inaxes is self.info:
                self.toggle_info()
            return
        p = self._map_point(event)
        if p is None:
            return
        i = self._house_at(p)
        houses = self.world.houses
        if event.button == 1 and i is not None:
            self.drag = i
        elif event.button == 1:
            if self.world.n >= MAX_HOUSES:
                self.hint.set_text(f"At most {MAX_HOUSES} houses.")
            elif self.world.in_lake(p, margin=20):
                self.hint.set_text("No houses in a lake.")
            else:
                self.edit(np.vstack([houses, p]))
                return
            self.fig.canvas.draw_idle()
        elif event.button == 3 and i is not None:
            self.edit(np.delete(houses, i, axis=0))

    def on_motion(self, event):
        p = self._map_point(event)
        if self.drag is None or p is None or self.info.get_visible() or self.world.in_lake(p, margin=20):
            return
        houses = self.world.houses.copy()
        houses[self.drag] = p
        self.edit(houses)

    def on_release(self, _event):
        self.drag = None

    def on_key(self, event):
        actions = {" ": self.toggle_run, "right": self.step, "r": self.reset, "n": self.new_random,
                   "l": self.toggle_lakes, "a": self.toggle_angles, "i": self.toggle_info, "+": lambda: self.change_speed(+1),
                   "=": lambda: self.change_speed(+1), "-": lambda: self.change_speed(-1)}
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
    App(world, fullscreen=a.fullscreen, houses=a.houses, seed=a.seed)
    plt.show()


if __name__ == "__main__":
    main()
