"""Presentation app: drones find a person trapped in a burning building and lead them out.

    python demo.py                                   interactive window
    python demo.py --fullscreen                      same, full screen
    python demo.py --scenario 3 --save figs/demo.gif record one mission to a GIF (no window)

The system: a Bayesian search map (where the person probably is), GA allocation of rooms to drones, and A* paths.
Every scenario is checked to be winnable (navsim/scenario.py).

Controls (right): scenario ◀ ▶ / random, rooms − +, drones − +, Start/Pause, Step, Reset, speed − +, view toggles,
and a 20-mission benchmark. Metrics (middle): success definition, live search-progress and health charts, the
end-of-mission report card against the best possible rescue, and the heuristics with their measured effect.
Keys: space start/pause · → step · r reset · n random scenario · i how it works · f full screen · q quit.
"""
import argparse
import csv
import logging
import os
import random
import sys
import time

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.transforms import Bbox, TransformedBbox
from matplotlib.widgets import Button, CheckButtons

from navsim.astar import c_min
from navsim.mission import Mission
from navsim.render import MapView

BG, CARD, EDGE = "#1a1a19", "#242422", "#383835"
TEXT, TEXT2, MUTED = "#f0efec", "#c3c2b7", "#898781"
BTN, BTN_H, BLUE, BLUE_H = "#2c2c2a", "#383835", "#3987e5", "#2a78d6"
GREEN, AMBER, RED, FIRE = "#0ca30c", "#fab219", "#e66767", "#ff9a3c"
EVENT_COLOR = {"info": TEXT2, "fire": FIRE, "block": RED, "ga": "#86b6ef", "good": GREEN, "bad": RED}
FIG_W, FIG_H = 16, 9
BENCH_N = 20

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("Segoe UI", "Segoe UI Symbol", "DejaVu Sans") if f in _have]
for k in ("keymap.home", "keymap.forward", "keymap.back", "keymap.save"):
    plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("r", "right", "left", "s", "n")]
plt.rcParams["toolbar"] = "None"
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def card(fig, x, y, w, h, title=None):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.007",
                                  mutation_aspect=FIG_W / FIG_H, transform=fig.transFigure,
                                  fc=CARD, ec=EDGE, lw=0.8, zorder=-1))
    if title:
        fig.text(x + 0.01, y + h - 0.011, title, fontsize=10, fontweight="bold", color=MUTED, va="top")


def phase(m):
    if m.outcome == "rescued":
        return f"RESCUED IN {m.t_rescue} s", GREEN, "white"
    if m.outcome == "casualty":
        return "PERSON LOST", "#d03b3b", "white"
    if m.outcome == "timeout":
        return "TIME OUT", "#d03b3b", "white"
    if m.t_find is None:
        return "SEARCHING", BLUE, "white"
    if m.escort is not None and m.escort.state == "ESCORT":
        return "LEADING THEM OUT", AMBER, "#0b0b0b"
    return "PERSON FOUND", AMBER, "#0b0b0b"


class EventLog:
    """Turns changes in the mission state into short, timestamped lines for the feed."""

    def __init__(self, m):
        self.m, self.lines = m, []
        self.sites = m.sites
        self.fire = self._fire_rooms()
        self.seen_block = set()
        self.state = [u.state for u in m.units]
        self.bumps = [0] * len(m.units)
        self.ga, self.found, self.escorting = 0, False, False
        self.add(f"fire burning in {', '.join('R%d' % r for r in sorted(self.fire))}", "fire")

    def _fire_rooms(self):
        b, w = self.m.b, self.m.world
        return set(np.unique(b.room_id[w.burning & (b.room_id >= 0)]).tolist())

    def _where(self, cell):
        b, (r, c) = self.m.b, cell
        rooms = sorted({int(b.room_id[rr, cc]) for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                        if 0 <= rr < b.shape[0] and 0 <= cc < b.shape[1] and b.room_id[rr, cc] >= 0})
        return "door of " + "/".join(f"R{k}" for k in rooms) if rooms else "corridor"

    def add(self, text, kind):
        if kind == "ga" and self.lines and self.lines[-1][2] == "ga":
            t, _, _, n = self.lines[-1]
            self.lines[-1] = (t, f"GA re-planned routes  x{n + 1}", "ga", n + 1)
        else:
            self.lines.append((self.m.t, text, kind, 1))

    def update(self):
        m = self.m
        fire = self._fire_rooms()
        for r in sorted(fire - self.fire):
            self.add(f"fire spread to R{r}", "fire")
        self.fire = fire
        for cell in map(tuple, np.argwhere(self.sites & (m.belief.p_block > 0.5))):
            if cell not in self.seen_block:
                self.seen_block.add(cell)
                self.add(f"blocked: {self._where(cell)}", "block")
        for i, u in enumerate(m.units):
            if self.state[i] == "QUEUE" and u.state != "QUEUE":
                self.add(f"drone {u.id + 1} entered", "info")
            if u.bumps > self.bumps[i]:
                self.add(f"drone {u.id + 1} hit a hidden blockage", "block")
            self.state[i], self.bumps[i] = u.state, u.bumps
        if m.ga_calls > self.ga:
            self.add("GA re-planned routes", "ga")
            self.ga = m.ga_calls
        if m.t_find is not None and not self.found:
            self.found = True
            self.add(f"PERSON FOUND by drone {m.escort.id + 1}" if m.escort else "PERSON FOUND", "good")
        if m.escort is not None and m.escort.state == "ESCORT" and not self.escorting:
            self.escorting = True
            self.add(f"drone {m.escort.id + 1} leading them out", "good")
        if m.outcome == "rescued":
            self.add(f"RESCUED after {m.t_rescue} s", "good")
        elif m.outcome:
            self.add("person lost" if m.outcome == "casualty" else "time out", "bad")


class Regions:
    """Region blitting. Static parts of the window are rendered once; each region keeps a copy of its static
    background and, per frame, only its own animated artists are redrawn and copied to the screen."""

    def __init__(self, fig):
        self.fig, self.items, self.ready = fig, {}, False
        fig.canvas.mpl_connect("draw_event", self._on_draw)

    def rect(self, x, y, w, h):
        return TransformedBbox(Bbox.from_bounds(x, y, w, h), self.fig.transFigure)

    def set(self, name, bbox, artists):
        for a in artists:
            a.set_animated(True)
        self.items[name] = dict(bbox=bbox, artists=artists, bg=None)
        self.ready = False

    def _on_draw(self, _event):             # after every full draw: grab the backgrounds, then add the dynamic parts
        c = self.fig.canvas
        for it in self.items.values():
            it["bg"] = c.copy_from_bbox(it["bbox"])
        for it in self.items.values():
            self._draw(it)
        self.ready = True

    def _draw(self, it):
        for a in it["artists"]:
            if a.get_visible() and (a.axes is None or a.axes.get_visible()):
                self.fig.draw_artist(a)

    def refresh(self, names):
        c = self.fig.canvas
        for n in names:
            it = self.items[n]
            c.restore_region(it["bg"])
            self._draw(it)
            c.blit(it["bbox"])


class App:
    LIMITS = {"rooms": (4, 20), "drones": (1, 8), "speed": (1, 10)}

    def __init__(self, scenario=3, rooms=10, drones=3, speed=2):
        self.scenario, self.rooms, self.drones, self.speed = scenario, rooms, drones, speed
        self.running, self.show_person, self.show_plan = False, True, True
        self.bench, self.bench_done = None, None
        self.frame_ms, self.buttons = [], {}
        fig = self.fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("Drone search & rescue")

        # ---------------- header + KPI tiles (left column)
        fig.text(0.014, 0.962, "DRONE SEARCH & RESCUE", fontsize=20, fontweight="bold", color=TEXT, va="center")
        fig.text(0.014, 0.928, "one floor · one entrance that is also the only exit · a fire that starts small and spreads",
                 fontsize=10, color=MUTED, va="center")
        self.pill = fig.text(0.585, 0.958, "", fontsize=12.5, fontweight="bold", ha="right", va="center",
                             bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.9", fc=BLUE, ec="none"))
        self.kpi = {}
        tiles = [("time", "TIME"), ("health", "PERSON HEALTH"), ("pos", "PROBABILITY SEARCHED"),
                 ("inside", "DRONES INSIDE")]
        tw, gap, x0, y0, th = 0.1375, 0.008, 0.012, 0.815, 0.09
        for i, (key, label) in enumerate(tiles):
            x = x0 + i * (tw + gap)
            card(fig, x, y0, tw, th)
            fig.text(x + 0.01, y0 + th - 0.014, label, fontsize=8.5, fontweight="bold", color=MUTED, va="top")
            self.kpi[key] = fig.text(x + 0.01, y0 + 0.027, "", fontsize=21, fontweight="bold", color=TEXT,
                                     va="center")
            if key == "health":
                self.ax_health = ax = fig.add_axes([x + 0.062, y0 + 0.02, tw - 0.074, 0.015])
                ax.barh(0, 1, color=EDGE, height=1)
                self.hbar = ax.barh(0, 1, color=GREEN, height=1)[0]
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.set_axis_off()
        self.kpi_rect = (x0, y0, 4 * tw + 3 * gap, th)

        # ---------------- maps
        card(fig, 0.012, 0.415, 0.573, 0.39, "REALITY  ·  fire (bright) and smoke (orange), the drones, the trapped person")
        card(fig, 0.012, 0.012, 0.573, 0.393, "WHAT THE DRONES BELIEVE  ·  blue = where the person probably is,"
                                              "  outlines = GA plan (D2·1 = drone 2's first room)")
        self.ax_truth = fig.add_axes([0.018, 0.42, 0.561, 0.355])
        self.ax_belief = fig.add_axes([0.018, 0.017, 0.561, 0.358])

        # ---------------- metrics column
        mx, mw = 0.595, 0.205
        card(fig, mx, 0.4, mw, 0.585, "SUCCESS METRICS")
        fig.text(mx + 0.01, 0.948, "Success = person reaches the exit alive", fontsize=9.5, color=TEXT, va="top")
        fig.text(mx + 0.01, 0.927, "Efficiency = best possible rescue time ÷ our time", fontsize=9.5, color=TEXT,
                 va="top")
        fig.text(mx + 0.01, 0.906, "(best possible = an all-knowing rescuer)", fontsize=8.5, color=MUTED, va="top")
        self.ax_pos = self._chart([mx + 0.032, 0.745, mw - 0.045, 0.115], "Probability searched (POS)", BLUE)
        self.ax_hp = self._chart([mx + 0.032, 0.585, mw - 0.045, 0.115], "Person health", GREEN)
        self.report = [fig.text(mx + 0.01, 0.535 - i * 0.0215, "", fontsize=10, color=TEXT2, va="center",
                                family="monospace") for i in range(6)]

        card(fig, mx, 0.2, mw, 0.19, "HEURISTICS")
        fig.text(mx + 0.01, 0.355, "A*:  f(n) = g(n) + h(n)", fontsize=9.5, color=TEXT, va="top")
        fig.text(mx + 0.01, 0.335, "h(n) = octile distance (straight 1, diagonal √2)", fontsize=9.5, color=TEXT,
                 va="top")
        self.heur = [fig.text(mx + 0.01, 0.305 - i * 0.02, "", fontsize=9.2, color=TEXT2, va="top")
                     for i in range(2)]
        fig.text(mx + 0.01, 0.262, "GA:  fitness = Σ P(room) × arrival time", fontsize=9.5, color=TEXT, va="top")
        self.heur.append(fig.text(mx + 0.01, 0.242, "", fontsize=9.2, color=TEXT2, va="top"))

        card(fig, mx, 0.012, mw, 0.178, "EVENTS")
        self.event_text = [fig.text(mx + 0.01, 0.155 - i * 0.0215, "", fontsize=9.2, color=TEXT2, va="center")
                           for i in range(7)]

        # ---------------- controls column
        cx, cw = 0.81, 0.178
        card(fig, cx, 0.7, cw, 0.285, "SCENARIO")
        self.lbl = {}
        self._stepper(0.9, "Scenario", "scenario", self.prev_scenario, self.next_scenario)
        self._button([cx + 0.01, 0.84, cw - 0.02, 0.038], "🎲  Random scenario", self.random_scenario, name="random")
        self._stepper(0.78, "Rooms", "rooms", lambda: self._change("rooms", -1), lambda: self._change("rooms", +1))
        self._stepper(0.725, "Drones", "drones", lambda: self._change("drones", -1),
                      lambda: self._change("drones", +1))

        card(fig, cx, 0.4, cw, 0.29, "PLAYBACK")
        self.b_start = self._button([cx + 0.01, 0.6, cw - 0.02, 0.048], "▶  Start", self.toggle, primary=True,
                                    name="start")
        self._button([cx + 0.01, 0.545, (cw - 0.025) / 2, 0.044], "Step  ⏭", self.step_once, name="step")
        self._button([cx + 0.015 + (cw - 0.025) / 2, 0.545, (cw - 0.025) / 2, 0.044], "■  Reset", self.reset,
                     name="reset")
        self._stepper(0.49, "Speed", "speed", lambda: self._change("speed", -1), lambda: self._change("speed", +1))
        ax = fig.add_axes([cx + 0.006, 0.405, cw - 0.012, 0.06], facecolor=CARD)
        ax.set_axis_off()
        self.checks = CheckButtons(ax, ["show the trapped person", "show the GA plan"], [True, True],
                                   label_props={"color": [TEXT] * 2, "fontsize": [10] * 2},
                                   frame_props={"edgecolor": [TEXT2] * 2, "facecolor": [CARD] * 2, "s": [70] * 2},
                                   check_props={"facecolor": [BLUE] * 2, "s": [70] * 2})
        self.checks.on_clicked(self._toggle_view)

        self.b_info = self._button([cx, 0.345, cw, 0.045], "ⓘ  How it works", self.toggle_info, name="info")

        card(fig, cx, 0.012, cw, 0.322, "BENCHMARK")
        fig.text(cx + 0.01, 0.3, f"{BENCH_N} scenarios in a row with the", fontsize=9, color=MUTED, va="top")
        fig.text(cx + 0.01, 0.283, "current rooms and drones", fontsize=9, color=MUTED, va="top")
        self.b_bench = self._button([cx + 0.01, 0.222, cw - 0.02, 0.042], f"Run {BENCH_N} missions",
                                    self.toggle_bench, name="bench")
        self.bench_text = [fig.text(cx + 0.01, 0.2 - i * 0.02, "", fontsize=9.8, color=TEXT2, va="center")
                           for i in range(4)]
        self.ax_bench = fig.add_axes([cx + 0.02, 0.03, cw - 0.03, 0.085], facecolor=CARD)
        self.ax_bench.set_axis_off()
        self._build_info()

        # ---------------- blitting regions (the maps are added in build())
        self.regions = r = Regions(fig)
        r.set("kpi", r.rect(*self.kpi_rect), [*self.kpi.values(), self.hbar])
        r.set("pill", r.rect(0.3, 0.93, 0.29, 0.058), [self.pill])
        r.set("pos", self.ax_pos.bbox, [self.ax_pos.line])
        r.set("hp", self.ax_hp.bbox, [self.ax_hp.line])
        r.set("report", r.rect(mx + 0.002, 0.415, mw - 0.004, 0.135), self.report)
        r.set("heur", r.rect(mx + 0.002, 0.225, mw - 0.004, 0.095), self.heur)
        r.set("events", r.rect(mx + 0.002, 0.015, mw - 0.004, 0.152), self.event_text)
        self._shown = {}

        fig.canvas.mpl_connect("key_press_event", self._key)
        self.build()
        self.timer = fig.canvas.new_timer(interval=40)
        self.timer.single_shot = True
        self.timer.add_callback(self._tick)
        self.timer.start()

    # ---------------- widget helpers
    def _chart(self, rect, title, color):
        ax = self.fig.add_axes(rect, facecolor=CARD)
        ax.set_title(title, fontsize=9, color=TEXT2, loc="left", pad=3)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(EDGE)
        ax.tick_params(colors=MUTED, labelsize=7.5, length=2)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1])
        ax.grid(True, color=EDGE, lw=0.5)
        ax.line, = ax.plot([], [], color=color, lw=2)
        return ax

    def _button(self, rect, text, cb, primary=False, name=None):
        b = Button(self.fig.add_axes(rect), text, color=BLUE if primary else BTN,
                   hovercolor=BLUE_H if primary else BTN_H)
        self.buttons[name or text] = b
        b.label.set_fontsize(10.5)
        b.label.set_color("white" if primary else TEXT)
        for s in b.ax.spines.values():
            s.set_color(BLUE if primary else EDGE)
        b.on_clicked(lambda _e: cb())
        return b

    def _stepper(self, y, label, key, dec, inc):
        cx = 0.81
        self.fig.text(cx + 0.01, y + 0.019, label, fontsize=10.5, color=TEXT2, va="center")
        self._button([cx + 0.075, y, 0.03, 0.038], "◀" if key == "scenario" else "−", dec, name=key + "-")
        self.lbl[key] = self.fig.text(cx + 0.1225, y + 0.019, "", fontsize=12, fontweight="bold", color=TEXT,
                                      ha="center", va="center")
        self._button([cx + 0.14, y, 0.03, 0.038], "▶" if key == "scenario" else "+", inc, name=key + "+")

    # ---------------- actions
    def _change(self, key, d):
        lo, hi = self.LIMITS[key]
        v = min(hi, max(lo, getattr(self, key) + d))
        if v == getattr(self, key):
            return
        setattr(self, key, v)
        if key == "speed":
            self.redraw()
        else:
            self.build()

    def prev_scenario(self):
        if self.scenario > 0:
            self.scenario -= 1
            self.build()

    def next_scenario(self):
        self.scenario += 1
        self.build()

    def random_scenario(self):
        self.scenario = random.randint(0, 9999)
        self.build()

    def build(self):
        """(Re)build the mission at t = 0 with the current settings; playback stops."""
        self.running = False
        self.m = Mission(self.scenario, "FULL", self.rooms, self.drones)
        self.log = EventLog(self.m)
        self.truth = MapView(self.ax_truth, self.m, "truth")
        self.belief = MapView(self.ax_belief, self.m, "belief")
        self.regions.set("truth", self.ax_truth.bbox, self.truth.artists)
        self.regions.set("belief", self.ax_belief.bbox, self.belief.artists)
        self.xmax = 120
        self.redraw()

    def reset(self):
        self.build()

    def toggle(self):
        if self.bench:
            return
        if self.ax_info.get_visible():           # starting closes the explanation panel
            self.toggle_info()
        if self.m.done:
            self.build()
        self.running = not self.running
        self.redraw()

    def _advance(self):
        self.m.step()
        self.log.update()

    def step_once(self):
        self.running = False
        if not self.m.done:
            self._advance()
        self.redraw()

    def _toggle_view(self, _label):
        self.show_person, self.show_plan = self.checks.get_status()
        self.redraw()

    # ---------------- "How it works" panel (covers the two maps while open)
    def _build_info(self):
        import textwrap
        fig = self.fig
        x0, y0, w, h = 0.012, 0.012, 0.573, 0.793
        self.ax_info = ax = fig.add_axes([x0, y0, w, h], facecolor=CARD, zorder=20)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(EDGE)
        ax.text(0.025, 0.965, "HOW IT WORKS", fontsize=15, fontweight="bold", color=TEXT, va="top",
                transform=ax.transAxes)
        ax.text(0.975, 0.965, "press ⓘ again or i to close", fontsize=9, color=MUTED, va="top", ha="right",
                transform=ax.transAxes)
        steps = [
            ("1  Where is the person?   Bayesian search map",
             "Every free cell starts equally likely. When a drone looks at a cell and does not see the person, that "
             "cell's probability is multiplied by (1 − p_detect) and the map is renormalised. p_detect is high up close "
             "and low far away or in smoke, so a quick look through smoke leaves some probability behind and the room "
             "is searched again. Blue on the lower map = still likely."),
            ("2  Which drone searches which rooms?   Genetic algorithm",
             "A chromosome is an order of all rooms, split between the drones. Fitness = expected time to find the "
             "person = Σ P(room) × time the drone reaches that room, so likely rooms that are close come first. "
             "60 candidates evolve for up to 100 generations (tournament selection, order crossover, inversion and "
             "swap mutation); the plan is re-made whenever the map changes."),
            ("3  How does a drone fly there?   A* search",
             "The building is a grid of 0.5 m cells; a drone moves in 8 directions (straight step = 1, diagonal "
             "step = √2). A* always expands the cell with the smallest f(n) = g(n) + h(n): g = distance flown so far, "
             "h = estimate of the distance still to go (the heuristic, right). Burning cells are never entered."),
            ("4  Rescue",
             "When a drone sees the person, the nearest drone flies to them and leads them back to the entrance along "
             "an A* path; the other drones leave."),
            ("Metrics",
             "Success = the person reaches the exit alive.  Efficiency = best possible rescue time ÷ our time, where "
             "best possible = an all-knowing rescuer that knows where the person is.  POS = probability that the "
             "search so far would have found the person."),
        ]
        y = 0.9
        for title, body in steps:
            ax.text(0.025, y, title, fontsize=11, fontweight="bold", color=BLUE if title[0].isdigit() else GREEN,
                    va="top", transform=ax.transAxes)
            lines = textwrap.fill(body, 78)
            ax.text(0.025, y - 0.037, lines, fontsize=9.6, color=TEXT2, va="top", transform=ax.transAxes,
                    linespacing=1.3)
            y -= 0.062 + 0.0245 * (lines.count("\n") + 1)

        # right column: the heuristic, with a worked example
        ax.text(0.6, 0.89, "The heuristic: octile distance", fontsize=11.5, fontweight="bold", color=FIRE, va="top",
                transform=ax.transAxes)
        d = fig.add_axes([x0 + 0.6 * w, y0 + 0.5 * h, 0.37 * w, 0.32 * h], facecolor=CARD, zorder=21)
        self.ax_oct = d
        for i in range(7):
            for j in range(5):
                d.add_patch(plt.Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, ec=EDGE, lw=0.8))
        path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (5, 3)]
        d.plot([p[0] for p in path], [p[1] for p in path], color=BLUE, lw=2.5, marker="o", ms=5, zorder=3)
        d.plot([0, 5], [0, 3], color=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=2)
        d.plot([0, 5, 5], [0, 0, 3], color=RED, lw=1.2, ls=(0, (1, 2)), zorder=2)
        for (xa, ya), (xb, yb) in zip(path, path[1:]):
            d.text((xa + xb) / 2 - 0.15, (ya + yb) / 2 + 0.2, "√2" if xb - xa and yb - ya else "1", fontsize=9,
                   color=TEXT, ha="center", zorder=4)
        d.text(0, -0.35, "start", fontsize=9, color=TEXT, ha="center", va="top")
        d.text(5, 3.35, "goal", fontsize=9, color=TEXT, ha="center", va="bottom")
        d.set_xlim(-0.6, 6.6)
        d.set_ylim(-0.9, 4.6)
        d.set_aspect("equal")
        d.set_axis_off()
        expl = [
            ("h = max(dx, dy) + (√2 − 1) · min(dx, dy)", BLUE),
            ("Example: dx = 5, dy = 3", TEXT),
            ("→ 3 diagonal (3 × 1.414) + 2 straight = 6.24 cells", TEXT),
            ("blue = octile 6.24 · grey = straight line 5.83", MUTED),
            ("red = 4-direction route 8", MUTED),
            ("", TEXT),
            ("Octile distance = the shortest route on an open", TEXT2),
            ("grid when you may move straight and diagonally:", TEXT2),
            ("go diagonally as far as possible, then straight.", TEXT2),
            ("Walls and fire only make the real route longer,", TEXT2),
            ("so h never overestimates (admissible) and A* still", TEXT2),
            ("returns the shortest path, while expanding about", TEXT2),
            ("70 % fewer cells than Dijkstra (no heuristic).", TEXT2),
        ]
        for i, (t, c) in enumerate(expl):
            ax.text(0.6, 0.47 - i * 0.034, t, fontsize=9.4, color=c, va="top", transform=ax.transAxes,
                    fontweight="bold" if c == BLUE else "normal")
        ax.set_visible(False)
        d.set_visible(False)

    def toggle_info(self):
        show = not self.ax_info.get_visible()
        if show:
            self.running = False
        for a in (self.ax_info, self.ax_oct):
            a.set_visible(show)
        for a in (self.ax_truth, self.ax_belief):
            a.set_visible(not show)
        self.b_info.label.set_text("✕  Close" if show else "ⓘ  How it works")
        self.redraw()

    def toggle_bench(self):
        if self.bench:                                   # stop a running benchmark
            self.bench = None
        else:
            self.running = False
            self.bench = dict(i=0, rows=[], first=self.scenario)
        self.redraw()

    def _bench_step(self):
        b = self.bench
        b["rows"].append(Mission(b["first"] + b["i"], "FULL", self.rooms, self.drones).run())
        b["i"] += 1
        if b["i"] >= BENCH_N:
            os.makedirs("results", exist_ok=True)
            with open(os.path.join("results", "benchmark.csv"), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(b["rows"][0]))
                wr.writeheader()
                wr.writerows(b["rows"])
            self.bench_done, self.bench = b["rows"], None

    def _key(self, e):
        {" ": self.toggle, "right": self.step_once, "r": self.reset, "n": self.random_scenario,
         "i": self.toggle_info}.get(
            e.key, lambda: None)()

    def _tick(self):
        """Timer callback. It reschedules itself only after drawing, so frames can never pile up."""
        try:
            if self.bench:
                self._bench_step()
                self.redraw(sync=True)
            elif self.running:
                t = time.perf_counter()
                for _ in range(self.speed):
                    self._advance()
                    if self.m.done:
                        self.running = False
                        break
                self.redraw(full=self.m.done, sync=True)   # a full redraw only when the mission ends
                self.frame_ms.append(1000 * (time.perf_counter() - t))
        finally:
            self.timer.interval = 30 if (self.running or self.bench) else 100
            self.timer.start()

    # ---------------- drawing
    def redraw(self, full=True, sync=False):
        """Update everything; then either redraw the whole window (full) or only the regions that changed."""
        m, w = self.m, self.m.world
        self.truth.update(m, show_person=self.show_person)
        self.belief.update(m, show_plan=self.show_plan)

        text, fc, tc = phase(m)
        if self.bench:
            text, fc, tc = f"BENCHMARK  {self.bench['i']} / {BENCH_N}", BLUE, "white"
        self.pill.set_text(f"●  {text}")
        self.pill.set_color(tc)
        self.pill.get_bbox_patch().set_facecolor(fc)

        pos = m.curve[-1][1] if m.curve else 0.0
        hcol = GREEN if w.health > 0.5 else (AMBER if w.health > 0.25 else RED)
        self.kpi["time"].set_text(f"{m.t} s")
        self.kpi["health"].set_text(f"{w.health:.2f}")
        self.kpi["health"].set_color(hcol)
        self.kpi["pos"].set_text(f"{100 * pos:.0f} %")
        self.kpi["inside"].set_text(f"{sum(u.state not in ('QUEUE', 'OUT') for u in m.units)} / {len(m.units)}")
        self.hbar.set_width(w.health)
        self.hbar.set_color(hcol)

        c = np.array(m.curve, float) if m.curve else np.empty((0, 3))
        self.ax_pos.line.set_data(c[:, 0], c[:, 1])
        self.ax_hp.line.set_data(c[:, 0], c[:, 2])
        while m.t + 5 > self.xmax:                  # the chart axes only change when time outgrows them
            self.xmax *= 2
            full = True
        for a in (self.ax_pos, self.ax_hp):
            a.set_xlim(0, self.xmax)
        self._report(m)
        self._heuristics(m)
        self._events()
        if full:
            self._bench_panel()
        for k in ("rooms", "drones"):
            self.lbl[k].set_text(str(getattr(self, k)))
        self.lbl["scenario"].set_text(f"#{self.scenario}")
        self.lbl["speed"].set_text(f"{self.speed}×")
        self.b_start.label.set_text("❚❚  Pause" if self.running else ("↻  Replay" if m.done else "▶  Start"))
        self.b_bench.label.set_text("■  Stop benchmark" if self.bench else f"Run {BENCH_N} missions")

        texts = {k: tuple((t.get_text(), t.get_color()) for t in arts)
                 for k, arts in (("report", self.report), ("heur", self.heur), ("events", self.event_text))}
        if full or not self.regions.ready:
            self._shown = texts
            if sync:
                self.fig.canvas.draw()
            else:
                self.fig.canvas.draw_idle()
            return
        names = ["truth", "belief", "kpi", "pill", "pos", "hp"]
        names += [k for k, v in texts.items() if self._shown.get(k) != v]
        self._shown = texts
        self.regions.refresh(names)

    def _report(self, m):
        if m.outcome == "rescued":
            res, col = "✓ RESCUED ALIVE", GREEN
        elif m.outcome:
            res, col = ("✗ PERSON LOST" if m.outcome == "casualty" else "✗ TIME OUT"), RED
        else:
            res, col = ("… running" if m.t else "… press Start"), TEXT2
        find = f"{m.t_find} s" if m.t_find is not None else "—"
        resc = f"{m.t_rescue} s" if m.t_rescue is not None else "—"
        eff = f"{100 * m.oracle_rescue / m.t_rescue:.0f} %" if m.t_rescue else ("0 %" if m.outcome else "—")
        lines = [(f"Result      {res}", col),
                 (f"Find        {find:>6}   best {m.oracle_find} s", TEXT2),
                 (f"Rescue      {resc:>6}   best {m.oracle_rescue} s", TEXT2),
                 (f"Efficiency  {eff:>6}", GREEN if m.outcome == "rescued" else TEXT2),
                 (f"Health      {m.world.health:6.2f}", TEXT2),
                 (f"Flown       {sum(u.dist for u in m.units) * 0.5:6.0f} m", TEXT2)]
        for t, (s, col) in zip(self.report, lines):
            t.set_text(s)
            t.set_color(col)

    def _heuristics(self, m):
        cm = c_min(m.cost) if m.cost is not None else 0.0
        n = m.nodes
        self.heur[0].set_text("never overestimates (each step costs ≥ its length)" if cm <= 1.0 + 1e-9 else
                              f"scaled by the cheapest step cost {cm:.1f}: still admissible")
        self.heur[1].set_text(f"{100 * (1 - n['tight'] / n['none']):.0f} % fewer nodes than Dijkstra (h = 0)"
                              if n["none"] else "node savings appear once drones plan")
        self.heur[2].set_text(f"= expected time to find  ·  {m.ga_calls} re-plans")

    def _events(self):
        for t, (when, line, kind, _) in zip(self.event_text, reversed(self.log.lines[-len(self.event_text):])):
            t.set_text(f"{when:>4d} s  {line}")
            t.set_color(EVENT_COLOR[kind])
            t.set_fontweight("bold" if kind in ("good", "bad") else "normal")
        for t in self.event_text[len(self.log.lines):]:
            t.set_text("")

    def _bench_panel(self):
        rows = self.bench["rows"] if self.bench else self.bench_done
        ax = self.ax_bench
        ax.clear()
        ax.set_axis_off()
        for t in self.bench_text:
            t.set_text("")
        if not rows:
            self.bench_text[0].set_text("running…" if self.bench else "not run yet")
            self.bench_text[0].set_color(MUTED)
            return
        ok = [r for r in rows if r["success"]]
        sr = 100 * len(ok) / len(rows)
        eff = 100 * np.mean([r["efficiency"] for r in rows])
        tf = np.mean([r["t_find"] for r in ok]) if ok else float("nan")
        lines = [(f"Success rate   {sr:3.0f} %  ({len(ok)}/{len(rows)})", GREEN if sr >= 90 else AMBER),
                 (f"Mean efficiency  {eff:3.0f} %", TEXT),
                 (f"Mean time to find  {tf:.0f} s", TEXT2),
                 ("running…" if self.bench else "saved: results/benchmark.csv", MUTED)]
        for t, (s, col) in zip(self.bench_text, lines):
            t.set_text(s)
            t.set_color(col)
        ax.bar(range(len(rows)), [r["efficiency"] for r in rows], width=0.75,
               color=[GREEN if r["success"] else RED for r in rows])
        ax.set_xlim(-0.6, BENCH_N - 0.4)
        ax.set_ylim(0, 1.05)
        ax.set_title("efficiency per mission", fontsize=8.5, color=MUTED, loc="left", pad=2)


def record(app, path, fps=10):
    """Run the current scenario to the end and write a GIF of the whole window."""
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=fps)
    with writer.saving(app.fig, path, dpi=60):
        writer.grab_frame(facecolor=BG)
        while not app.m.done:
            app._advance()
            app.redraw()
            writer.grab_frame(facecolor=BG)
        for _ in range(2 * fps):
            writer.grab_frame(facecolor=BG)
    print("wrote", path, "outcome:", app.m.outcome, "t =", app.m.t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=10)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--save", help="record the mission to this GIF instead of opening a window")
    a = ap.parse_args()
    app = App(a.scenario, a.rooms, a.drones, a.speed)
    if a.save:
        record(app, a.save)
        return
    if a.fullscreen:
        app.fig.canvas.manager.full_screen_toggle()
    plt.show()


if __name__ == "__main__":
    main()
