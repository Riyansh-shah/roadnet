"""Presentation app: an AI and a non-AI rescue team race on the same burning building.

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

from navsim.config import VISIBILITY, Config
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
        return fig.text(x + 0.01, y + h - 0.011, title, fontsize=10, fontweight="bold", color=MUTED, va="top")


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
        self.ga, self.found, self.escorting, self.ended = 0, False, False, False
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
            pass                                   # re-plans are not shown on screen
            self.ga = m.ga_calls
        if m.t_find is not None and not self.found:
            self.found = True
            self.add(f"PERSON FOUND by drone {m.escort.id + 1}" if m.escort else "PERSON FOUND", "good")
        if m.escort is not None and m.escort.state == "ESCORT" and not self.escorting:
            self.escorting = True
            self.add(f"drone {m.escort.id + 1} leading them out", "good")
        if m.outcome and not self.ended:          # the other team may still be running: log the end once
            self.ended = True
            if m.outcome == "rescued":
                self.add(f"RESCUED after {m.t_rescue} s", "good")
            else:
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


APPROACHES = (("AI", "FULL"), ("NON-AI", "NONAI"))
AI_TITLE = "AI APPROACH  ·  Bayesian search map + A* paths"
AI_BELIEF_TITLE = ("AI: WHAT IT BELIEVES  ·  brighter = person more likely  ·  white = what the drones see now  ·  "
                   "each miss: P × (1 − p_detect), then rescale")
VIS_NAMES = list(VISIBILITY)
ROWS = [  # (label, key, lower is better? None = no colouring)
    ("Status", "status", None),
    ("Time to find", "find", True),
    ("Time to rescue", "rescue", True),
    ("Efficiency", "eff", False),
    ("Distance flown", "dist", True),
    ("Area covered", "cover", None),
    ("Planner cells expanded", "nodes", True),
    ("Decision time / tick", "ms", True),
    ("Person health", "health", False),
]


def mission_values(m):
    """Raw numbers (for colouring) and display strings of one mission, for the comparison table."""
    lost = m.outcome is not None and m.outcome != "rescued"
    v = {
        "status": (None, "RESCUED" if m.outcome == "rescued" else ("LOST" if lost else
                   ("found" if m.t_find is not None else "searching"))),
        "find": (m.t_find, f"{m.t_find} s" if m.t_find is not None else "—"),
        "rescue": (m.t_rescue, f"{m.t_rescue} s" if m.t_rescue is not None else "—"),
        "eff": (m.oracle_rescue / m.t_rescue if m.t_rescue else (0.0 if lost else None),
                f"{100 * m.oracle_rescue / m.t_rescue:.0f} %" if m.t_rescue else ("0 %" if lost else "—")),
        "dist": (sum(u.dist for u in m.units) * 0.5, f"{sum(u.dist for u in m.units) * 0.5:.0f} m"),
        "cover": (m.coverage(), f"{100 * m.coverage():.0f} %"),
        "nodes": (m.plan_nodes, f"{m.plan_nodes:,}"),
        "ms": (float(np.mean(m.tick_ms)) if m.tick_ms else None,
               f"{np.mean(m.tick_ms):.1f} ms" if m.tick_ms else "—"),
        "health": (m.world.health, f"{m.world.health:.2f}"),
    }
    return v


class App:
    LIMITS = {"rooms": (4, 20), "drones": (1, 8), "speed": (1, 10)}

    def __init__(self, scenario=3, rooms=10, drones=3, speed=2, visibility="Clear"):
        self.scenario, self.rooms, self.drones, self.speed = scenario, rooms, drones, speed
        self.visibility = visibility
        self.running, self.show_person, self.show_plan, self.ai_belief = False, True, True, False
        self.bench, self.bench_done = None, None
        self.frame_ms, self.buttons = [], {}
        fig = self.fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("Drone search & rescue: AI vs non-AI")

        # ---------------- header + KPI tiles
        fig.text(0.014, 0.962, "DRONE SEARCH & RESCUE  ·  AI vs NON-AI", fontsize=19, fontweight="bold", color=TEXT,
                 va="center")
        fig.text(0.014, 0.928, "the same building, fire and person for both · one entrance that is also the only exit",
                 fontsize=10, color=MUTED, va="center")
        self.pill = fig.text(0.585, 0.958, "", fontsize=12, fontweight="bold", ha="right", va="center",
                             bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.9", fc=BLUE, ec="none"))
        self.kpi = {}
        tiles = [("time", "TIME"), ("AI", "AI"), ("NON-AI", "NON-AI"), ("vis", "VISIBILITY")]
        tw, gap, x0, y0, th = 0.1375, 0.008, 0.012, 0.815, 0.09
        for i, (key, label) in enumerate(tiles):
            x = x0 + i * (tw + gap)
            card(fig, x, y0, tw, th)
            fig.text(x + 0.01, y0 + th - 0.014, label, fontsize=8.5, fontweight="bold",
                     color={"AI": BLUE, "NON-AI": GREEN}.get(key, MUTED), va="top")
            self.kpi[key] = fig.text(x + 0.01, y0 + 0.027, "", fontsize=17, fontweight="bold", color=TEXT,
                                     va="center")
        self.kpi_rect = (x0, y0, 4 * tw + 3 * gap, th)

        # ---------------- the two maps
        self.ai_title = card(fig, 0.012, 0.415, 0.573, 0.39, AI_TITLE)
        card(fig, 0.012, 0.012, 0.573, 0.393, "NON-AI APPROACH  ·  room-by-room sweep (nearest room first) + "
                                              "plain shortest path (Dijkstra)")
        self.ax_ai = fig.add_axes([0.018, 0.42, 0.561, 0.355])
        self.ax_non = fig.add_axes([0.018, 0.017, 0.561, 0.358])

        # ---------------- comparison column
        mx, mw = 0.595, 0.205
        card(fig, mx, 0.4, mw, 0.585, "AI vs NON-AI  (live)")
        self.ax_pos = self._chart([mx + 0.032, 0.79, mw - 0.045, 0.125], "Probability searched (POS)")
        self.lines = {"AI": self.ax_pos.plot([], [], color=BLUE, lw=2, label="AI")[0],
                      "NON-AI": self.ax_pos.plot([], [], color=GREEN, lw=2, label="NON-AI")[0]}
        self.ax_pos.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=TEXT2)
        ty = 0.725
        fig.text(mx + 0.01, ty, "metric", fontsize=8.5, fontweight="bold", color=MUTED, va="center")
        fig.text(mx + 0.128, ty, "AI", fontsize=8.5, fontweight="bold", color=BLUE, va="center", ha="right")
        fig.text(mx + 0.198, ty, "NON-AI", fontsize=8.5, fontweight="bold", color=GREEN, va="center", ha="right")
        self.cells = {}
        for i, (label, key, _) in enumerate(ROWS):
            y = ty - 0.031 * (i + 1)
            fig.text(mx + 0.01, y, label, fontsize=9.3, color=TEXT2, va="center")
            self.cells[key] = (fig.text(mx + 0.128, y, "", fontsize=9.6, color=TEXT, va="center", ha="right"),
                               fig.text(mx + 0.198, y, "", fontsize=9.6, color=TEXT, va="center", ha="right"))
        fig.text(mx + 0.01, 0.412, "green = better of the two", fontsize=8, color=MUTED, va="center")

        card(fig, mx, 0.2, mw, 0.19, "HEURISTICS")
        fig.text(mx + 0.01, 0.355, "A*:  f(n) = g(n) + h(n)", fontsize=9.5, color=TEXT, va="top")
        fig.text(mx + 0.01, 0.335, "h(n) = octile distance (straight 1, diagonal √2)", fontsize=9.5, color=TEXT,
                 va="top")
        self.heur = [fig.text(mx + 0.01, 0.305 - i * 0.02, "", fontsize=9.2, color=TEXT2, va="top")
                     for i in range(2)]
        fig.text(mx + 0.01, 0.262, "Bayes:  after a miss, P(cell) × (1 − p_detect)", fontsize=9.5, color=TEXT,
                 va="top")
        self.heur.append(fig.text(mx + 0.01, 0.242, "", fontsize=9.2, color=TEXT2, va="top"))

        card(fig, mx, 0.012, mw, 0.178, "EVENTS  (AI mission)")
        self.event_text = [fig.text(mx + 0.01, 0.155 - i * 0.0215, "", fontsize=9.2, color=TEXT2, va="center")
                           for i in range(7)]

        # ---------------- controls column
        cx, cw = 0.81, 0.178
        card(fig, cx, 0.655, cw, 0.33, "SCENARIO")
        self.lbl = {}
        self._stepper(0.9, "Scenario", "scenario", self.prev_scenario, self.next_scenario)
        self._button([cx + 0.01, 0.852, cw - 0.02, 0.036], "🎲  Random scenario", self.random_scenario, name="random")
        self._stepper(0.8, "Rooms", "rooms", lambda: self._change("rooms", -1), lambda: self._change("rooms", +1))
        self._stepper(0.75, "Drones", "drones", lambda: self._change("drones", -1),
                      lambda: self._change("drones", +1))
        fig.text(cx + 0.01, 0.682, "Visibility", fontsize=10.5, color=TEXT2, va="center")
        self.b_vis = self._button([cx + 0.075, 0.663, cw - 0.085, 0.038], "", self.next_visibility,
                                  name="visibility")

        card(fig, cx, 0.39, cw, 0.255, "PLAYBACK")
        self.b_start = self._button([cx + 0.01, 0.575, cw - 0.02, 0.044], "▶  Start", self.toggle, primary=True,
                                    name="start")
        self._button([cx + 0.01, 0.527, (cw - 0.025) / 2, 0.04], "Step  ⏭", self.step_once, name="step")
        self._button([cx + 0.015 + (cw - 0.025) / 2, 0.527, (cw - 0.025) / 2, 0.04], "■  Reset", self.reset,
                     name="reset")
        self._stepper(0.485, "Speed", "speed", lambda: self._change("speed", -1), lambda: self._change("speed", +1))
        ax = fig.add_axes([cx + 0.006, 0.395, cw - 0.012, 0.08], facecolor=CARD)
        ax.set_axis_off()
        self.checks = CheckButtons(ax, ["show the trapped person", "show plans and paths", "AI map: what it believes"],
                                   [True, True, False],
                                   label_props={"color": [TEXT] * 3, "fontsize": [9.6] * 3},
                                   frame_props={"edgecolor": [TEXT2] * 3, "facecolor": [CARD] * 3, "s": [60] * 3},
                                   check_props={"facecolor": [BLUE] * 3, "s": [60] * 3})
        self.checks.on_clicked(self._toggle_view)

        self.b_info = self._button([cx, 0.338, cw, 0.044], "ⓘ  How it works", self.toggle_info, name="info")

        card(fig, cx, 0.012, cw, 0.315, "BENCHMARK")
        fig.text(cx + 0.01, 0.295, f"the same {BENCH_N} scenarios, both approaches,", fontsize=8.8, color=MUTED,
                 va="top")
        fig.text(cx + 0.01, 0.278, "current rooms, drones and visibility", fontsize=8.8, color=MUTED, va="top")
        self.b_bench = self._button([cx + 0.01, 0.215, cw - 0.02, 0.042], f"Run {BENCH_N} missions",
                                    self.toggle_bench, name="bench")
        self.bench_text = [fig.text(cx + 0.01, 0.19 - i * 0.024, "", fontsize=9.6, color=TEXT2, va="center")
                           for i in range(7)]
        self._build_info()
        self._build_results()

        # ---------------- blitting regions (the maps are added in build())
        self.regions = r = Regions(fig)
        r.set("kpi", r.rect(*self.kpi_rect), list(self.kpi.values()))
        r.set("pill", r.rect(0.3, 0.93, 0.29, 0.058), [self.pill])
        r.set("pos", self.ax_pos.bbox, list(self.lines.values()))
        r.set("table", r.rect(mx + 0.11, 0.425, mw - 0.112, 0.285), [t for pair in self.cells.values() for t in pair])
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
    def _chart(self, rect, title):
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

    def next_visibility(self):
        self.visibility = VIS_NAMES[(VIS_NAMES.index(self.visibility) + 1) % len(VIS_NAMES)]
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

    def _cfg(self):
        return Config(pd0=VISIBILITY[self.visibility])

    def _map(self, name):
        m, ax = self.ms[name], (self.ax_ai if name == "AI" else self.ax_non)
        return MapView(ax, m, "belief" if (name == "AI" and self.ai_belief) else "truth")

    def build(self):
        """(Re)build both missions on the same scenario at t = 0; playback stops."""
        self.running = False
        self.ms = {name: Mission(self.scenario, method, self.rooms, self.drones, self._cfg())
                   for name, method in APPROACHES}
        self.m = self.ms["AI"]
        self.log = EventLog(self.m)
        self.views = {name: self._map(name) for name in self.ms}
        self.regions.set("map_ai", self.ax_ai.bbox, self.views["AI"].artists)
        self.regions.set("map_non", self.ax_non.bbox, self.views["NON-AI"].artists)
        self.xmax = 120
        self.redraw()

    def reset(self):
        self.build()

    @property
    def done(self):
        return all(m.done for m in self.ms.values())

    def toggle(self):
        if self.bench:
            return
        self._close_overlays()
        if self.done:
            self.build()
        self.running = not self.running
        self.redraw()

    def _advance(self):
        for m in self.ms.values():
            if not m.done:
                m.step()
        self.log.update()

    def step_once(self):
        self.running = False
        if not self.done:
            self._advance()
        self.redraw()

    def _toggle_view(self, _label):
        self.show_person, self.show_plan, belief = self.checks.get_status()
        if belief != self.ai_belief:
            self.ai_belief = belief
            self.views["AI"] = self._map("AI")
            self.regions.set("map_ai", self.ax_ai.bbox, self.views["AI"].artists)
            self.ai_title.set_text(AI_BELIEF_TITLE if belief else AI_TITLE)
        self.redraw()

    # ---------------- overlays: "How it works" and benchmark results (both cover the maps)
    def _overlay(self):
        ax = self.fig.add_axes([0.012, 0.012, 0.573, 0.793], facecolor=CARD, zorder=20)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(EDGE)
        return ax

    def _build_info(self):
        import textwrap
        fig = self.fig
        x0, y0, w, h = 0.012, 0.012, 0.573, 0.793
        self.ax_info = ax = self._overlay()
        ax.text(0.025, 0.965, "HOW IT WORKS", fontsize=15, fontweight="bold", color=TEXT, va="top",
                transform=ax.transAxes)
        ax.text(0.975, 0.965, "press ⓘ again or i to close", fontsize=9, color=MUTED, va="top", ha="right",
                transform=ax.transAxes)
        steps = [
            ("1  Where is the person?   Bayesian search map",
             "Every free cell starts equally likely. When a drone looks at a cell and does not see the person, that "
             "cell's probability is multiplied by (1 − p_detect) and the map is renormalised. p_detect is high up close "
             "and low far away or in smoke, so a look that may have missed leaves probability behind and the room is "
             "searched again."),
            ("2  Which drone searches which rooms?   Genetic algorithm",
             "A chromosome is an order of all rooms, split between the drones. Fitness = expected time to find the "
             "person = Σ P(room) × time the drone reaches that room. 60 candidates evolve for up to 100 generations "
             "(tournament selection, order crossover, inversion and swap mutation); re-planned when the map changes."),
            ("3  How does a drone fly there?   A* search",
             "Grid of 0.5 m cells, 8 directions (straight step = 1, diagonal = √2). A* expands the cell with the "
             "smallest f(n) = g(n) + h(n): g = distance flown so far, h = the octile distance still to go (right)."),
            ("NON-AI approach (bottom map)",
             "No probabilities, no optimisation, no heuristic: each drone takes the nearest room nobody has searched, "
             "looks at every cell of it once, and moves on; paths come from plain Dijkstra search. A room it has "
             "looked at is never searched again, even if the look could have missed the person."),
            ("Metrics",
             "Success = the person reaches the exit alive. Efficiency = best possible rescue time ÷ actual time (best "
             "possible = an all-knowing rescuer). POS = probability the search so far would have found the person."),
        ]
        y = 0.9
        for title, body in steps:
            ax.text(0.025, y, title, fontsize=11, fontweight="bold",
                    color=BLUE if title[0].isdigit() else (GREEN if title.startswith("NON") else AMBER),
                    va="top", transform=ax.transAxes)
            lines = textwrap.fill(body, 78)
            ax.text(0.025, y - 0.037, lines, fontsize=9.6, color=TEXT2, va="top", transform=ax.transAxes,
                    linespacing=1.3)
            y -= 0.062 + 0.0245 * (lines.count("\n") + 1)

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
            ("75 % fewer cells than Dijkstra (no heuristic).", TEXT2),
        ]
        for i, (t, c) in enumerate(expl):
            ax.text(0.6, 0.47 - i * 0.034, t, fontsize=9.4, color=c, va="top", transform=ax.transAxes,
                    fontweight="bold" if c == BLUE else "normal")
        ax.set_visible(False)
        d.set_visible(False)

    def _build_results(self):
        self.ax_res = ax = self._overlay()
        self.res_title = ax.text(0.025, 0.965, "", fontsize=14, fontweight="bold", color=TEXT, va="top",
                                 transform=ax.transAxes)
        ax.text(0.975, 0.965, "press the benchmark button to close", fontsize=9, color=MUTED, va="top", ha="right",
                transform=ax.transAxes)
        heads = [(0.03, "metric", MUTED, "left"), (0.5, "AI", BLUE, "right"), (0.7, "NON-AI", GREEN, "right"),
                 (0.93, "difference", MUTED, "right")]
        for x, t, c, ha in heads:
            ax.text(x, 0.86, t, fontsize=11, fontweight="bold", color=c, ha=ha, va="center", transform=ax.transAxes)
        self.res_rows = []
        for i in range(10):
            y = 0.8 - i * 0.062
            self.res_rows.append([ax.text(x, y, "", fontsize=11, color=TEXT2, ha=ha, va="center",
                                          transform=ax.transAxes) for x, _, _, ha in heads])
        self.res_note = ax.text(0.03, 0.08, "", fontsize=9.2, color=MUTED, va="center", transform=ax.transAxes)
        ax.set_visible(False)

    def _close_overlays(self):
        changed = False
        if self.ax_info.get_visible():
            self.toggle_info(redraw=False)
            changed = True
        if self.ax_res.get_visible():
            self._show_results(False)
            changed = True
        return changed

    def _show_maps(self, show):
        for a in (self.ax_ai, self.ax_non):
            a.set_visible(show)

    def toggle_info(self, redraw=True):
        show = not self.ax_info.get_visible()
        if show:
            self.running = False
            self._show_results(False)
        for a in (self.ax_info, self.ax_oct):
            a.set_visible(show)
        self._show_maps(not show)
        self.b_info.label.set_text("✕  Close" if show else "ⓘ  How it works")
        if redraw:
            self.redraw()

    def _show_results(self, show):
        self.ax_res.set_visible(show)
        if show:
            for a in (self.ax_info, self.ax_oct):
                a.set_visible(False)
            self.b_info.label.set_text("ⓘ  How it works")
        self._show_maps(not show)

    def toggle_bench(self):
        if self.bench:                                   # stop a running benchmark
            self.bench = None
        elif self.ax_res.get_visible():                  # close the results
            self._show_results(False)
        else:
            self.running = False
            self.bench = dict(i=0, rows=[], first=self.scenario)
        self.redraw()

    def _bench_step(self):
        b = self.bench
        cfg = self._cfg()
        for name, method in APPROACHES:
            r = Mission(b["first"] + b["i"], method, self.rooms, self.drones, cfg).run()
            r["approach"], r["visibility"] = name, self.visibility
            b["rows"].append(r)
        b["i"] += 1
        if b["i"] >= BENCH_N:
            os.makedirs("results", exist_ok=True)
            with open(os.path.join("results", "benchmark.csv"), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(b["rows"][0]))
                wr.writeheader()
                wr.writerows(b["rows"])
            self.bench_done, self.bench = b["rows"], None
            self._fill_results()
            self._show_results(True)

    def _fill_results(self):
        rows = self.bench_done
        by = {name: [r for r in rows if r["approach"] == name] for name, _ in APPROACHES}
        both = [i for i in range(len(by["AI"])) if by["AI"][i]["success"] and by["NON-AI"][i]["success"]]

        def mean(name, key, idx=None, scale=1.0):
            vals = [by[name][i][key] for i in (idx if idx is not None else range(len(by[name])))]
            return scale * float(np.mean(vals)) if vals else float("nan")

        spec = [  # label, AI value, NON-AI value, format, lower is better
            ("People rescued alive", sum(r["success"] for r in by["AI"]), sum(r["success"] for r in by["NON-AI"]),
             "{:.0f} / " + str(len(by["AI"])), False),
            ("Efficiency (vs best possible)", mean("AI", "efficiency", scale=100), mean("NON-AI", "efficiency",
                                                                                     scale=100), "{:.0f} %", False),
            ("Time to find *", mean("AI", "t_find", both), mean("NON-AI", "t_find", both), "{:.1f} s", True),
            ("Time to rescue *", mean("AI", "t_rescue", both), mean("NON-AI", "t_rescue", both), "{:.1f} s", True),
            ("Distance flown *", mean("AI", "unit_dist", both, 0.5), mean("NON-AI", "unit_dist", both, 0.5),
             "{:.0f} m", True),
            ("Area covered", mean("AI", "coverage", scale=100), mean("NON-AI", "coverage", scale=100), "{:.0f} %",
             None),
            ("Planner cells expanded", mean("AI", "plan_nodes"), mean("NON-AI", "plan_nodes"), "{:,.0f}", True),
            ("Decision time per tick", mean("AI", "tick_ms_mean"), mean("NON-AI", "tick_ms_mean"), "{:.1f} ms",
             True),
            ("Person health at the end", mean("AI", "health"), mean("NON-AI", "health"), "{:.2f}", False),
        ]
        for cells, (label, a, n, fmt, lower) in zip(self.res_rows, spec):
            diff = a - n
            better = None if lower is None or abs(diff) < 1e-9 else ("AI" if (diff < 0) == lower else "NON-AI")
            vals = [label, fmt.format(a), fmt.format(n), ("+" if diff > 0 else "") + fmt.format(diff)]
            for t, v in zip(cells, vals):
                t.set_text(v)
            cells[1].set_color(GREEN if better == "AI" else TEXT)
            cells[2].set_color(GREEN if better == "NON-AI" else TEXT)
        for cells in self.res_rows[len(spec):]:
            for t in cells:
                t.set_text("")
        self.res_title.set_text(f"BENCHMARK  ·  scenarios #{rows[0]['seed']}-#{rows[-1]['seed']}  ·  "
                                f"{self.rooms} rooms, {self.drones} drones, visibility: {self.visibility}")
        self.res_note.set_text(f"* averaged over the {len(both)} scenarios both approaches rescued.  Green = better.  "
                               "Saved to results/benchmark.csv")

    def _key(self, e):
        {" ": self.toggle, "right": self.step_once, "r": self.reset, "n": self.random_scenario,
         "i": self.toggle_info}.get(e.key, lambda: None)()

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
                    if self.done:
                        self.running = False
                        break
                finished = any(m.done for m in self.ms.values()) and self._just_finished()
                self.redraw(full=self.done or finished, sync=True)
                self.frame_ms.append(1000 * (time.perf_counter() - t))
        finally:
            self.timer.interval = 30 if (self.running or self.bench) else 100
            self.timer.start()

    def _just_finished(self):
        """True once, at the frame where one of the two missions ends (to refresh the static parts)."""
        state = tuple(m.done for m in self.ms.values())
        changed = state != getattr(self, "_done_state", None)
        self._done_state = state
        return changed

    # ---------------- drawing
    def _status(self, m):
        if m.outcome == "rescued":
            return f"rescued {m.t_rescue} s", GREEN
        if m.outcome:
            return "LOST", RED
        if m.t_find is not None:
            return f"found {m.t_find} s", AMBER
        return "searching", TEXT

    def redraw(self, full=True, sync=False):
        """Update everything; then either redraw the whole window (full) or only the regions that changed."""
        for name, v in self.views.items():
            v.update(self.ms[name], show_person=self.show_person, show_plan=self.show_plan)
        ai, non = self.ms["AI"], self.ms["NON-AI"]

        text, fc, tc = phase(ai)
        text = "AI: " + text
        if self.bench:
            text, fc, tc = f"BENCHMARK  {self.bench['i']} / {BENCH_N}", BLUE, "white"
        self.pill.set_text(f"●  {text}")
        self.pill.set_color(tc)
        self.pill.get_bbox_patch().set_facecolor(fc)

        self.kpi["time"].set_text(f"{max(ai.t, non.t)} s")
        for name in ("AI", "NON-AI"):
            s, col = self._status(self.ms[name])
            self.kpi[name].set_text(s)
            self.kpi[name].set_color(col)
        self.kpi["vis"].set_text(self.visibility)

        for name, line in self.lines.items():
            c = np.array(self.ms[name].curve, float) if self.ms[name].curve else np.empty((0, 3))
            line.set_data(c[:, 0], c[:, 1])
        while max(ai.t, non.t) + 5 > self.xmax:      # the chart axes only change when time outgrows them
            self.xmax *= 2
            full = True
        self.ax_pos.set_xlim(0, self.xmax)

        va, vn = mission_values(ai), mission_values(non)
        for label, key, lower in ROWS:
            ta, tn = self.cells[key]
            (a, sa), (n, sn) = va[key], vn[key]
            ta.set_text(sa)
            tn.set_text(sn)
            better = None
            if lower is not None and a is not None and n is not None and abs(a - n) > 1e-9:
                better = "AI" if (a < n) == lower else "NON"
            ta.set_color(GREEN if better == "AI" else TEXT)
            tn.set_color(GREEN if better == "NON" else TEXT)

        self._heuristics(ai, non)
        self._events()
        if full:
            self._bench_panel()
        for k in ("rooms", "drones"):
            self.lbl[k].set_text(str(getattr(self, k)))
        self.lbl["scenario"].set_text(f"#{self.scenario}")
        self.lbl["speed"].set_text(f"{self.speed}×")
        self.b_vis.label.set_text(f"{self.visibility}  ▸")
        self.b_start.label.set_text("❚❚  Pause" if self.running else ("↻  Replay" if self.done else "▶  Start"))
        self.b_bench.label.set_text("■  Stop benchmark" if self.bench else
                                    ("✕  Close results" if self.ax_res.get_visible() else f"Run {BENCH_N} missions"))

        texts = {k: tuple((t.get_text(), t.get_color()) for t in arts)
                 for k, arts in (("table", [t for p in self.cells.values() for t in p]), ("heur", self.heur),
                                 ("events", self.event_text))}
        if full or not self.regions.ready:
            self._shown = texts
            if sync:
                self.fig.canvas.draw()
            else:
                self.fig.canvas.draw_idle()
            return
        names = ["map_ai", "map_non", "kpi", "pill", "pos"]
        names += [k for k, v in texts.items() if self._shown.get(k) != v]
        self._shown = texts
        self.regions.refresh(names)

    def _heuristics(self, ai, non):
        self.heur[0].set_text("never overestimates (each step costs ≥ its length)")
        if ai.plan_nodes and non.plan_nodes:
            self.heur[1].set_text(f"AI planner: {ai.plan_nodes:,} cells · NON-AI: {non.plan_nodes:,}")
        else:
            self.heur[1].set_text("cells expanded appear once drones plan")
        self.heur[2].set_text(f"then renormalise  ·  searched so far: {100 * (ai.curve[-1][1] if ai.curve else 0):.0f} %")

    def _events(self):
        for t, (when, line, kind, _) in zip(self.event_text, reversed(self.log.lines[-len(self.event_text):])):
            t.set_text(f"{when:>4d} s  {line}")
            t.set_color(EVENT_COLOR[kind])
            t.set_fontweight("bold" if kind in ("good", "bad") else "normal")
        for t in self.event_text[len(self.log.lines):]:
            t.set_text("")

    def _bench_panel(self):
        rows = self.bench["rows"] if self.bench else self.bench_done
        for t in self.bench_text:
            t.set_text("")
        if not rows:
            self.bench_text[0].set_text("running…" if self.bench else "not run yet")
            self.bench_text[0].set_color(MUTED)
            return
        n = len(rows) // 2
        lines = [(f"after {n} scenario{'s' if n != 1 else ''}:", MUTED)]
        for name, _ in APPROACHES:
            rs = [r for r in rows if r["approach"] == name]
            ok = sum(r["success"] for r in rs)
            lines.append((f"{name:7s} rescued {ok}/{len(rs)}", BLUE if name == "AI" else GREEN))
            lines.append((f"        efficiency {100 * np.mean([r['efficiency'] for r in rs]):.0f} %", TEXT2))
        lines.append(("running…" if self.bench else "full table on the left", MUTED))
        for t, (s, col) in zip(self.bench_text, lines):
            t.set_text(s)
            t.set_color(col)


def record(app, path, fps=10):
    """Run the current scenario (both approaches) to the end and write a GIF of the whole window."""
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=fps)
    with writer.saving(app.fig, path, dpi=60):
        writer.grab_frame(facecolor=BG)
        while not app.done:
            app._advance()
            app.redraw()
            writer.grab_frame(facecolor=BG)
        for _ in range(2 * fps):
            writer.grab_frame(facecolor=BG)
    print("wrote", path, "AI:", app.ms["AI"].outcome, app.ms["AI"].t_rescue, "| NON-AI:", app.ms["NON-AI"].outcome,
          app.ms["NON-AI"].t_rescue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=10)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--visibility", default="Clear", choices=VIS_NAMES)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--save", help="record the race to this GIF instead of opening a window")
    a = ap.parse_args()
    app = App(a.scenario, a.rooms, a.drones, a.speed, a.visibility)
    if a.save:
        record(app, a.save)
        return
    if a.fullscreen:
        app.fig.canvas.manager.full_screen_toggle()
    plt.show()


if __name__ == "__main__":
    main()
