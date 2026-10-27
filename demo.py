"""Presentation app: drones search a burning building for a trapped person and lead them out.

    python demo.py                                   interactive window with controls
    python demo.py --fullscreen                      same, full screen
    python demo.py --seed 3 --save figs/demo.gif     record a run to a GIF (no window)

Right-hand panel: rooms and drones sliders, scenario seed / New scenario, the method, Start/Pause, Step,
Reset, speed, and view toggles; below them a live event feed.
Keys: space = start/pause, right arrow = step, r = reset, n = new scenario, f = full screen, q = quit.
"""
import argparse
import logging
import random
import sys

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from navsim.config import METHODS
from navsim.mission import Mission
from navsim.render import DARK, draw_map

# dark theme
BG, CARD, EDGE = "#1a1a19", "#242422", "#383835"
TEXT, TEXT2, MUTED = "#f0efec", "#c3c2b7", "#898781"
BTN, BTN_H, BLUE, BLUE_H = "#2c2c2a", "#383835", "#3987e5", "#2a78d6"
GREEN, AMBER, RED, FIRE = "#0ca30c", "#fab219", "#e66767", "#ff9a3c"
EVENT_COLOR = {"info": TEXT2, "fire": FIRE, "block": RED, "ga": "#86b6ef", "good": GREEN, "bad": RED}

METHOD_LABELS = {"FULL": "Fuzzy A* + Bayes + GA", "noFuzzy": "without fuzzy A*", "noBayes": "without Bayes",
                 "noGA": "without GA", "B0": "none of the three"}
FIG_W, FIG_H = 16, 9

_have = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [f for f in ("Segoe UI", "Segoe UI Symbol", "DejaVu Sans") if f in _have]
for k in ("keymap.home", "keymap.forward", "keymap.back", "keymap.save"):
    plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("r", "right", "left", "s", "n")]
plt.rcParams["toolbar"] = "None"
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # symbol font has no bold weight; harmless


def card(fig, x, y, w, h, title=None):
    fig.add_artist(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                  mutation_aspect=FIG_W / FIG_H, transform=fig.transFigure,
                                  fc=CARD, ec=EDGE, lw=0.8, zorder=-1))
    if title:
        fig.text(x + 0.012, y + h - 0.012, title, fontsize=10, fontweight="bold", color=MUTED, va="top")


def phase(m):
    """Phase text, pill colour, text colour."""
    if m.outcome == "rescued":
        return f"RESCUED IN {m.t_rescue} s", GREEN, "white"
    if m.outcome == "casualty":
        return "PERSON LOST", "#d03b3b", "white"
    if m.outcome == "timeout":
        return "TIME OUT", "#d03b3b", "white"
    if m.t_find is None:
        return "SEARCHING", BLUE, "white"
    if m.escort is not None and m.escort.state == "ESCORT":
        return "ESCORTING OUT", AMBER, "#0b0b0b"
    return "PERSON FOUND", AMBER, "#0b0b0b"


class EventLog:
    """Turns changes in the mission state into short, timestamped lines for the feed."""

    def __init__(self, m):
        self.m, self.lines = m, []
        b = m.b
        self.sites = b.door.copy()
        r0, r1 = b.corr_rows
        self.sites[r0:r1 + 1, b.cut_cols] = True
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
            self.lines[-1] = (t, f"GA re-planned the routes  x{n + 1}", "ga", n + 1)
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
            if u.state == "OUT" and self.state[i] == "EXIT":
                self.add(f"drone {u.id + 1} left the building", "info")
            self.state[i], self.bumps[i] = u.state, u.bumps
        if m.ga_calls > self.ga:
            self.add("GA re-planned the routes", "ga")
            self.ga = m.ga_calls
        if m.t_find is not None and not self.found:
            self.found = True
            who = f" by drone {m.escort.id + 1}" if m.escort else ""
            self.add(f"PERSON FOUND{who}", "good")
        if m.escort is not None and m.escort.state == "ESCORT" and not self.escorting:
            self.escorting = True
            self.add(f"drone {m.escort.id + 1} is leading them out", "good")
        if m.outcome == "rescued":
            self.add(f"RESCUED after {m.t_rescue} s", "good")
        elif m.outcome:
            self.add("person lost" if m.outcome == "casualty" else "time out", "bad")


class App:
    def __init__(self, seed=3, rooms=10, drones=3, method="FULL", speed=2):
        self.seed, self.rooms, self.drones, self.method, self.speed = seed, rooms, drones, method, speed
        self.running, self.show_person, self.show_routes = False, True, True
        fig = self.fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("Drone search & rescue")

        # ---- header, phase pill, KPI tiles
        fig.text(0.017, 0.958, "DRONE SEARCH & RESCUE", fontsize=21, fontweight="bold", color=TEXT, va="center")
        fig.text(0.017, 0.922, "Fuzzy A* path planning  ·  Bayesian search map  ·  GA allocation of rooms to drones"
                 "   —   one floor, one entrance that is also the only exit", fontsize=10.5, color=MUTED, va="center")
        self.pill = fig.text(0.733, 0.955, "", fontsize=13, fontweight="bold", ha="right", va="center",
                             bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.9", fc=BLUE, ec="none"))
        self.kpi = {}
        names = [("time", "TIME"), ("health", "PERSON HEALTH"), ("inside", "DRONES INSIDE"), ("ga", "GA RE-PLANS"),
                 ("ms", "DECISION TIME")]
        tw, gap, x0, y0, th = 0.1356, 0.01, 0.015, 0.795, 0.1
        for i, (key, label) in enumerate(names):
            x = x0 + i * (tw + gap)
            card(fig, x, y0, tw, th)
            fig.text(x + 0.012, y0 + th - 0.016, label, fontsize=9, fontweight="bold", color=MUTED, va="top")
            self.kpi[key] = fig.text(x + 0.012, y0 + 0.03, "", fontsize=23, fontweight="bold", color=TEXT, va="center")
            if key == "health":
                self.ax_health = fig.add_axes([x + 0.065, y0 + 0.022, tw - 0.08, 0.017])
            if key == "ms":
                fig.text(x + tw - 0.012, y0 + 0.03, "per 1 s tick", fontsize=9, color=MUTED, ha="right", va="center")

        # ---- map cards
        card(fig, 0.015, 0.415, 0.72, 0.368, "REALITY  —  smoke and fire, blocked doors, the trapped person")
        card(fig, 0.015, 0.015, 0.72, 0.39, "WHAT THE DRONES BELIEVE  —  probability of the person's location, "
                                             "planned routes")
        self.ax_truth = fig.add_axes([0.025, 0.425, 0.70, 0.31])
        self.ax_belief = fig.add_axes([0.025, 0.025, 0.70, 0.33])
        for ax in (self.ax_truth, self.ax_belief):
            ax.set_facecolor(CARD)
        self._legend(0.735, 0.783 - 0.008, [
            Line2D([], [], marker="s", ls="", color=GREEN, label="entrance / exit"),
            Line2D([], [], marker="x", ls="", color=RED, markeredgewidth=2, label="blocked"),
            Line2D([], [], marker="o", ls="", color=BLUE, markersize=8, label="drone"),
            Line2D([], [], marker="*", ls="", color="#ff5c5c", markersize=12, label="person")])
        self._legend(0.735, 0.405 - 0.008, [
            Patch(color="#6da7ec", label="likely location"),
            Line2D([], [], color=TEXT2, lw=1.3, ls=(0, (3, 2)), label="GA route"),
            Line2D([], [], color=TEXT2, lw=2, label="A* path"),
            Line2D([], [], marker="x", ls="", color=RED, markeredgewidth=2, label="believed blocked")])

        # ---- sidebar: scenario
        sx, sw = 0.75, 0.235
        card(fig, sx, 0.775, sw, 0.21, "SCENARIO")
        self.s_rooms = self._slider(0.905, "Rooms", 4, 20, rooms)
        self.s_drones = self._slider(0.86, "Drones", 1, 8, drones)
        fig.text(sx + 0.012, 0.811, "Seed", fontsize=10.5, color=TEXT2, va="center")
        self.seed_box = TextBox(fig.add_axes([0.79, 0.792, 0.06, 0.038]), "", initial=str(seed),
                                color=BTN, hovercolor=BTN_H, textalignment="center")
        self.seed_box.text_disp.set_color(TEXT)
        self.seed_box.text_disp.set_fontsize(11)
        self._edges(self.seed_box.ax)
        self.seed_box.on_submit(self._seed_submit)
        self._button([0.857, 0.792, 0.118, 0.038], "New scenario", self.new_scenario)

        # ---- sidebar: method
        card(fig, sx, 0.555, sw, 0.21, "METHOD")
        ax = fig.add_axes([0.758, 0.565, 0.22, 0.165], facecolor=CARD)
        ax.set_axis_off()
        keys = list(METHODS)
        n = len(keys)
        self.radio = RadioButtons(ax, [METHOD_LABELS[k] for k in keys], active=keys.index(method),
                                  activecolor=BLUE, label_props={"color": [TEXT] * n, "fontsize": [11] * n},
                                  radio_props={"s": [110] * n, "edgecolor": [TEXT2] * n})
        self.radio.on_clicked(lambda lab: self._set(method=keys[[METHOD_LABELS[k] for k in keys].index(lab)]))

        # ---- sidebar: playback
        card(fig, sx, 0.3, sw, 0.245, "PLAYBACK")
        self.b_start = self._button([0.76, 0.462, 0.105, 0.048], "▶  Start", self.toggle, primary=True)
        self._button([0.87, 0.462, 0.105, 0.048], "Step  ⏭", self.step_once)
        self._button([0.76, 0.408, 0.215, 0.044], "■  Stop / reset", self.reset)
        self.s_speed = self._slider(0.372, "Speed", 1, 10, speed)
        ax = fig.add_axes([0.758, 0.305, 0.22, 0.055], facecolor=CARD)
        ax.set_axis_off()
        self.checks = CheckButtons(ax, ["show the trapped person", "show planned routes"], [True, True],
                                   label_props={"color": [TEXT] * 2, "fontsize": [10.5] * 2},
                                   frame_props={"edgecolor": [TEXT2] * 2, "facecolor": [CARD] * 2, "s": [80] * 2},
                                   check_props={"facecolor": [BLUE] * 2, "s": [80] * 2})
        self.checks.on_clicked(self._toggle_view)

        # ---- sidebar: events
        card(fig, sx, 0.015, sw, 0.275, "EVENTS")
        self.event_text = [fig.text(sx + 0.012, 0.248 - i * 0.0235, "", fontsize=10, color=TEXT2, va="center")
                           for i in range(9)]
        fig.text(sx + sw / 2, 0.028, "space start/pause  ·  → step  ·  r reset  ·  n new  ·  f full screen",
                 fontsize=8.5, color=MUTED, ha="center", va="center")

        fig.canvas.mpl_connect("key_press_event", self._key)
        self.build()
        self.timer = fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.tick)
        self.timer.start()

    # ---- widget helpers
    def _legend(self, x, y, handles):
        self.fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(x - 0.006, y), ncol=len(handles),
                        frameon=False, fontsize=9.5, labelcolor=TEXT2, handletextpad=0.4, columnspacing=1.2)

    @staticmethod
    def _edges(ax, color=EDGE):
        for s in ax.spines.values():
            s.set_color(color)

    def _slider(self, y, label, lo, hi, val):
        ax = self.fig.add_axes([0.81, y, 0.13, 0.024], facecolor=CARD)
        s = Slider(ax, label, lo, hi, valinit=val, valstep=1, valfmt="%d", color=BLUE, track_color=EDGE,
                   initcolor="none", handle_style={"facecolor": TEXT, "edgecolor": BG, "size": 13})
        s.label.set_color(TEXT2)
        s.label.set_fontsize(10.5)
        s.valtext.set_color(TEXT)
        s.valtext.set_fontsize(11)
        s.valtext.set_x(1.08)                    # clear of the handle at the maximum
        s.on_changed(lambda v, lab=label: self._slider_changed(lab, int(v)))
        return s

    def _button(self, rect, text, cb, primary=False):
        b = Button(self.fig.add_axes(rect), text, color=BLUE if primary else BTN,
                   hovercolor=BLUE_H if primary else BTN_H)
        b.label.set_fontsize(11)
        b.label.set_color("white" if primary else TEXT)
        self._edges(b.ax, BLUE if primary else EDGE)
        b.on_clicked(lambda _e: cb())
        return b

    def _slider_changed(self, label, v):
        if label == "Speed":
            self.speed = v
        elif label == "Rooms" and v != self.rooms:
            self._set(rooms=v)
        elif label == "Drones" and v != self.drones:
            self._set(drones=v)

    def _seed_submit(self, text):
        try:
            seed = int(text)
        except ValueError:
            self.seed_box.set_val(str(self.seed))
            return
        if seed != self.seed:
            self._set(seed=seed)

    def _toggle_view(self, _label):
        self.show_person, self.show_routes = self.checks.get_status()
        self.redraw()

    def _key(self, e):
        if self.seed_box.capturekeystrokes:
            return
        {" ": self.toggle, "right": self.step_once, "r": self.reset, "n": self.new_scenario}.get(e.key, lambda: None)()

    # ---- actions
    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self.build()

    def build(self):
        """(Re)build the mission at t = 0 with the current settings; playback stops."""
        self.m = Mission(self.seed, self.method, self.rooms, self.drones)
        self.log = EventLog(self.m)
        self.running = False
        self.redraw()

    def reset(self):
        self.build()

    def new_scenario(self):
        self.seed = random.randint(0, 9999)
        self.seed_box.set_val(str(self.seed))
        self.build()

    def toggle(self):
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

    def tick(self):
        if not self.running:
            return
        for _ in range(self.speed):
            self._advance()
            if self.m.done:
                self.running = False
                break
        self.redraw()

    # ---- drawing
    def redraw(self):
        m, w = self.m, self.m.world
        draw_map(self.ax_truth, m, "truth", routes=False, show_person=self.show_person, theme=DARK, labels=True)
        draw_map(self.ax_belief, m, "belief", routes=self.show_routes, theme=DARK, labels=True)

        text, fc, tc = phase(m)
        self.pill.set_text(f"●  {text}")
        self.pill.set_color(tc)
        self.pill.get_bbox_patch().set_facecolor(fc)

        inside = sum(u.state not in ("QUEUE", "OUT") for u in m.units)
        ms = m.tick_ms[-1] if m.tick_ms else 0.0
        hcol = GREEN if w.health > 0.5 else (AMBER if w.health > 0.25 else RED)
        self.kpi["time"].set_text(f"{m.t} s")
        self.kpi["health"].set_text(f"{w.health:.2f}")
        self.kpi["health"].set_color(hcol)
        self.kpi["inside"].set_text(f"{inside} / {len(m.units)}")
        self.kpi["ga"].set_text(f"{m.ga_calls}")
        self.kpi["ms"].set_text(f"{ms:.0f} ms")
        ax = self.ax_health
        ax.clear()
        ax.barh(0, 1, color=EDGE, height=1)
        ax.barh(0, w.health, color=hcol, height=1)
        ax.set_xlim(0, 1)
        ax.set_axis_off()

        for t, (when, line, kind, _) in zip(self.event_text, reversed(self.log.lines[-len(self.event_text):])):
            t.set_text(f"{when:>4d} s   {line}")
            t.set_color(EVENT_COLOR[kind])
            t.set_fontweight("bold" if kind in ("good", "bad") else "normal")
        for t in self.event_text[len(self.log.lines):]:
            t.set_text("")
        self.b_start.label.set_text("❚❚  Pause" if self.running else ("↻  Replay" if m.done else "▶  Start"))
        self.fig.canvas.draw_idle()


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
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=10)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--method", default="FULL", choices=list(METHODS))
    ap.add_argument("--speed", type=int, default=2)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--save", help="record the run to this GIF instead of opening a window")
    a = ap.parse_args()
    app = App(a.seed, a.rooms, a.drones, a.method, a.speed)
    if a.save:
        record(app, a.save)
        return
    if a.fullscreen:
        app.fig.canvas.manager.full_screen_toggle()
    plt.show()


if __name__ == "__main__":
    main()
