"""Presentation app: drones search a burning building for a trapped person and lead them out.

    python demo.py                                   interactive window with controls
    python demo.py --fullscreen                      same, full screen
    python demo.py --seed 3 --save figs/demo.gif     record a run to a GIF (no window)

Controls on the right: rooms, drones and speed sliders; the method; the scenario seed (or New scenario);
Start/Pause, Step, Stop/reset; show the hidden person; show the planned routes.
Keys: space = start/pause, right arrow = step, r = reset, n = new scenario, f = full screen, q = quit.
"""
import argparse
import random
import sys

import matplotlib

if "--save" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox

from navsim.config import METHODS
from navsim.mission import Mission
from navsim.render import CRITICAL, GOOD, GRID, INK, INK2, MUTED, PALETTE, SURFACE, draw_map

METHOD_LABELS = {"FULL": "Fuzzy A* + Bayes + GA", "noFuzzy": "without fuzzy A*", "noBayes": "without Bayes",
                 "noGA": "without GA", "B0": "none of the three"}
PANEL = "#f3f2ee"

for k in ("keymap.home", "keymap.forward", "keymap.back", "keymap.save"):
    plt.rcParams[k] = [x for x in plt.rcParams[k] if x not in ("r", "right", "left", "s", "n")]
plt.rcParams["toolbar"] = "None"


def phase(m):
    """Short phase text and its colour."""
    if m.outcome == "rescued":
        return f"RESCUED in {m.t_rescue} s", GOOD
    if m.outcome == "casualty":
        return "PERSON LOST to the smoke", CRITICAL
    if m.outcome == "timeout":
        return "TIME OUT", CRITICAL
    if m.t_find is None:
        return "searching", INK
    if m.escort is not None and m.escort.state == "ESCORT":
        return f"found at {m.t_find} s, escorting out", PALETTE[0]
    return f"found at {m.t_find} s, drone on its way", PALETTE[0]


class App:
    def __init__(self, seed=3, rooms=10, drones=3, method="FULL", speed=2):
        self.seed, self.rooms, self.drones, self.method, self.speed = seed, rooms, drones, method, speed
        self.running, self.show_person, self.show_routes = False, True, True

        fig = self.fig = plt.figure(figsize=(16, 9), facecolor="white")
        if fig.canvas.manager is not None:
            fig.canvas.manager.set_window_title("Drone search & rescue")
        fig.text(0.02, 0.955, "Drone search & rescue in a burning building", fontsize=18, fontweight="bold", color=INK)
        fig.text(0.02, 0.925, "Fuzzy A* path planning  ·  Bayesian search map  ·  GA allocation of rooms to drones",
                 fontsize=11, color=INK2)
        self.status = fig.text(0.02, 0.885, "", fontsize=13, color=INK, fontweight="bold")
        self.ax_truth = fig.add_axes([0.02, 0.475, 0.70, 0.37])
        self.ax_belief = fig.add_axes([0.02, 0.065, 0.70, 0.37])
        fig.legend(handles=[
            Line2D([], [], marker="s", ls="", color=GOOD, label="entrance / exit"),
            Line2D([], [], marker="x", ls="", color=CRITICAL, label="blocked (truth) / believed blocked"),
            Line2D([], [], marker="o", ls="", color=PALETTE[0], markersize=9, label="drone"),
            Line2D([], [], marker="*", ls="", color=PALETTE[7], markersize=13, label="trapped person"),
            Line2D([], [], color=INK2, lw=1.2, ls=(0, (3, 2)), label="GA route"),
            Line2D([], [], color=INK2, lw=2, label="A* path")],
            loc="lower left", bbox_to_anchor=(0.02, 0.005), ncol=6, frameon=False, fontsize=10, labelcolor=INK2)

        # ---- control panel
        fig.add_artist(plt.Rectangle((0.745, 0.015), 0.24, 0.965, transform=fig.transFigure, color=PANEL, zorder=-1))
        fig.text(0.76, 0.945, "SETTINGS", fontsize=12, fontweight="bold", color=INK)
        self.s_rooms = self._slider(0.895, "Rooms", 4, 20, rooms)
        self.s_drones = self._slider(0.850, "Drones", 1, 8, drones)
        self.s_speed = self._slider(0.805, "Speed", 1, 10, speed)
        fig.text(0.76, 0.765, "Method", fontsize=11, color=INK2)
        ax = fig.add_axes([0.76, 0.585, 0.215, 0.175], facecolor=PANEL)
        ax.set_axis_off()
        keys = list(METHODS)
        self.radio = RadioButtons(ax, [METHOD_LABELS[k] for k in keys], active=keys.index(method),
                                  activecolor=PALETTE[0])
        for t in self.radio.labels:
            t.set_fontsize(11)
        self.radio.on_clicked(lambda lab: self._set(method=keys[[METHOD_LABELS[k] for k in keys].index(lab)]))

        fig.text(0.76, 0.545, "Scenario", fontsize=11, color=INK2)
        self.seed_box = TextBox(fig.add_axes([0.825, 0.535, 0.055, 0.04]), "", initial=str(seed),
                                textalignment="center")
        self.seed_box.on_submit(self._seed_submit)
        self._button([0.887, 0.535, 0.088, 0.04], "New scenario", self.new_scenario)

        self.b_start = self._button([0.76, 0.455, 0.105, 0.06], "Start", self.toggle, PALETTE[0], "white")
        self._button([0.87, 0.455, 0.105, 0.06], "Step", self.step_once)
        self._button([0.76, 0.385, 0.215, 0.055], "Stop / reset", self.reset)

        ax = fig.add_axes([0.76, 0.29, 0.215, 0.08], facecolor=PANEL)
        ax.set_axis_off()
        self.checks = CheckButtons(ax, ["show the trapped person", "show planned routes"], [True, True])
        for t in self.checks.labels:
            t.set_fontsize(11)
        self.checks.on_clicked(self._toggle_view)

        fig.text(0.76, 0.258, "LIVE", fontsize=12, fontweight="bold", color=INK)
        self.stats = fig.text(0.76, 0.24, "", fontsize=11, color=INK, va="top", family="monospace",
                              linespacing=1.45)
        self.ax_health = fig.add_axes([0.875, 0.169, 0.085, 0.016])
        fig.text(0.76, 0.035, "space start/pause · → step · r reset · n new · f full screen",
                 fontsize=9, color=MUTED)

        fig.canvas.mpl_connect("key_press_event", self._key)
        self.build()
        self.timer = fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.tick)
        self.timer.start()

    # ---- widget helpers
    def _slider(self, y, label, lo, hi, val):
        s = Slider(self.fig.add_axes([0.815, y, 0.13, 0.025], facecolor=GRID), label, lo, hi, valinit=val,
                   valstep=1, valfmt="%d", color=PALETTE[0], initcolor="none")
        s.label.set_fontsize(11)
        s.valtext.set_fontsize(11)
        s.on_changed(lambda v, lab=label: self._slider_changed(lab, int(v)))
        return s

    def _button(self, rect, text, cb, color="white", fg=INK):
        b = Button(self.fig.add_axes(rect), text, color=color, hovercolor=GRID if color == "white" else "#256abf")
        b.label.set_fontsize(11)
        b.label.set_color(fg)
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
        self.running = False
        self.redraw()

    def reset(self):
        self.build()

    def new_scenario(self):
        self.seed = random.randint(0, 9999)
        self.seed_box.set_val(str(self.seed))     # fires on_submit only on Enter, so rebuild here
        self.build()

    def toggle(self):
        if self.m.done:
            self.build()
        self.running = not self.running
        self.redraw()

    def step_once(self):
        self.running = False
        self.m.step()
        self.redraw()

    def tick(self):
        if not self.running:
            return
        for _ in range(self.speed):
            self.m.step()
            if self.m.done:
                self.running = False
                break
        self.redraw()

    # ---- drawing
    def redraw(self):
        m, w = self.m, self.m.world
        draw_map(self.ax_truth, m, "truth", routes=False, show_person=self.show_person)
        draw_map(self.ax_belief, m, "belief", routes=self.show_routes)
        self.ax_truth.set_title("What is really happening: smoke and fire (orange), blocked doors, the trapped person",
                                fontsize=11, color=INK2, loc="left")
        what = "P(person) in blue, darker = more likely" if m.belief.bayes else "blue = not yet covered"
        self.ax_belief.set_title(f"What the drones believe: {what}", fontsize=11, color=INK2, loc="left")

        text, col = phase(m)
        self.status.set_text(f"t = {m.t:3d} s      {text}")
        self.status.set_color(col)
        inside = sum(u.state not in ("QUEUE", "OUT") for u in m.units)
        ms = m.tick_ms[-1] if m.tick_ms else 0.0
        self.stats.set_text("\n".join([
            f"time           {m.t} s",
            f"drones inside  {inside} / {len(m.units)}",
            "person health",
            f"GA replans     {m.ga_calls}",
            f"decision time  {ms:.0f} ms",
            f"building       {m.b.n_rooms} rooms, fire in R{w.fire_room}",
            f"method         {self.method}",
        ]))
        ax = self.ax_health
        ax.clear()
        ax.barh(0, 1, color=GRID, height=1)
        ax.barh(0, w.health, color=GOOD if w.health > 0.5 else (PALETTE[3] if w.health > 0.25 else CRITICAL), height=1)
        ax.text(1.04, 0, f"{w.health:.2f}", va="center", fontsize=10, color=INK, transform=ax.get_yaxis_transform())
        ax.set_xlim(0, 1)
        ax.set_axis_off()
        self.b_start.label.set_text("Pause" if self.running else ("Replay" if m.done else "Start"))
        self.fig.canvas.draw_idle()


def record(app, path, every=1, fps=10):
    """Run the current scenario to the end and write a GIF of the whole window."""
    from matplotlib.animation import PillowWriter
    writer = PillowWriter(fps=fps)
    with writer.saving(app.fig, path, dpi=60):
        writer.grab_frame()
        while not app.m.done:
            for _ in range(every):
                app.m.step()
            app.redraw()
            writer.grab_frame()
        for _ in range(2 * fps):
            writer.grab_frame()
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
