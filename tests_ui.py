"""Headless UI test: clicks every button, edits houses with the mouse, presses the keys and runs a race.

    python tests_ui.py

Events go through the real matplotlib canvas (pixel positions), so button hit-testing is exercised too.
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent

import demo
from roadnet.world import random_world

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {len(results):2d}. {name}" + (f"  ({detail})" if detail else ""))


def mouse(app, kind, px, button=1):
    MouseEvent(kind, app.fig.canvas, px[0], px[1], button)._process()


def click(app, px, button=1):
    mouse(app, "button_press_event", px, button)
    mouse(app, "button_release_event", px, button)


def press(app, button):
    click(app, button.ax.transAxes.transform((0.5, 0.5)))


def key(app, k):
    KeyEvent("key_press_event", app.fig.canvas, k)._process()


def on_map(app, x, y, view=0):
    return app.views[view].ax.transData.transform((x, y))


app = demo.App(random_world(12, 9))
app.fig.canvas.draw()

# buttons
press(app, app.buttons[0])
check("Triangle button loads the triangle", app.world.name == "triangle" and app.world.n == 3)
press(app, app.buttons[1])
check("Square button loads the square", app.world.name == "square" and app.world.n == 4)
press(app, app.buttons[2])
check("Random button makes a new random map", app.world.name.startswith("random") and app.world.n == 12)
press(app, app.buttons[4])
check("houses + gives 13 houses", app.world.n == 13 and app.houses_text.get_text() == "13 houses")
press(app, app.buttons[3])
press(app, app.buttons[3])
check("houses − twice gives 11 houses", app.world.n == 11)
press(app, app.b_lakes)
check("Lakes on adds lakes and keeps the houses", len(app.world.lakes) > 0 and app.world.n == 11,
      f"{len(app.world.lakes)} lakes")
press(app, app.b_lakes)
check("Lakes off removes them", len(app.world.lakes) == 0)
speed = app.speed
press(app, app.buttons[-4])            # speed +
check("speed + raises the speed", app.speed == speed + 1, app.speed_text.get_text())
press(app, app.buttons[-5])            # speed −
check("speed − lowers it again", app.speed == speed)
press(app, app.b_step)
check("Step does one evaluation per climber", all(s is not None for s in app.race.snaps) and not app.running,
      f"evals {[s.evals for s in app.race.snaps]}")
press(app, app.b_reset)
check("Reset clears the race", not app.race.started)
press(app, app.b_angles)
check("Angles toggle switches labels on", all(v.show_angles for v in app.views))
press(app, app.b_angles)
press(app, app.b_run)
check("Solve starts the race", app.running and app.b_run.label.get_text().endswith("Pause"))
press(app, app.b_run)
check("Pause stops it", not app.running)

# info panel
press(app, app.b_info)
n = app.world.n
click(app, on_map(app, 600, 600))
check("How it works panel opens and blocks map clicks", app.info.get_visible() is False and app.world.n == n)
key(app, "i")
key(app, "i")
check("i opens and closes the panel", not app.info.get_visible())

# mouse editing
press(app, app.b_clear)
check("Clear map removes every house", app.world.n == 0 and app.counters[0].get_text() == "add at least 3 houses")
for x, y in [(200, 200), (1000, 250), (600, 1000), (650, 500)]:
    click(app, on_map(app, x, y, view=1))
check("clicking a map adds houses", app.world.n == 4, f"{app.world.n} houses")
mouse(app, "button_press_event", on_map(app, 650, 500))
mouse(app, "motion_notify_event", on_map(app, 640, 420))
mouse(app, "button_release_event", on_map(app, 640, 420))
moved = np.abs(app.world.houses[3] - [640, 420]).max()
check("dragging moves a house", moved < 2 and app.world.n == 4, f"now at {app.world.houses[3].round()}")
click(app, on_map(app, 640, 420, view=2), button=3)
check("right-click deletes a house", app.world.n == 3)

# keys
key(app, "n")
check("n makes a new random map", app.world.name.startswith("random"))
key(app, "l")
check("l toggles lakes", len(app.world.lakes) > 0)
key(app, "a")
check("a toggles angle labels", app.views[0].show_angles)

# a full race, with frame time
app.preset("square")
key(app, "a")
frames = []
key(app, " ")
while app.race.snaps[0] is None or not app.race.finished:
    t = time.perf_counter()
    app.tick()                         # one animation frame: advance + redraw
    frames.append(time.perf_counter() - t)
best = [s.best.cost for s in app.race.snaps]
check("every climber finishes the square at the H (2,732 m)", all(abs(b - 2732.05) < 1 for b in best),
      ", ".join(f"{b:,.1f}" for b in best))
check("frame time under 150 ms", np.mean(frames) < 0.15,
      f"mean {1000 * np.mean(frames):.0f} ms, max {1000 * max(frames):.0f} ms over {len(frames)} frames")
mouse(app, "motion_notify_event", app.chart.transData.transform((600, 2800)))
check("hovering the chart reads out the three climbers", app.subtitle.get_text().startswith("at 600 evals"),
      app.subtitle.get_text())

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
