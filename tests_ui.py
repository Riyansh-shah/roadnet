"""Headless test of the presentation app: clicks every control through matplotlib's event system, checks that it
does what it says, and that a frame stays within budget. Run: python tests_ui.py"""
import os
import random
import sys
import tempfile
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.argv.append("--save")              # makes demo.py pick the headless backend
import demo                            # noqa: E402
from matplotlib.backend_bases import KeyEvent, MouseEvent  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(name)


def click(app, name):
    ax = app.buttons[name].ax
    x, y = ax.transAxes.transform((0.5, 0.5))
    for kind in ("button_press_event", "button_release_event"):
        app.fig.canvas.callbacks.process(kind, MouseEvent(kind, app.fig.canvas, x, y, button=1))


def key(app, k):
    app.fig.canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", app.fig.canvas, k))


random.seed(1)
app = demo.App(scenario=3, rooms=10, drones=3)
app.fig.canvas.draw()
check("starts paused at t = 0 on scenario #3", app.m.t == 0 and not app.running and app.m.seed == 3)

click(app, "scenario+")
check("▶ goes to the next scenario", app.scenario == 4 and app.m.seed == 4)
click(app, "scenario-")
check("◀ goes back", app.scenario == 3 and app.m.seed == 3)
before = (app.m.world.fire_room, app.m.world.person)
click(app, "random")
check("random scenario builds a different one", app.scenario != 3 and (app.m.world.fire_room, app.m.world.person) != before,
      f"(#{app.scenario})")

click(app, "rooms+")
check("rooms + rebuilds with 11 rooms", app.rooms == 11 and app.m.b.n_rooms == 11)
for _ in range(15):
    click(app, "rooms+")
check("rooms stop at 20", app.rooms == 20 and app.m.b.n_rooms == 20)
for _ in range(20):
    click(app, "rooms-")
check("rooms stop at 4", app.rooms == 4 and app.m.b.n_rooms == 4)
for _ in range(6):
    click(app, "rooms+")
click(app, "drones+")
check("drones + rebuilds with 4 drones", app.drones == 4 and len(app.m.units) == 4)
for _ in range(10):
    click(app, "drones+")
check("drones stop at 8", len(app.m.units) == 8)
for _ in range(10):
    click(app, "drones-")
check("drones stop at 1", len(app.m.units) == 1)
for _ in range(2):
    click(app, "drones+")
click(app, "speed+")
check("speed + / − changes the speed", app.speed == 3)
click(app, "speed-")

click(app, "start")
check("Start runs", app.running)
t0 = app.m.t
for _ in range(15):
    app._tick()
check("running advances time", app.m.t > t0, f"(t = {app.m.t})")
frames = sorted(app.frame_ms)
median = frames[len(frames) // 2]
check("a typical frame (simulate + draw) stays under 150 ms", median < 150,
      f"(median {median:.0f} ms over {len(frames)} frames at 10 rooms, 3 drones)")
click(app, "start")
t1 = app.m.t
app._tick()
check("Pause stops time", not app.running and app.m.t == t1)
click(app, "step")
check("Step advances exactly one second", app.m.t == t1 + 1)
key(app, " ")
check("space starts and stops", app.running)
key(app, " ")
click(app, "reset")
check("Reset returns to t = 0", app.m.t == 0 and not app.running)

app.checks.set_active(0)
check("hiding the person works", not app.show_person)
app.checks.set_active(0)

click(app, "info")
check("'How it works' opens the explanation over the maps", app.ax_info.get_visible() and not app.ax_truth.get_visible())
click(app, "info")
check("clicking it again closes it", not app.ax_info.get_visible() and app.ax_truth.get_visible())
key(app, "i")
click(app, "start")
check("Start closes the explanation and runs", app.running and not app.ax_info.get_visible())
click(app, "start")
click(app, "reset")

click(app, "start")
t = time.time()
while not app.m.done and time.time() - t < 120:
    app._tick()
txt = app.report[0].get_text()
check("a mission runs to the end and fills the report card", app.m.done and ("RESCUED" in txt or "LOST" in txt or
      "TIME OUT" in txt), f"({txt.strip()}, t = {app.m.t})")
check("the phase indicator shows the outcome", any(w in app.pill.get_text() for w in ("RESCUED", "LOST", "TIME")))

demo.BENCH_N = 3
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    click(app, "bench")
    check("benchmark starts", app.bench is not None)
    t = time.time()
    while app.bench and time.time() - t < 300:
        app._tick()
    check("benchmark finishes and saves a CSV", app.bench_done is not None and len(app.bench_done) == 3 and
          os.path.exists(os.path.join("results", "benchmark.csv")),
          f"({app.bench_text[0].get_text().strip()})")
    os.chdir(cwd)

print("all UI checks passed" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
