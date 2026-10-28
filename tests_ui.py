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
check("starts paused at t = 0 on scenario #3, both teams on the same scenario",
      app.m.t == 0 and not app.running and app.ms["AI"].seed == app.ms["NON-AI"].seed == 3)
check("both teams face the same fire and person",
      app.ms["AI"].world.person == app.ms["NON-AI"].world.person and
      (app.ms["AI"].world.burning == app.ms["NON-AI"].world.burning).all())

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

click(app, "visibility")
check("visibility button switches both teams to smoky (0.7)", app.visibility == "Smoky" and
      all(m.cfg.pd0 == 0.7 for m in app.ms.values()))
click(app, "visibility")
click(app, "visibility")
check("and cycles back to clear", app.visibility == "Clear" and app.ms["AI"].cfg.pd0 == 0.95)

click(app, "start")
check("Start runs", app.running)
t0 = app.m.t
for _ in range(15):
    app._tick()
check("running advances both teams together", app.ms["AI"].t > t0 and app.ms["AI"].t == app.ms["NON-AI"].t,
      f"(t = {app.m.t})")
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
app.checks.set_active(2)
check("the AI map can switch to what the AI believes", app.ai_belief and app.views["AI"].layer == "belief")
app.checks.set_active(2)

click(app, "info")
check("'How it works' opens the explanation over the maps", app.ax_info.get_visible() and not app.ax_ai.get_visible())
click(app, "info")
check("clicking it again closes it", not app.ax_info.get_visible() and app.ax_ai.get_visible())
key(app, "i")
click(app, "start")
check("Start closes the explanation and runs", app.running and not app.ax_info.get_visible())
click(app, "start")
click(app, "reset")

click(app, "start")
t = time.time()
while not app.done and time.time() - t < 120:
    app._tick()
st = [c.get_text() for c in app.cells["status"]]
check("the race runs to the end and the table shows both outcomes", app.done and all(s in ("RESCUED", "LOST") for s in st),
      f"(AI {st[0]}, NON-AI {st[1]})")
check("the phase indicator shows the AI outcome", any(w in app.pill.get_text() for w in ("RESCUED", "LOST", "TIME")))

demo.BENCH_N = 3
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as tmp:
    os.chdir(tmp)
    click(app, "bench")
    check("benchmark starts", app.bench is not None)
    t = time.time()
    while app.bench and time.time() - t < 300:
        app._tick()
    check("benchmark runs both approaches on the same scenarios and saves a CSV",
          app.bench_done is not None and len(app.bench_done) == 6 and
          [r["seed"] for r in app.bench_done if r["approach"] == "AI"] ==
          [r["seed"] for r in app.bench_done if r["approach"] == "NON-AI"] and
          os.path.exists(os.path.join("results", "benchmark.csv")))
    check("the results table opens over the maps", app.ax_res.get_visible() and
          app.res_rows[0][1].get_text() != "", f"(rescued: AI {app.res_rows[0][1].get_text()}, "
          f"NON-AI {app.res_rows[0][2].get_text()})")
    os.chdir(cwd)

print("all UI checks passed" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
