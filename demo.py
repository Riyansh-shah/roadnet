"""Live demo: watch the units search, replan and escort in real time.

    python demo.py --seed 3 --rooms 10 --units 3 --method FULL
    python demo.py --seed 3 --save demo.gif          (writes a GIF instead of opening a window)

Top panel: ground truth (smoke/fire, true blockages, the hidden person).
Bottom panel: what the team believes (P(person) map or not-yet-covered cells, believed blockages,
each unit's allocated route dashed and its current A* path solid). Space pauses.
"""
import argparse

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from navsim.config import METHODS
from navsim.mission import Mission
from navsim.render import INK, INK2, draw_map, status_line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--rooms", type=int, default=10)
    ap.add_argument("--units", type=int, default=3)
    ap.add_argument("--method", default="FULL", choices=list(METHODS))
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--save", help="write a GIF to this path instead of showing a window")
    a = ap.parse_args()

    m = Mission(a.seed, a.method, a.rooms, a.units)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(11, 7.2))
    fig.patch.set_facecolor("white")
    title = fig.suptitle("", fontsize=10, color=INK, x=0.02, ha="left")

    def frames():
        yield 0
        while not m.done:
            m.step()
            yield m.t
        for _ in range(2 * a.fps):       # hold the last frame
            yield m.t

    def draw(_):
        draw_map(top, m, "truth", legend=True)
        draw_map(bot, m, "belief")
        top.set_title("ground truth", fontsize=9, color=INK2, loc="left")
        bot.set_title("shared belief: " + ("P(person)" if m.belief.bayes else "cells not yet covered")
                      + ", allocated routes (dashed), A* paths (solid)", fontsize=9, color=INK2, loc="left")
        title.set_text(status_line(m))

    anim = FuncAnimation(fig, draw, frames=frames, interval=1000 // a.fps, cache_frame_data=False,
                         save_count=2000, repeat=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if a.save:
        anim.save(a.save, writer=PillowWriter(fps=a.fps), dpi=70)
        print("wrote", a.save, "outcome:", m.outcome, "t =", m.t)
        return
    paused = [False]

    def on_key(e):
        if e.key == " ":
            (anim.resume if paused[0] else anim.pause)()
            paused[0] = not paused[0]

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    main()
