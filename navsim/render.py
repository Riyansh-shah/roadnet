"""Drawing the building, ground truth and shared belief with matplotlib (used by demo.py and figures.py).

Two themes: LIGHT for the report figures and DARK for the presentation app. Both use one-hue sequential ramps
(smoke in orange, P(person) in blue). On the dark theme brighter means more, so fire glows.
"""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgba

# categorical slots in fixed order; methods keep their colour in every figure
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PALETTE_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
METHOD_COLOR = {"FULL": PALETTE[0], "noGA": PALETTE[1], "noBayes": PALETTE[2], "noFuzzy": PALETTE[3], "B0": PALETTE[4]}
METHOD_ORDER = ["FULL", "noFuzzy", "noBayes", "noGA", "B0"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"

SMOKE = LinearSegmentedColormap.from_list("smoke", [SURFACE, "#f7c3a8", "#eb6834", "#a8401a", "#5c1f08"])
PROB = LinearSegmentedColormap.from_list("prob", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
_UNIT_SLOTS = [0, 6, 2, 5, 4, 7, 3, 1]          # drone k gets this palette slot
UNIT_COLORS = [PALETTE[i] for i in _UNIT_SLOTS]

LIGHT = dict(floor=SURFACE, wall="#383835", furn="#c3c2b7", smoke=SMOKE, prob=PROB, blocked=CRITICAL, exit=GOOD,
             ring="white", label=MUTED, person="#e34948", units=UNIT_COLORS, trail_alpha=0.55)
_DF = "#262624"
DARK = dict(floor=_DF, wall="#5a5954", furn="#3a3a37",
            smoke=LinearSegmentedColormap.from_list("smoke_d", [_DF, "#5c1f08", "#a8401a", "#d95926", "#ff8c2e",
                                                                "#ffc04d"]),
            prob=LinearSegmentedColormap.from_list("prob_d", [_DF, "#123b6e", "#1c5cab", "#3987e5", "#6da7ec"]),
            blocked="#e66767", exit=GOOD, ring="#f0efec", label="#898781", person="#ff5c5c",
            units=[PALETTE_DARK[i] for i in _UNIT_SLOTS], trail_alpha=0.7)


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _image(b, field, cmap, vmax, th):
    img = cmap(np.clip(field / max(vmax, 1e-12), 0, 1))
    img[b.solid & ~b.furn] = to_rgba(th["wall"])
    img[b.furn] = to_rgba(th["furn"])
    return img


def draw_map(ax, m, layer="truth", vmax=None, routes=True, trails=True, legend=False, show_person=True,
             theme=LIGHT, labels=False):
    """layer = 'truth' (smoke, true blockages, the person) or 'belief' (P(person) or coverage, believed blockages).
    show_person=False hides the person on the truth layer until a drone has found them.
    labels=True writes the room names and marks the entrance."""
    th = theme
    b, w, bel = m.b, m.world, m.belief
    ax.clear()
    ax.set_axis_off()
    if layer == "truth":
        ax.imshow(_image(b, w.h, th["smoke"], 1.0, th), interpolation="nearest")
        blocked = np.argwhere(w.blocked)
    else:
        if bel.bayes:
            field = np.where(b.area_id >= 0, bel.pp, 0)
        else:
            field = np.where(b.area_id >= 0, ~bel.looked, 0).astype(float)
        ax.imshow(_image(b, field, th["prob"], vmax or field.max(), th), interpolation="nearest")
        blocked = np.argwhere((bel.p_block > 0.5) & ~b.solid)
    if labels:
        for k in range(b.n_rooms):
            rr, cc = np.nonzero(b.room_id == k)
            ax.text(cc.min() + 0.2, rr.min() + 0.3, f"R{k}", color=th["label"], fontsize=7, ha="left", va="top",
                    zorder=2)
        er, ec = b.entrance
        ax.text(ec - 0.9, er, "IN / OUT", color=th["exit"], fontsize=7, fontweight="bold", rotation=90,
                ha="right", va="center", zorder=4)
    if len(blocked):
        ax.scatter(blocked[:, 1], blocked[:, 0], marker="x", s=28, c=th["blocked"], lw=1.6, zorder=4)
    ex = np.array(b.exits)
    ax.scatter(ex[:, 1], ex[:, 0], marker="s", s=40, c=th["exit"], edgecolors=th["ring"], lw=1, zorder=4)

    for u in m.units:
        if u.pos is None:
            continue
        col = th["units"][u.id % len(th["units"])]
        if trails and len(u.trail) > 1:
            tr = np.array(u.trail)
            ax.plot(tr[:, 1], tr[:, 0], color=col, lw=1.2, alpha=th["trail_alpha"], zorder=3)
        if layer == "belief" and routes and u.state != "OUT":
            if len(u.path) > 1:
                p = np.array(u.path)
                ax.plot(p[:, 1], p[:, 0], color=col, lw=2, zorder=5)
            if u.state == "SEARCH":
                nodes = [u.pos] + [tuple(b.viewpoints[a]) for a in ([u.area] if u.area >= 0 else []) + u.route]
                nodes = np.array(nodes)
                ax.plot(nodes[:, 1], nodes[:, 0], color=col, lw=1.2, ls=(0, (3, 2)), zorder=5)
        if u.state != "OUT":
            ax.scatter([u.pos[1]], [u.pos[0]], s=120, c=col, edgecolors=th["ring"], lw=1.3, zorder=6)
            ax.text(u.pos[1], u.pos[0], str(u.id + 1), color="white", fontsize=7, fontweight="bold",
                    ha="center", va="center", zorder=7)
    if (layer == "truth" and show_person) or m.t_find is not None:
        pr, pc = w.person
        col = th["person"] if w.health > 0 else th["label"]
        ax.scatter([pc], [pr], marker="o", s=480, c=col, alpha=0.18, lw=0, zorder=7)     # halo
        ax.scatter([pc], [pr], marker="*", s=230, c=col, edgecolors=th["ring"], lw=0.8, zorder=8)
    if legend:
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="s", ls="", color=th["exit"], label="entrance / exit"),
             Line2D([], [], marker="x", ls="", color=th["blocked"],
                    label="blocked" if layer == "truth" else "believed blocked"),
             Line2D([], [], marker="o", ls="", color=th["units"][0], label="drone"),
             Line2D([], [], marker="*", ls="", color=th["person"], markersize=10, label="person")]
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=8,
                  labelcolor=INK2)
