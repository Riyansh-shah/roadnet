"""Drawing the building, ground truth and shared belief with matplotlib (used by demo.py and figures.py)."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# categorical slots in fixed order; methods keep their colour in every figure
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLOR = {"FULL": PALETTE[0], "noGA": PALETTE[1], "noBayes": PALETTE[2], "noFuzzy": PALETTE[3], "B0": PALETTE[4]}
METHOD_ORDER = ["FULL", "noFuzzy", "noBayes", "noGA", "B0"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
WALL, FURN = "#383835", "#c3c2b7"

# one-hue sequential ramps: smoke in orange, P(person) in blue
SMOKE = LinearSegmentedColormap.from_list("smoke", [SURFACE, "#f7c3a8", "#eb6834", "#a8401a", "#5c1f08"])
PROB = LinearSegmentedColormap.from_list("prob", [SURFACE, "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
UNIT_COLORS = [PALETTE[0], PALETTE[6], PALETTE[2], PALETTE[5], PALETTE[4], PALETTE[7], PALETTE[3], PALETTE[1]]


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _image(b, field, cmap, vmax):
    img = cmap(np.clip(field / max(vmax, 1e-12), 0, 1))
    img[b.solid & ~b.furn] = np.array([0x38, 0x38, 0x35, 255]) / 255
    img[b.furn] = np.array([0xc3, 0xc2, 0xb7, 255]) / 255
    return img


def draw_map(ax, m, layer="truth", vmax=None, routes=True, trails=True, legend=False, show_person=True):
    """layer = 'truth' (smoke, true blockages, the person) or 'belief' (P(person) or coverage, believed blockages).
    show_person=False hides the person on the truth layer until a drone has found them."""
    b, w, bel = m.b, m.world, m.belief
    ax.clear()
    ax.set_axis_off()
    if layer == "truth":
        ax.imshow(_image(b, w.h, SMOKE, 1.0), interpolation="nearest")
        blocked = np.argwhere(w.blocked)
    else:
        if bel.bayes:
            field = np.where(b.area_id >= 0, bel.pp, 0)
        else:
            field = np.where(b.area_id >= 0, ~bel.looked, 0).astype(float)
        ax.imshow(_image(b, field, PROB, vmax or field.max()), interpolation="nearest")
        blocked = np.argwhere((bel.p_block > 0.5) & ~b.solid)
    if len(blocked):
        ax.scatter(blocked[:, 1], blocked[:, 0], marker="x", s=28, c=CRITICAL, lw=1.6, zorder=4)
    ex = np.array(b.exits)
    ax.scatter(ex[:, 1], ex[:, 0], marker="s", s=40, c=GOOD, edgecolors="white", lw=1, zorder=4)

    for u in m.units:
        if u.pos is None:
            continue
        col = UNIT_COLORS[u.id % len(UNIT_COLORS)]
        if trails and len(u.trail) > 1:
            tr = np.array(u.trail)
            ax.plot(tr[:, 1], tr[:, 0], color=col, lw=1.2, alpha=0.55, zorder=3)
        if layer == "belief" and routes and u.state != "OUT":
            if len(u.path) > 1:
                p = np.array(u.path)
                ax.plot(p[:, 1], p[:, 0], color=col, lw=2, zorder=5)
            if u.state == "SEARCH":
                nodes = [u.pos] + [tuple(b.viewpoints[a]) for a in ([u.area] if u.area >= 0 else []) + u.route]
                nodes = np.array(nodes)
                ax.plot(nodes[:, 1], nodes[:, 0], color=col, lw=1.2, ls=(0, (3, 2)), zorder=5)
        if u.state != "OUT":
            ax.scatter([u.pos[1]], [u.pos[0]], s=110, c=col, edgecolors="white", lw=1.5, zorder=6)
            ax.text(u.pos[1], u.pos[0], str(u.id + 1), color="white", fontsize=7, fontweight="bold",
                    ha="center", va="center", zorder=7)
    if (layer == "truth" and show_person) or m.t_find is not None:
        pr, pc = w.person
        ax.scatter([pc], [pr], marker="*", s=220, c="#e34948" if w.health > 0 else INK, edgecolors="white",
                   lw=1, zorder=8)
    if legend:
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="s", ls="", color=GOOD, label="entrance / exit"),
             Line2D([], [], marker="x", ls="", color=CRITICAL, label="blocked" if layer == "truth" else "believed blocked"),
             Line2D([], [], marker="o", ls="", color=PALETTE[0], label="drone"),
             Line2D([], [], marker="*", ls="", color="#e34948", markersize=10, label="person")]
        ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4, frameon=False, fontsize=8,
                  labelcolor=INK2)


def status_line(m):
    w = m.world
    if m.outcome == "rescued":
        phase = f"RESCUED at t = {m.t_rescue}"
    elif m.outcome:
        phase = m.outcome.upper()
    elif m.t_find is None:
        phase = "searching"
    else:
        phase = f"found at t = {m.t_find}, " + ("escorting" if m.escort and m.escort.state == "ESCORT" else "approaching")
    ms = m.tick_ms[-1] if m.tick_ms else 0.0
    return (f"{m.method}   t = {m.t:3d} s   {phase}   person health {w.health:.2f}   "
            f"GA runs {m.ga_calls}   compute {ms:5.1f} ms/tick")
