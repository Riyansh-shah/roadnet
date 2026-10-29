"""Drawing one map: land, lakes, houses, roads and junctions (matplotlib)."""
import numpy as np
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Circle, Polygon

from .network import junction_angles
from .world import SIZE

LAND, LAND_EDGE = "#26402a", "#383835"
LAKE, LAKE_EDGE = "#2b5f8a", "#4f86b3"
ROAD, ROAD_CASING = "#e8e4d8", "#141413"
WALL, ROOF, OUTLINE = "#f0efec", "#e66767", "#141413"
JUNCTION = "#f5c542"
LABEL = "#f0efec"

HOUSE_W = 46.0      # house drawing size in metres


def house_shapes(x, y):
    w = HOUSE_W / 2
    body = [(x - w, y - w), (x + w, y - w), (x + w, y + w * 0.25), (x - w, y + w * 0.25)]
    roof = [(x - w * 1.25, y + w * 0.2), (x + w * 1.25, y + w * 0.2), (x, y + w * 1.35)]
    return Polygon(body, closed=True), Polygon(roof, closed=True)


class MapView:
    """One map axes. set_world() when the houses or lakes change; show() for every new network.

    With animated=True the network and houses are left out of normal redraws, so an app can keep the rest of the map
    as a cached background and redraw only dynamic_artists() each frame (blitting).
    """

    def __init__(self, ax, animated=False):
        self.ax = ax
        self.animated = animated
        ax.set_xlim(0, SIZE)
        ax.set_ylim(0, SIZE)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor(LAND)
        for s in ax.spines.values():
            s.set_color(LAND_EDGE)
        self.lakes = PatchCollection([], facecolor=LAKE, edgecolor=LAKE_EDGE, linewidth=1.5, zorder=1)
        self.casing = LineCollection([], colors=ROAD_CASING, linewidths=5.0, capstyle="round", zorder=2)
        self.roads = LineCollection([], colors=ROAD, linewidths=2.6, capstyle="round", zorder=3)
        self.walls = PatchCollection([], facecolor=WALL, edgecolor=OUTLINE, linewidth=1.0, zorder=5)
        self.roofs = PatchCollection([], facecolor=ROOF, edgecolor=OUTLINE, linewidth=1.0, zorder=6)
        for artist in (self.lakes, self.casing, self.roads, self.walls, self.roofs):
            ax.add_collection(artist)
        (self.glow,) = ax.plot([], [], "o", ms=15, color=JUNCTION, alpha=0.22, zorder=4)
        (self.junctions,) = ax.plot([], [], "o", ms=7.5, color=JUNCTION, mec=OUTLINE, mew=1.2, zorder=7)
        self.angle_texts = []
        self.show_angles = False
        self.houses = np.zeros((0, 2))
        for artist in self._network():
            artist.set_animated(animated)

    def _network(self):
        return [self.casing, self.roads, self.glow, self.walls, self.roofs, self.junctions]

    def dynamic_artists(self):
        """What changes from frame to frame, in drawing order."""
        return self._network() + self.angle_texts

    def set_world(self, world):
        self.houses = world.houses
        self.lakes.set_paths([Circle((cx, cy), r) for cx, cy, r in world.lakes])
        shapes = [house_shapes(x, y) for x, y in world.houses]
        self.walls.set_paths([b for b, _ in shapes])
        self.roofs.set_paths([r for _, r in shapes])

    def show(self, junctions, edges):
        """Draw the network over houses + junctions (edges index into that combined list)."""
        pts = np.vstack([self.houses, junctions]) if len(junctions) else self.houses
        segs = pts[edges] if len(edges) else np.zeros((0, 2, 2))
        self.casing.set_segments(segs)
        self.roads.set_segments(segs)
        self.junctions.set_data(junctions[:, 0], junctions[:, 1]) if len(junctions) else self.junctions.set_data([], [])
        self.glow.set_data(*self.junctions.get_data())
        for t in self.angle_texts:
            t.remove()
        self.angle_texts = []
        if self.show_angles:
            n = len(self.houses)
            for j in range(n, len(pts)):
                self._label_angles(pts, edges, j)

    def _label_angles(self, pts, edges, node):
        others = np.concatenate([edges[edges[:, 0] == node, 1], edges[edges[:, 1] == node, 0]])
        if len(others) < 2:
            return
        v = pts[others] - pts[node]
        heading = np.sort(np.arctan2(v[:, 1], v[:, 0]))
        for h, gap in zip(heading, junction_angles(pts, edges, node)):
            mid = h + np.radians(gap) / 2
            x, y = pts[node] + 62 * np.array([np.cos(mid), np.sin(mid)])
            self.angle_texts.append(self.ax.text(x, y, f"{gap:.0f}°", color=LABEL, fontsize=7.5, ha="center",
                                                 va="center", zorder=8, animated=self.animated))
