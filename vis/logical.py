import arguebuf as ab
from pathlib import Path
import queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util import find_major_claim, normalize

inode_colors = {
    0: "#0212f9",
    1: "#020b82",
    2: "#5863fc",
}
attack_colors = {
    0: "#fc0505",
    1: "#aa0000",
    2: "#fc5353",
}
support_colors = {
    0: "#1dfc05",
    1: "#11ad00",
    2: "#52fc3f",
}


class NodeWrapper:
    def __init__(self, node, parent, x_pos, y_pos, width=256):
        self.node = node
        self.parent = parent
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width

    def __str__(self) -> str:
        return f"{self.node.id} ({self.node.label}) {self.y_pos} - {self.width}"


def render(graph: ab.Graph, path: Path, normalize_graph=True, dpi=50) -> None:
    if normalize_graph:
        normalize(graph)
    q = queue.Queue()
    major_claim = find_major_claim(graph)
    q.put(NodeWrapper(major_claim, None, 0, 0))
    nodes = []
    while not q.empty():
        n = q.get()
        nodes.append(n)
        children = graph.incoming_nodes(n.node)
        for i, c in enumerate(children):
            width = n.width / len(children)
            q.put(NodeWrapper(c, n, n.x_pos + i * width, n.y_pos + 1, width))
    max_height = nodes[-1].y_pos + 1
    height = 256 / max_height

    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)
    ax.set_aspect("equal")
    x_number = 0
    y_level = 0
    for n in nodes:
        color = inode_colors[x_number % 3]
        if isinstance(n.node, ab.SchemeNode):
            color = (
                attack_colors[x_number % 3]
                if n.node.label == "Attack"
                else support_colors[x_number % 3]
            )
        ax.add_patch(
            patches.Rectangle(
                (n.x_pos, n.y_pos * height),
                n.width,
                height,
                edgecolor="black",
                facecolor=color,
            )
        )
        x_number += 1
        if n.y_pos > y_level:
            y_level = n.y_pos
            x_number = 0

        # ax.text(n.x_pos + n.width / 2, -n.y_pos + rand, n.node.label, ha='center', va='center')

    ax.set_axis_off()
    # remove white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=dpi)
    plt.close()
