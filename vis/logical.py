import arguebuf as ab
from pathlib import Path
import queue
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from util import find_heighest_root_node, fig2img, layerize, ColorNode
from PIL import Image
import io

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
    def __init__(self, node, x_pos, y_pos, color_node: ColorNode, width: float = 10):
        self.node = node
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.width = width
        self.color_node = color_node

    def __str__(self) -> str:
        return f"{self.node.id} ({self.node.label}) {self.y_pos} - {self.width}"


def render(
    graph: ab.Graph,
    root: ab.AbstractNode,
    outer_height: float = 10,
    outer_width: float = 10,
    dpi: int = 50,
    normalize_height: bool = False,
) -> Image.Image:
    # TODO: make option with normalized height
    q = queue.Queue()
    q.put(NodeWrapper(root, 0, -1, ColorNode(root, []), outer_width))
    nodes = []
    color_node_map = {}
    # BFS
    min_height = 0
    while not q.empty():
        n = q.get()
        nodes.append(n)
        y_level = n.y_pos
        if y_level < min_height:
            min_height = y_level
        children = graph.incoming_nodes(n.node)
        previous_neighbor = None
        for i, c in enumerate(children):
            width = n.width / len(children)

            neighbors = [n.color_node]
            if previous_neighbor:
                neighbors.append(previous_neighbor)
            elif color_node_map.get((n.x_pos - width, y_level - 1)):
                neighbors.append(color_node_map[(n.x_pos - width, y_level - 1)])

            color_node = ColorNode(c, neighbors)
            node = NodeWrapper(c, n.x_pos + i * width, y_level - 1, color_node, width)
            q.put(node)
            previous_neighbor = color_node
            color_node_map[(n.x_pos, y_level - 1)] = color_node

    height = outer_height / abs(min_height)
    if normalize_height:
        height = outer_height / (len(layerize(graph, find_heighest_root_node(graph))))

    _, ax = plt.subplots(figsize=(outer_width, outer_height), dpi=dpi)
    ax.set_xlim(0, outer_width)
    ax.set_ylim(0, outer_height)
    # ax.set_aspect("equal")
    y_level = 0
    for n in nodes:
        color = n.color_node.get_color()
        ax.add_patch(
            patches.Rectangle(
                (n.x_pos, outer_height + n.y_pos * height),
                n.width,
                height,
                facecolor=color,
            )
        )

    ax.set_axis_off()
    # remove white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig2img(plt)
