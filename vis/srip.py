from typing import Literal, Callable
import arguebuf as ab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
import math
from util import layerize, find_major_claim

#  TODO: Test SRIP1 and SRIP2

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


class Node:
    children = []
    x = 0
    y = 0
    w = 0
    text = ""
    node_type = "i"
    sticky = False
    span = 0
    index = 0

    def set_ll_width(self, x: float, y: float, w: float):
        self.x, self.y, self.w = x, y, w

    def __init__(
        self, label, type: Literal["i", "a", "s"], index: int, children: list
    ) -> None:
        self.text = label
        self.node_type = type
        self.children = children
        self.index = index

    def color(self):
        if self.node_type == "i":
            return inode_colors[self.index % 3]
        elif self.node_type == "a":
            return attack_colors[self.index % 3]
        else:
            return support_colors[self.index % 3]


@dataclass
class SRIP_Config:
    """
    Configuration for SRIP1 and SRIP2.

    Parameters:
    ---------
    Plot parameters:
    - dpi (int): The resolution of the plot. Default is 50.
    - W (int): The width of the plot. Default is 10.
    - H (int): The height of the plot. Default is 10.
    - inode_colors (dict): The colors for the inode nodes.
    - attack_colors (dict): The colors for the attack nodes.
    - support_colors (dict): The colors for the support nodes.
    ---------
    SRIP parameters:
    - gamma (int) Specifies the maximum horizontal gap between nodes. Default is 0. Note: A value too high will result in overlapping nodes.
    - rho (float): The space reclaiming parameter. Default is 0.5.
    - epsilon (float) in [0,W]: maximum width of a node
    - sigma (float) in [0,1]: sticky node shrinking factor
    - lambda_ (int): depth span of sticky node, i.e., how long do sticky nodes stay
    """

    dpi: int = 100
    W: int = 10
    H: int = 10

    gamma: float = 0
    rho: float = 0.5
    epsilon: float = W
    sigma = 0.25
    lambda_ = 1


def convert_from_AbstractNode_to_Node(
    graph: ab.Graph, node: ab.AbstractNode, idx=0
) -> Node:
    node_type = "i"
    if isinstance(node, ab.SchemeNode):
        node_type = "a" if node.label == "Attack" else "s"
    new_node = Node(
        node.label,
        node_type,
        idx,
        [
            convert_from_AbstractNode_to_Node(graph, c, i)
            for i, c in enumerate(graph.incoming_nodes(node))
        ],
    )
    return new_node


def _select_color(node, config, idx):
    if node.node_type == "i":
        return config.inode_colors[idx % 3]
    elif node.node_type == "a":
        return config.attack_colors[idx % 3]
    else:
        return config.support_colors[idx % 3]


def SRIP1(
    r: Node, graph: ab.Graph, path: Path, config: SRIP_Config = SRIP_Config()
) -> None:
    """
    Draws a space reclaiming icicle plots (SRIP1) from https://doi.org/10.1109/PacificVis48177.2020.4908 and saves it to path.

    Parameters:
    - r (Node): The root node (major claim).
    - num_layers (int): The number of layers. Obtain this as len(layerize(graph, find_major_claim(graph)).
    - path (Path): The path to save the plot.
    """
    plot = plt.figure()
    h = config.H / len(layerize(graph, find_major_claim(graph)))  # height of each layer
    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, config.W)
    ax.set_ylim(config.H, 0)
    ax.set_axis_off()
    # remove white border
    plot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    points = [(0, 0), (config.W, 0), (config.W, h), (0, h)]
    ax.add_patch(patches.Polygon(points, fill=False))
    r.set_ll_width(0, h, config.W)
    if len(r.children) > 0:
        _SRIP1_r(1, [r], len(r.children), config.W, h, config, ax)
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=config.dpi)


def _SRIP1_r(
    d: float,  # depth
    P: list[
        Node
    ],  # List of all parent nodes at depth d − 1 with attributes (x, y, w) set;
    m: int,  # m > 0: number of child nodes at depth d
    w: float,  # available horizontal width at depth d
    h: float,  # height of each layer
    config: SRIP_Config,
    ax,
):
    U = w - (m - 1) * config.gamma
    x, y = (config.W - w) / 2, (d + 1) * h
    P2, w2, m2 = [], 0, 0

    for p in P:
        p0 = (p.x, p.y)
        for c in p.children:
            p1 = (p0[0] + 1 / len(p.children) * p.w, p0[1])
            tri = U / m
            points = [p0, p1, (x + tri, y), (x, y)]
            ax.add_patch(patches.Polygon(points, fill=False))
            c.set_ll_width(x, y, tri)
            x, p0, n = x + tri + config.gamma, p1, len(c.children)
            if n > 0:
                P2, w2, m2 = P2 + [c], w2 + tri, m2 + n
    if m2 > 0:
        _SRIP1_r(d + 1, P2, m2, w2 + config.rho * (config.W - w2), h, config, ax)


def SRIP2(
    r: Node,
    graph: ab.Graph,
    path: Path,
    weight_func: Callable,
    config: SRIP_Config = SRIP_Config(),
):
    """
    Draws a space reclaiming icicle plots (SRIP1) from https://doi.org/10.1109/PacificVis48177.2020.4908 and saves it to path.

    Parameters:
    - r (Node): The root node (major claim).
    - num_layers (int): The number of layers. Obtain this as len(layerize(graph, find_major_claim(graph)).
    - path (Path): The path to save the plot.
    """
    plot = plt.figure()
    h = config.H / len(layerize(graph, find_major_claim(graph)))  # height of each layer
    _, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0 - 0.001, config.W + 0.001)
    ax.set_ylim(config.H + 0.001, 0 - 0.001)
    ax.set_axis_off()
    # remove white border
    plot.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # computeWeights()
    o = (config.W - config.epsilon) / 2

    points = [(o, 0), (config.W - o, 0), (config.W - o, h), (o, h)]
    ax.add_patch(patches.Polygon(points, fill=True, color=r.color()))
    r.set_ll_width(0, h, config.W)
    C = r.children
    if len(C) > 0:
        _SRIP2_r(
            1, [r], len(C), weight_func(C), config.W, 0, h, weight_func, config, ax
        )
    plt.savefig(path, bbox_inches=0, pad_inches=0, dpi=config.dpi)


def _SRIP2_r(
    d: float,  # depth
    P: list[
        Node
    ],  # List of all parent nodes at depth d − 1 with attributes (x, y, w) set;
    m: int,  # m > 0: number of child nodes at depth d
    A: float,  # total weight of child nodes at depth d
    w: float,  # available horizontal width at depth d
    g: float,  # width taken by sticky nodes at depth d
    h: float,  # height of each layer
    weight_func: Callable,
    config: SRIP_Config,
    ax,
):
    gamma_d = 0
    if m > 1:
        gamma_d = max(0, min(config.gamma, math.floor((w - g - m) / (m - 1))))
    if gamma_d == 0 and g > w:
        w = g
    U = w - g - (m - 1) * gamma_d
    x, y = (config.W - w) / 2, (d + 1) * h
    P2, m2, A2, w2, g2 = [], 0, 0, 0, 0
    for p in P:
        p0 = (p.x, p.y)
        if p.sticky:
            C = [p]
        else:
            C = p.children
        for c in C:
            if c.sticky:
                p1 = (p0[0] + p.w, p0[1])
                c.set_ll_width(x, y, config.sigma * p.w)
                x, c.span = x + c.w, c.span + 1
            else:
                p1 = (p0[0] + (weight_func(c) / weight_func(C) * p.w), p0[1])
                tri = weight_func(c) / A * U
                delta = min(tri, config.epsilon)
                o = (tri - delta) / 2
                points = [p0, p1, (x + o + delta, y), (x + o, y)]
                # print(points)
                ax.add_patch(patches.Polygon(points, fill=True, color=c.color()))
                c.set_ll_width(x + o, y, delta)
                x = x + tri + gamma_d
            p0, C2, n = p1, c.children, len(c.children)
            c.sticky = n == 0
            if c.sticky and c.span < config.lambda_:
                P2, g2 = P2 + [c], g2 + c.w
            if not c.sticky:
                P2, m2, A2, w2 = P2 + [c], m2 + n, A2 + weight_func(C2), w2 + tri
    if m2 > 0:
        _SRIP2_r(
            d + 1,
            P2,
            m2,
            A2,
            w2 + config.rho * (config.W - w2),
            config.sigma * g2,
            h,
            weight_func,
            config,
            ax,
        )
