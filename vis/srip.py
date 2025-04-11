from typing import Callable, cast
import arguebuf as ab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from dataclasses import dataclass
import math
from util import layerize, find_heighest_root_node, fig2img, ColorNode
from PIL import Image


class Node:
    children: list["Node"] = []
    x: float = 0
    y: float = 0
    w: float = 0
    node: ab.AbstractNode
    sticky: bool = False
    span: int = 0
    index: int = 0
    color_node: ColorNode | None

    def set_ll_width(self, x: float, y: float, w: float):
        self.x, self.y, self.w = x, y, w

    def __init__(
        self,
        label: str,
        index: int,
        children: list,
        node: ab.AbstractNode,
        color_node: ColorNode | None = None,
    ) -> None:
        self.text = label
        self.node_type = type
        self.children = children
        self.index = index
        self.color_node = color_node
        self.node = node

    def set_color_node(self, color_node: ColorNode):
        self.color_node = color_node


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
    W: float = 10
    H: float = 10

    gamma: float = 0
    rho: float = 0.5
    epsilon: float = W
    sigma = 0.25
    lambda_ = 1
    normalize_height: bool = False


def convert_from_AbstractNode_to_Node(
    graph: ab.Graph, node: ab.AbstractNode, color_node: ColorNode | None = None, idx=0
) -> Node:
    return Node(
        node.label,
        idx,
        [
            convert_from_AbstractNode_to_Node(graph, c, None, i)
            for i, c in enumerate(graph.incoming_nodes(node))
        ],
        node,
        color_node,
    )


def default_weight(x: Node | list[Node]):
    return 1 if isinstance(x, Node) else len(x)


def SRIP1(
    r: ab.AbstractNode, graph: ab.Graph, config: SRIP_Config = SRIP_Config()
) -> Image.Image:
    """
    Draws a space reclaiming icicle plots (SRIP1) from https://doi.org/10.1109/PacificVis48177.2020.4908 and saves it to path.

    Parameters:
    - r (Node): The root node (major claim).
    - num_layers (int): The number of layers. Obtain this as len(layerize(graph, find_major_claim(graph)).
    - path (Path): The path to save the plot.
    """
    root_for_height = find_heighest_root_node(graph) if config.normalize_height else r
    h = config.H / len(layerize(graph, root_for_height))  # height of each layer
    r_node = convert_from_AbstractNode_to_Node(graph, r, ColorNode(r, []))
    _, ax = plt.subplots(figsize=(config.W, config.H))
    ax.set_xlim(0, config.W)
    ax.set_ylim(config.H, 0)
    ax.set_axis_off()
    # remove white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    points = [(0, 0), (config.W, 0), (config.W, h), (0, h)]
    ax.add_patch(patches.Polygon(points, fill=False))
    r_node.set_ll_width(0, h, config.W)
    if len(r_node.children) > 0:
        _SRIP1_r(1, [r_node], len(r_node.children), config.W, h, config, ax)
    return fig2img(plt)


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
    color_node_map={},
):
    U = w - (m - 1) * config.gamma
    x, y = (config.W - w) / 2, (d + 1) * h
    P2, w2, m2 = [], 0, 0

    for p in P:
        p0 = (p.x, p.y)
        previous_neighbor = None
        for c in p.children:
            neighbors = [p.color_node]
            if previous_neighbor:
                neighbors.append(previous_neighbor)
            elif color_node_map.get((x - tri, y)):
                neighbors.append(color_node_map[(x - tri, y)])

            p1 = (p0[0] + 1 / len(p.children) * p.w, p0[1])
            tri = U / m
            points = [p0, p1, (x + tri, y), (x, y)]
            ax.add_patch(patches.Polygon(points, fill=False))
            c.set_ll_width(x, y, tri)
            x, p0, n = x + tri + config.gamma, p1, len(c.children)
            if n > 0:
                P2, w2, m2 = P2 + [c], w2 + tri, m2 + n
            previous_neighbor = c.color_node
            color_node_map[(x, y)] = c.color_node
    if m2 > 0:
        _SRIP1_r(d + 1, P2, m2, w2 + config.rho * (config.W - w2), h, config, ax)


def SRIP2(
    r: ab.AtomNode,
    graph: ab.Graph,
    weight_func: Callable,
    config: SRIP_Config = SRIP_Config(),
) -> Image.Image:
    """
    Draws a space reclaiming icicle plots (SRIP2) from https://doi.org/10.1109/PacificVis48177.2020.4908 and saves it to path.

    Parameters:
    - r (Node): The root node (major claim).
    - num_layers (int): The number of layers. Obtain this as len(layerize(graph, find_major_claim(graph)).
    - path (Path): The path to save the plot.
    """
    root_for_height = find_heighest_root_node(graph) if config.normalize_height else r
    r_node = convert_from_AbstractNode_to_Node(graph, r, ColorNode(r, []))
    h = config.H / len(layerize(graph, root_for_height))  # height of each layer
    _, ax = plt.subplots(figsize=(config.W, config.H))
    ax.set_xlim(0, config.W)
    ax.set_ylim(config.H, 0)
    ax.set_axis_off()
    # remove white border
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # computeWeights()
    offset = (config.W - config.epsilon) / 2

    points = [(offset, 0), (config.W - offset, 0), (config.W - offset, h), (offset, h)]
    if isinstance(r_node.color_node, ColorNode):
        ax.add_patch(
            patches.Polygon(points, fill=True, color=r_node.color_node.get_color())
        )
    r_node.set_ll_width(0, h, config.W)
    C = r_node.children
    if len(C) > 0:
        _SRIP2_r(
            1,
            [r_node],
            len(C),
            weight_func(C),
            config.W,
            0,
            h,
            weight_func,
            config,
            ax,
        )
    return fig2img(plt)


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
    previous_neighbor = None
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
                offset = (tri - delta) / 2

                p_color_node = p.color_node
                p_color_node = cast(ColorNode, p_color_node)
                neighbors = [p_color_node]
                if previous_neighbor:
                    neighbors.append(previous_neighbor)

                points = [p0, p1, (x + offset + delta, y), (x + offset, y)]
                c.set_color_node(ColorNode(c.node, neighbors))
                if isinstance(c.color_node, ColorNode):
                    ax.add_patch(
                        patches.Polygon(
                            points, fill=True, color=c.color_node.get_color()
                        )
                    )
                c.set_ll_width(x + offset, y, delta)
                x = x + tri + gamma_d
            p0, C2, n = p1, c.children, len(c.children)
            c.sticky = n == 0
            if c.sticky and c.span < config.lambda_:
                P2, g2 = P2 + [c], g2 + c.w
            if not c.sticky:
                P2, m2, A2, w2 = P2 + [c], m2 + n, A2 + weight_func(C2), w2 + tri
            previous_neighbor = c.color_node
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
