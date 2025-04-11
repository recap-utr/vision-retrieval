import matplotlib.pyplot as plt
import matplotlib.patches as patches
import arguebuf as ab
from PIL import Image
from util import fig2img, ColorNode


class Node:
    children: list["Node"] = []
    neighbors: list[ColorNode] = []
    node: ab.AbstractNode
    color: ColorNode
    parent: "Node | None"
    is_dummy: bool

    def __init__(
        self,
        node: ab.AbstractNode,
        parent: "Node | None" = None,
        grandparent: "Node | None" = None,
        is_dummy: bool = False,
    ) -> None:
        self.node = node
        self.parent = parent
        self.neighbors = [parent.color] if (parent and not parent.is_dummy) else []
        if grandparent and not grandparent.is_dummy:
            self.neighbors.append(grandparent.color)
        self.color = ColorNode(node, self.neighbors)
        self.is_dummy = is_dummy

    def set_children(self, children: list["Node"]):
        self.children = children

    def __str__(self):
        s = f"[dummy: {self.is_dummy} {self.color.get_color()}]. Children: {len(self.children)}"
        return s


class Rectangle:
    def __init__(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        color: tuple[float, float, float],
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color

    def __str__(self) -> str:
        return f"({self.x}, {self.y}) - ({self.width}, {self.height}): {self.color}"


def get_treemap_rects(
    tree_root: Node, x: float, y: float, width: float, height: float, horizontal: bool
) -> list[Rectangle]:
    parts = len(tree_root.children)
    # no children
    if parts == 0 and not tree_root.is_dummy:
        return [Rectangle(x, y, width, height, tree_root.color.get_color())]
    # is scheme node
    if parts == 0 and tree_root.is_dummy:
        return []

    next_horizontal = horizontal
    if not tree_root.is_dummy:
        own_width = width if horizontal else 0.1 * width
        own_height = 0.1 * height if horizontal else height
        children_width = width / parts if horizontal else 0.9 * width
        children_height = height * 0.9 if horizontal else height / parts
        res = [Rectangle(x, y, own_width, own_height, tree_root.color.get_color())]
        if horizontal:
            y += own_height
        else:
            x += own_width
        next_horizontal = not horizontal
    # is not scheme node: node itself will not be drawn
    else:
        res = []
        children_width = width / parts if horizontal else width
        children_height = height / parts if not horizontal else height
    for i, child in enumerate(tree_root.children):
        if horizontal:
            child_x = x + i * children_width
            child_y = y
        else:
            child_x = x
            child_y = y + i * children_height
        res += get_treemap_rects(
            child,
            child_x,
            child_y,
            children_width,
            children_height,
            next_horizontal,
        )
    return res


def _get_children(graph: ab.Graph, node: ab.AbstractNode):
    res = [
        child
        for child in graph.incoming_nodes(node)
        if isinstance(child, ab.SchemeNode)
    ]
    if isinstance(node, ab.AtomNode):
        return res
    for child in graph.incoming_atom_nodes(node):
        res += [c for c in graph.incoming_nodes(child) if isinstance(c, ab.SchemeNode)]
    return res


def build_tree(
    graph: ab.Graph,
    root: ab.AbstractNode,
    parent: Node | None = None,
    grandparent: Node | None = None,
) -> Node:
    node = Node(
        root, parent, grandparent, is_dummy=(not isinstance(root, ab.SchemeNode))
    )
    if len(_get_children(graph, root)) > 0:
        abstract_children = _get_children(graph, root)
        children = [build_tree(graph, abstract_children[0], node, parent)]
        previous_child = children[-1]
        for child in abstract_children[1:]:
            children.append(build_tree(graph, child, node, previous_child))
            previous_child = children[-1]
        node.set_children(children)
    return node


def visualize_treemap(
    graphpath: str,
    root: ab.AbstractNode,
    height: float = 10,
    width: float = 10,
    dpi: int = 100,
) -> Image.Image:
    graph = ab.load.file(graphpath)
    return visualize_treemap_inmem(graph, root, height, width, dpi=dpi)


def visualize_treemap_inmem(
    graph: ab.Graph,
    root: ab.AbstractNode,
    height: float = 10,
    width: float = 10,
    dpi: int = 100,
) -> Image.Image:
    if root is None and graph.root_node is None:
        raise ValueError("Root node is ambiguous. Please provide a root node.")
    tree = build_tree(graph, root)

    # get treemap rects
    rects = get_treemap_rects(tree, 0, 0, width, height, False)

    # create png
    return fig2img(draw_treemap(rects, height, width, dpi=dpi))


def draw_treemap(rects: list[Rectangle], height, width, dpi: int):
    fig = plt.figure(figsize=(width, height), dpi=dpi)

    # Create axes that fill the entire figure
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    # ax.set_aspect("equal", adjustable="box")
    ax.axis("off")  # Remove axis labeling
    ax.margins(0)  # Remove padding
    for rect in rects:
        ax.add_patch(
            patches.Rectangle(
                (rect.x, rect.y),
                rect.width,
                rect.height,
                fill=True,
                facecolor=rect.color,
            )
        )

    return plt
