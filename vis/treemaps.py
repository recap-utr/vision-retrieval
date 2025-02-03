import matplotlib.pyplot as plt
import matplotlib.patches as patches
import arguebuf as ab
from PIL import Image
from pathlib import Path

colors = {"Attack": "#e6194B", "Support": "#3cb44b"}


def get_treemap_rects(tree, x, y, width, height, horizontal, depth=0) -> list:
    parts = len(tree["children"])
    if parts == 0:
        return (
            [
                {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "depth": depth,
                    "label": tree["label"],
                }
            ]
            if "label" in tree
            else []
        )
    if "label" in tree:
        own_width = width if horizontal else 0.1 * width
        own_height = 0.1 * height if horizontal else height
        children_width = width / parts if horizontal else 0.9 * width
        children_height = height * 0.9 if horizontal else height / parts
        res = [
            {
                "x": x,
                "y": y,
                "width": own_width,
                "height": own_height,
                "depth": depth,
                "label": tree["label"],
            }
        ]
        if horizontal:
            y += own_height
        else:
            x += own_width
    else:
        children_width = width / parts if horizontal else width
        children_height = height / parts if not horizontal else height
        res = []
    for i, child in enumerate(tree["children"]):
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
            not horizontal,
            depth + 1,
        )
    return res


def _get_children(graph: ab.Graph, node: ab.AtomNode):
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


def build_tree(graph: ab.Graph, root: ab.AtomNode):
    if not isinstance(root, ab.SchemeNode):
        return {
            "children": [
                build_tree(graph, child) for child in _get_children(graph, root)
            ]
        }
    if len(graph.incoming_nodes(root)) == 0:
        return {"id": root.id, "label": root.label, "children": []}
    return {
        "id": root.id,
        "label": root.label,
        "children": [build_tree(graph, child) for child in _get_children(graph, root)],
    }


def draw_treemap(rects, height, width, savepath: str | Path):
    _, ax = plt.subplots(1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")  # Remove axis labeling
    ax.margins(0)  # Remove padding
    for rect in rects:
        if "label" not in rect and rect["label"] != None:
            continue
        ax.add_patch(
            patches.Rectangle(
                (rect["x"], rect["y"]),
                rect["width"],
                rect["height"],
                fill=True,
                facecolor=colors.get(rect["label"], "white"),
                linewidth=1,
                edgecolor="black",
            )
        )

    plt.savefig(savepath)
    plt.close()


def visualize_treemap(
    graphpath: str, savepath: str | Path, height: int = 256, width: int = 256
):
    graph = ab.load.file(graphpath)
    visualize_treemap_inmem(graph, savepath, height, width)


def visualize_treemap_inmem(
    graph: ab.Graph, savepath: str | Path, height: int = 256, width: int = 256
):
    # find source nodes (i.e. nodes without incoming edges)
    source_nodes = [
        node for node in graph.nodes.values() if len(graph.outgoing_edges(node)) == 0
    ]

    # build tree
    tree = (
        build_tree(graph, graph.major_claim)
        if graph.major_claim
        else build_tree(graph, source_nodes[0])
    )

    # get treemap rects
    rects = get_treemap_rects(tree, 0, 0, width, height, True)

    # create png
    draw_treemap(rects, height, width, savepath)


def multi_treemaps(graph: ab.Graph, savepath: str, height: int = 256, width: int = 256):
    source_nodes = [
        node for node in graph.nodes.values() if len(graph.outgoing_edges(node)) == 0
    ]

    # build tree
    trees = [build_tree(graph, sn) for sn in source_nodes]

    # get treemap rects
    rects = [get_treemap_rects(tree, 0, 0, width, height, True) for tree in trees]

    # create png
    [
        draw_treemap(rects, height, width, f"{savepath}-{idx}.png")
        for idx, rects in enumerate(rects)
    ]
    [standard_resize(f"{savepath}-{idx}.png") for idx, rects in enumerate(rects)]


def standard_resize(file: str | Path):
    img = Image.open(file)
    img = img.crop(
        (
            img.width / 2 - 177,
            img.height / 2 - 182,
            img.width / 2 + 192,
            img.height / 2 + 177,
        )
    ).resize((256, 256))
    img.save(file)
