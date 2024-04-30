import arguebuf as ab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# 9-class blues from https://colorbrewer2.org/#type=sequential&scheme=Blues&n=9
colors = [
    "#f7fbff",
    "#deebf7",
    "#c6dbef",
    "#9ecae1",
    "#6baed6",
    "#4292c6",
    "#2171b5",
    "#08519c",
    "#08306b"
]
colors = [
    "#e6194B",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c"
]
def get_depth_color(depth: int):
    return colors[depth % len(colors)]

def build_tree(graph: ab.Graph, root: ab.AtomNode):
    if len(graph.incoming_atom_nodes(root)) == 0:
        return {"id": root.id, "label": root.label, "children": []}
    return {"id": root.id, "label": root.label, "children": [build_tree(graph, child) for child in graph.incoming_atom_nodes(root)]}

def get_treemap_rects(tree, x, y, width, height, horizontal, depth=0) -> list:
    parts = len(tree["children"])
    if parts == 0:
        return []
    children_width = width // parts if horizontal else width
    children_height = height // parts if not horizontal else height
    res = []
    for i, child in enumerate(tree["children"]):
        if horizontal:
            child_x = x  + i * children_width
            child_y = y
        else:
            child_x = x
            child_y = y + i * children_height
        res.append({"x": child_x, "y": child_y, "width": children_width, "height": children_height, "depth": depth})
        res += get_treemap_rects(child, child_x, child_y, children_width, children_height, not horizontal, depth+1)
    return res
        
def draw_treemap(rects, height, width, savepath):
    _, ax = plt.subplots(1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Remove axis labeling
    ax.margins(0)  # Remove padding
    for rect in rects:
        ax.add_patch(patches.Rectangle((rect["x"], rect["y"]), rect["width"], rect["height"], fill=True, facecolor=get_depth_color(rect["depth"]), linewidth=1, edgecolor="black"))

    plt.savefig(savepath)
    plt.close()

def visualize_treemap(graphpath: str, savepath: str, height: int = 256, width: int = 256):
    graph = ab.load.file(graphpath)
    visualize_treemap_inmem(graph, savepath, height, width)

def visualize_treemap_inmem(graph: ab.Graph, savepath: str, height: int = 256, width: int = 256):
    # find source nodes (i.e. nodes without incoming edges)
    source_nodes = [node for node in graph.nodes.values() if len(graph.outgoing_edges(node)) == 0]
    
    # build tree
    tree = build_tree(graph, graph.major_claim) if graph.major_claim else build_tree(graph, source_nodes[0])

    # get treemap rects
    rects = get_treemap_rects(tree, 0, 0, width, height, True)

    # create png
    draw_treemap(rects, height, width, savepath)

def multi_treemaps(graph: ab.Graph, savepath: str, height: int = 256, width: int = 256):
    source_nodes = [node for node in graph.nodes.values() if len(graph.outgoing_edges(node)) == 0]
    
    # build tree
    trees = [build_tree(graph, sn) for sn in source_nodes]

    # get treemap rects
    rects = [get_treemap_rects(tree, 0, 0, width, height, True) for tree in trees]

    # create png
    [draw_treemap(rects, height, width, f"{savepath}-{idx}.png") for idx, rects in enumerate(rects)]
    [standard_resize(f"{savepath}-{idx}.png") for idx, rects in enumerate(rects)]

def standard_resize(file: str):
    img = Image.open(file)
    img = img.crop((img.width/2-177, img.height/2-182, img.width/2+192, img.height/2+187)).resize((256, 256))
    img.save(file)

