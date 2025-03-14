import arguebuf as ab
from PIL import Image


def normalize(graph: ab.Graph) -> None:
    s_nodes = [n for n in graph.scheme_nodes.values()]
    for n in s_nodes:
        # scheme node refers to another scheme node instead of atom node
        incoming = graph.incoming_nodes(n)
        incoming_s_nodes = [
            node for node in incoming if isinstance(node, ab.SchemeNode)
        ]
        incoming_i_nodes = [node for node in incoming if isinstance(node, ab.AtomNode)]
        if len(incoming_s_nodes) > 0:
            # find the s_node's edge to n
            s_node = incoming_s_nodes[0]
            edge = [e for e in graph.outgoing_edges(s_node)][0]
            # this edge should lead to the i-node beneath n
            graph.remove_edge(edge)
            graph.add_edge(ab.Edge(s_node, incoming_i_nodes[0]))


def find_heighest_root_node(graph: ab.Graph) -> ab.AbstractNode:
    layers = [layerize(graph, root) for root in graph.root_nodes]
    max_depth = max([len(layer) for layer in layers])
    max_depth_index = [i for i, layer in enumerate(layers) if len(layer) == max_depth]
    return list(graph.root_nodes)[max_depth_index[0]]


def layerize(
    graph: ab.Graph, major_claim: ab.AbstractNode
) -> list[list[ab.AbstractNode]]:
    layers = [[major_claim]]
    children_from_this_layer = True
    current_depth = 1
    while children_from_this_layer:
        children_from_this_layer = False
        new_layer = []
        for node in layers[current_depth - 1]:
            for child in graph.incoming_nodes(node):
                new_layer.append(child)
                children_from_this_layer = True
        if children_from_this_layer:
            layers.append(new_layer)
        current_depth += 1
    return layers


def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    fig.close()
    return img