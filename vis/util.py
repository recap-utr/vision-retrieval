import arguebuf as ab


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


def find_major_claim(graph: ab.Graph) -> ab.AbstractNode:
    if graph.major_claim:
        return graph.major_claim
    mj = [n for n in graph.atom_nodes.values() if len(graph.outgoing_nodes(n)) == 0]
    return mj[0]


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
