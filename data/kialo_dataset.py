import subprocess
from pybars import Compiler
from glob import glob
import arguebuf as ab

compiler = Compiler()

def _list(_, options, items):
    result = []
    for item in items:
        result.append(options['fn'](item))
        result.append('\n')
    return result

def get_color(node_label):
    if node_label == "Support":
        return "green"
    elif node_label == "Attack":
        return "red"
    return "blue"


def export_graph(inp: ab.Graph, file: str, return_str = False):
    source = """
        digraph "" {
        nodesep=0.02
        layersep=0.02
        ranksep=0.02
        node [height=0.2,
            label="",
            style=filled,
            width=0.2,
            shape=ellipse,
            penwidth=0,
            color=blue
        ];
        sep=-10
        edge [arrowhead=none,
            style=tapered
        ];
        {{#list nodes}}"{{id}}" [color="{{color}}"] {{/list}}
        {{#list edges}}"{{source}}" -> "{{target}}" {{/list}}
    }
    """
    template = compiler.compile(source)
    helpers = {
        'list': _list,
    }
    output = template({'nodes': [{"id": node.id, "color": get_color(node.label)} for node in inp.nodes.values()], 'edges': [{"source": edge.source.id, "target": edge.target.id} for edge in inp.edges.values()]}, helpers=helpers)
    if return_str:
        return output
    subprocess.run(["dot", "-Tpng", "-o", file], input=output.encode())

files = glob("kialo-graphnli/*.json")
OUTPUT = "images/"

for file in files:
    inp = ab.load.file(file)
    export_graph(inp, OUTPUT + file.split("/")[-1].replace(".json", ".png"))