from multiprocessing import Process
import random
import pygraphviz as pgv
import os

import os
from PIL import Image

def process_file_chunk(lines, number):
    current_nodes = set()
    MAX_NODES = 40
    counter = 0
    MAX_IMAGES_PER_THREAD = 10000
    MAX_RANKSEP = 1
    MAX_NODESEP = .25
    G = pgv.AGraph(strict=False, directed=False)

    available_algos = ['dot', 'circo', 'fdp', 'sfdp', 'osage', 'patchwork']
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'black']
    styles = ["filled", "dashed", "dotted", "solid"]
    edge_styles = ["tapered", "dashed", "dotted", "bold"]
    next_size = MAX_NODES
    uniform = False
    directed = False
    ranksep = 0.5
    nodesep = 0.25

    # read file line by line
    for i, line in enumerate(lines):
        if line.startswith('#'):
            continue
        # if counter > MAX_IMAGES_PER_THREAD:
        #     break
        # try:
        parts = line.split()
        # add nodes to current_nodes
        current_nodes.add(int(parts[0]))
        current_nodes.add(int(parts[1]))
        G.add_edge(int(parts[0]), int(parts[1]))
        if len(G.nodes()) > next_size:
            # choose random layout
            G.layout(available_algos[random.randint(0, len(available_algos)-1)])
            chosen_color = colors[random.randint(0, len(colors)-1)]
            chosen_style = styles[random.randint(0, len(styles)-1)]
            if not uniform:
                for n in G.nodes_iter():
                    n.attr['color'] = colors[random.randint(0, len(colors)-1)]
                    n.attr['style'] = styles[random.randint(0, len(styles)-1)]
                    n.attr["label"] = ""
                for e in G.edges_iter():
                    e.attr["style"] = edge_styles[random.randint(0, len(edge_styles)-1)]
                    e.attr['color'] = colors[random.randint(0, len(colors)-1)]
            else:
                for n in G.nodes_iter():
                    n.attr["label"] = ""
                    n.attr["style"] = chosen_style
                    n.attr['color'] = chosen_color
                for e in G.edges_iter():
                    e.attr["style"] = chosen_style
                    e.attr['color'] = chosen_color
            path = f"data/images/{number}-{counter}.png"
            G.draw(path)
            image = Image.open(path)
            # image is too big too sensibly compress to 512x512, delete
            if image.size[0] > 5000 or image.size[1] > 5000:
                os.remove(path)
            else:
                image.thumbnail((224,224), Image.LANCZOS)
                image.save(path, "png") 
            counter += 1
            current_nodes = set()
            next_size = random.randint(5, MAX_NODES)
            uniform = random.randbytes(1)[0] % 2 == 0
            directed = random.randbytes(1)[0] % 2 == 0
            ranksep = random.random() * MAX_RANKSEP
            nodesep = random.random() * MAX_NODESEP
            G = pgv.AGraph(strict=False, directed=directed, ranksep=ranksep, nodesep=nodesep)
            del image
            print(f"{number}: processed {i} from {len(lines)} lines")
        # except Exception as e:
        #     print(e)
        #     continue

# Determine the total number of lines in the file
file = "data/com-lj.ungraph.txt"
total_lines = sum(1 for line in open(file))

# Determine the number of lines per chunk
num_processes = 8
lines_per_chunk = total_lines // num_processes

lines = []
with open(file, 'r') as f:
    lines = f.readlines()

# Create a thread for each chunk
processes = []
for i in range(num_processes):
    start_line = i * lines_per_chunk
    end_line = (i + 1) * lines_per_chunk - 1 if i < num_processes - 1 else total_lines - 1
    process = Process(target=process_file_chunk, args=(lines[start_line:end_line], len(processes)))
    process.start()
    processes.append(process)

# Wait for all processes to complete
for process in processes:
    process.join()