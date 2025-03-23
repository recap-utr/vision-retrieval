# Visualizations
In this directory, the code to **visualize argumentation graphs** using our visualizations, and to create the **training datasets** can be found.

- `logical.py`, `srip.py` and `treemaps.py` implement the actual graph drawing.
- These are wrapped in a typer-CLI accessible in `render.py` to generate a visualization based on a argumentation graph file.
- `random_graphs.py` provides a CLI to generate synthetical graphs of the specified visualization. **Note**: For unbiased training, the resulting images should be de-duplicated using a tool like [fclones](https://github.com/pkolaczk/fclones).
- `resize.py` is a utility to re-size every image inside a folder to the same dimensions.
- `generate_finetuning_datasets.py` provides a CLI to batch create visualizations for entire datasets. This is useful to generate finetuning and evaluation samples. The argumentation graphs have to be in a [arguebuf compatible](https://arguebuf.readthedocs.io/en/latest/arguebuf/load.html) format.