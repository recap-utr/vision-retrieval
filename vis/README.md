# Visualizations
In this directory, the code to **visualize argumentation graphs** using our visualizations, and to create the **training datasets** can be found.

- `logical.py`, `srip.py` and `treemaps.py` implement the actual graph drawing.
- `random_graphs.py` contains the code to generate synthetical graphs of the specified visualization. **Note**: For unbiased training, the resulting images should be de-duplicated using a tool like [fclones](https://github.com/pkolaczk/fclones).
- `resize.py` is a utility to re-size every image inside a folder to the same dimensions.
- `treemap_dataset.py` and `logical_srip_datasets.py` contain code to batch generate visualization for a folder of argumentation graphs. This is used to generate the fine-tuning datasets, as well as the casebase and requests for the evaluation. The argumentation graphs have to be in a [arguebuf compatible](https://arguebuf.readthedocs.io/en/latest/arguebuf/load.html) format.