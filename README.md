# Retrieving Argument Graphs Using Vision Transformers
This is the repository for our paper "Retrieving Argument Graphs Using Vision Transformers".
It contains our code
- to visualize argumentation graphs using our visualizations: Treemaps, Logical and Space-Reclaiming Icicle Plots (`vis` directory),
- for pre-training and fine-tuning our vision transformers (`training` directory),
- for generating a training dataset to fine-tune a GPT-4o model (`oai_finetuning` directory),
- to evaluate the various models (`eval` directory).

A more detailed explanation of our scripts can be found in the README.md within those directories.


## Data
A corpus of publicly available argumentation graphs is available on [GitHub](https://github.com/recap-utr/arguebase-public). For our evaluation, we used the `microtexts` dataset, together with the requests found [here](https://github.com/recap-utr/arguelauncher).


## Installation
The project's dependencies are managed by [uv](https://docs.astral.sh/uv/). After [installing uv](https://docs.astral.sh/uv/#installation), create a virtual environment for the project using `uv sync`. After that, the various scripts can be executed using `uv run python <script>`.
