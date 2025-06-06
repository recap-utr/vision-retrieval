import json
import os
from statistics import mean
from collections import defaultdict
from glob import glob
from pathlib import Path
import typer
from eval_cli import Mode
from typing import Annotated

app = typer.Typer()

@app.command()
def process_files(input_folder: Annotated[Path, typer.Argument(help="This folder should contain one JSON eval file per model. The files should be named results_MODE_MODEL.json.")], mode: Mode):
    # Store all values in a nested dictionary
    all_values = defaultdict(lambda: defaultdict(list))

    # Process each JSON file
    for filename in glob(os.path.join(input_folder, f"results_{mode}_*.json")):
        with open(os.path.join(input_folder, filename), "r") as f:
            data = json.load(f)

            # For each model type (logical_ft_arg, logical_pt, etc.)
            for model, metrics in data.items():
                # For each metric in the model
                for metric, value in metrics.items():
                    all_values[model][metric].append(value)

    # Process the collected values
    result = {}
    for model in all_values:
        result[model] = {}
        for metric, values in all_values[model].items():
            if metric in ["duration", "embedding_time"]:
                # Calculate average for duration and embedding_time
                result[model][metric] = mean(values)
            else:
                # Check if all values are the same for other metrics
                if len(set(values)) > 1:
                    print(f"Warning: Different values found for {model} - {metric}")
                result[model][metric] = values[0]  # Take the first value

    print(create_latex_table(result))


def create_latex_table(data: dict[str, dict[str, float]]):
    metrics = [
        "ndcg",
        "map",
        "recall",
        "correctness",
        "completeness",
        "duration",
        "embedding_time",
    ]

    # Start the LaTeX table
    latex = (
        "\\begin{table}[h]\n\\centering\n\\begin{tabular}{l"
        + "r" * len(metrics)
        + "}\n"
    )
    latex += "\\toprule Model & {\\scshape Ndcg} & {\\scshape Map} & {\\scshape Recall} & {\\scshape Cor} & {\\scshape Com} & {\\scshape Dur} & {\\scshape Emb}\\midrule\n"

    # Add data rows
    for model in data:
        row = [model.replace("_", "\\_")]
        for metric in metrics:
            value = data[model][metric]
            # Round to 2 decimal places
            row.append(f"{value:.2f}")
        latex += " & ".join(row) + " \\\\\n"

    # Close the table
    latex += "\\bottomrule\n\\end{tabular}\n\\caption{Model Metrics}\n\\label{tab:metrics}\n\\end{table}"

    return latex


if __name__ == "__main__":
    app()
