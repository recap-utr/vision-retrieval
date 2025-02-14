import json
import os
from statistics import mean
from collections import defaultdict
from glob import glob

MODE = "simple"


def process_files(folder_path):
    # Store all values in a nested dictionary
    all_values = defaultdict(lambda: defaultdict(list))

    # Process each JSON file
    for filename in glob(os.path.join(folder_path, f"results_{MODE}_*.json")):
        with open(os.path.join(folder_path, filename), "r") as f:
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

    return result


def create_latex_table(data):
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


def main():
    # Replace with your folder path
    folder_path = "../data/eval_all/results/torch_models"

    # Process all files
    results = process_files(folder_path)

    # Create and print LaTeX table
    latex_table = create_latex_table(results)
    print(latex_table)


if __name__ == "__main__":
    main()
