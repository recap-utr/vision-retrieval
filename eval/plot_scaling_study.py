import matplotlib.pyplot as plt
import json
import csv
import typer
from pathlib import Path

"""
This script generates two plots for the paper's scaling study comparing the scaling
 of the A* approach to the scaling of the vision-based approach when increasing the number of s-nodes in an argument graph.
 It expects the following files to be present:
1. ../data/retrieval_scaling/gpu_results.json
2. ../data/retrieval_scaling/astar_results.csv
They should be generated using the eval_cli / A* search.
"""

app = typer.Typer()

@app.command()
def plot_scaling_study(torch_times__file: Path, astar_times_file: Path, output_folder: Path):
    data = json.load(open(torch_times__file, "r"))
    visualization_times = data["visualization_times"]
    embedding_times = data["embedding_times"]
    similarity_computation_times = data["similarity_computation_times"]
    with open(astar_times_file, "r") as file:
        reader = csv.DictReader(file)
        astar_times = {row["name"].split(".")[0]: row["duration"] for row in reader}

    # Convert string keys to integers for proper sorting
    vis_times = {int(k): v for k, v in visualization_times.items()}
    emb_times = {int(k): v for k, v in embedding_times.items()}
    sim_times = {int(k): v for k, v in similarity_computation_times.items()}
    total = {k: vis_times[k] + emb_times[k] + sim_times[k] for k in vis_times.keys()}

    # Sort the data points by number of S-nodes
    x_values = sorted(vis_times.keys())
    vis_y = [vis_times[x] for x in x_values]
    emb_y = [emb_times[x] for x in x_values]
    sim_y = [sim_times[x] for x in x_values]
    total_y = [total[x] for x in x_values]

    # Create the plot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot vision-based processing times on the primary y-axis
    ax1.plot(x_values, vis_y, "b-", label="Visualization Time (ViT)")
    ax1.plot(x_values, emb_y, "r-", label="Embedding Time (ViT)")
    ax1.plot(x_values, sim_y, "g-", label="Similarity Computation Time (ViT)")
    ax1.plot(x_values, total_y, "orange", linewidth=2, label="Total Processing Time (ViT)")

    # Set up the primary y-axis
    ax1.set_xlabel("Number of S-nodes")
    ax1.set_ylabel("Vision Processing Time (seconds)", color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Create a secondary y-axis for A* search times
    ax2 = ax1.twinx()

    # Filter and plot A* search times on the secondary y-axis
    astar_x = [int(k) for k in astar_times.keys() if k.isdigit()]
    astar_y = [float(astar_times[str(x)]) for x in astar_x if str(x) in astar_times]
    ax2.plot(astar_x, astar_y, "purple", linewidth=2, linestyle='-', label="Total Processing Time (A*)")
    ax2.set_ylabel("A* Search Time (seconds)", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Title and layout adjustments
    plt.title("Performance Scaling with Number of S-nodes")

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')

    # Add minor gridlines
    ax1.grid(True, which="minor", linestyle=":", alpha=0.4)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_folder / "combined_scaling_study.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    app()