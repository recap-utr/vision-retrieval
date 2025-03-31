import matplotlib.pyplot as plt
import json
import csv

"""
This script generates two plots for the paper's scaling study comparing the scaling
 of the A* approach to the scaling of the vision-based approach when increasing the number of s-nodes in an argument graph.
 It expects the following files to be present:
1. ../data/retrieval_scaling/gpu_results.json
2. ../data/retrieval_scaling/astar_results.csv
They should be generated using the eval_cli / A* search.
"""

data = json.load(open("../data/retrieval_scaling/gpu_results.json"))
visualization_times = data["visualization_times"]
embedding_times = data["embedding_times"]
similarity_computation_times = data["similarity_computation_times"]
with open("../data/retrieval_scaling/astar_results.csv", "r") as file:
    reader = csv.DictReader(file)
    astar_times = {row["name"].split(".")[0]: row["duration"] for row in reader}


# Convert string keys to integers for proper sorting
vis_times = {int(k): v for k, v in visualization_times.items()}
emb_times = {int(k): v for k, v in embedding_times.items()}
sim_times = {int(k): v for k, v in similarity_computation_times.items()}
# sim2_times = {int(k): v for k, v in sim2.items()}
total = {k: vis_times[k] + emb_times[k] + sim_times[k] for k in vis_times.keys()}

# Sort the data points by number of S-nodes
x_values = sorted(vis_times.keys())
vis_y = [vis_times[x] for x in x_values]
emb_y = [emb_times[x] for x in x_values]
sim_y = [sim_times[x] for x in x_values]
# sim2_y = [sim2_times[x] for x in x_values]
total_y = [total[x] for x in x_values]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot each line with different styles and colors
plt.plot(x_values, vis_y, "b", label="Visualization Time")
plt.plot(x_values, emb_y, "r", label="Embedding Time")
plt.plot(x_values, sim_y, "g", label="Similarity Computation Time")
# plt.plot(x_values, sim2_y, "grey", label="Similarity Computation Time 2")
plt.plot(x_values, total_y, "orange", label="Total Processing Time")

# Customize the plot
plt.xlabel("Number of S-nodes")
plt.ylabel("Time (seconds)")
plt.title("Performance Scaling with Number of S-nodes")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Use logarithmic scale if the values span multiple orders of magnitude
# plt.yscale('log')  # Uncomment if needed

# Add minor gridlines
plt.grid(True, which="minor", linestyle=":", alpha=0.4)

# Rotate x-axis labels if they overlap
# plt.xticks(x_values, rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig("../data/scaling_analysis.png", dpi=300, bbox_inches="tight")
plt.close()

# Filter the data points for the keys in astar_times (4-20)
astar_keys = range(4, 21)
filtered_total_y = [total[k] for k in astar_keys if k in total]
filtered_astar_y = [
    float(astar_times[str(k)]) for k in astar_keys if str(k) in astar_times
]

# Create the second plot
plt.figure(figsize=(10, 6))

# Plot each line with different styles and colors
plt.plot(astar_keys, filtered_total_y, "orange", label="Total Processing Time")
plt.plot(astar_keys, filtered_astar_y, "purple", label="A* Search Time")

# Customize the plot
plt.xlabel("Number of S-nodes")
plt.ylabel("Time (seconds)")
plt.title("Total Vision-Based Processing Time vs A* Search Time")
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend()

# Use logarithmic scale if the values span multiple orders of magnitude
# plt.yscale('log')  # Uncomment if needed

# Add minor gridlines
plt.grid(True, which="minor", linestyle=":", alpha=0.4)

# Rotate x-axis labels if they overlap
# plt.xticks(astar_keys, rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig("../data/total_vs_astar_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
