import matplotlib.pyplot as plt
import numpy as np
import json

BASEPATH = "../data/retrieval_scaling"
CPU_FILE = f"{BASEPATH}/cpu_results.json"
GPU_FILE = f"{BASEPATH}/gpu_results.json"


def compare_timing_files(data1, data2):
    """
    Compare timing data between two files.
    """

    # Verify both files have the same categories
    categories = set(data1.keys()) & set(data2.keys())
    if not categories:
        raise ValueError("No matching categories found between files")

    # Create plots
    n_categories = len(categories)
    fig, axs = plt.subplots(n_categories, 2, figsize=(15, 5 * n_categories))

    for idx, category in enumerate(sorted(categories)):
        dict1 = data1[category]
        dict2 = data2[category]

        # Calculate differences
        keys = set(dict1.keys()) & set(dict2.keys())
        differences = [float(dict1[k]) - float(dict2[k]) for k in keys]
        avg_diff = np.mean(differences)
        std_diff = np.std(differences)

        # Print statistics
        print(f"\nCategory: {category}")
        print(f"Average difference (file1 - file2): {avg_diff:.6f} seconds")
        print(f"Standard deviation of differences: {std_diff:.6f} seconds")

        # Plot 1: Scatter plot of values
        axs[idx, 0].scatter(
            list(keys), [dict1[k] for k in keys], label="File 1", alpha=0.5
        )
        axs[idx, 0].scatter(
            list(keys), [dict2[k] for k in keys], label="File 2", alpha=0.5
        )
        axs[idx, 0].set_title(f"{category} - Values Comparison")
        axs[idx, 0].set_xlabel("Key")
        axs[idx, 0].set_ylabel("Time (seconds)")
        axs[idx, 0].legend()
        axs[idx, 0].tick_params(axis="x", rotation=45)

        # Plot 2: Histogram of differences
        axs[idx, 1].hist(differences, bins=30)
        axs[idx, 1].axvline(
            avg_diff, color="r", linestyle="dashed", label=f"Mean diff: {avg_diff:.6f}"
        )
        axs[idx, 1].set_title(f"{category} - Differences Distribution")
        axs[idx, 1].set_xlabel("Difference (seconds)")
        axs[idx, 1].set_ylabel("Count")
        axs[idx, 1].legend()

    plt.tight_layout()
    plt.savefig("comparison.png")
    plt.close()


data1 = json.load(open(CPU_FILE))
data2 = json.load(open(GPU_FILE))
compare_timing_files(data1, data2)
