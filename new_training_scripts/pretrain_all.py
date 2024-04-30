from new_training_scripts.pretraining import main

datasets = [
    # "kblw/pretraining_samples_large",
    # "kblw/graphimages_twopi", ("kblw/graphviz_treemap","278679f"),
    # "kblw/graphviz_treemap"
    "kblw/treemap_sat"
]

if __name__ == "__main__":
    for entry in datasets:
        if isinstance(entry, tuple):
            main(entry[0], revision=entry[1])
        else:
            main(entry)
