from pretraining import main

datasets = [
    # "kblw/pretraining_samples_large",
    # "kblw/graphimages_twopi", ("kblw/graphviz_treemap","278679f"),
    # "kblw/graphviz_treemap"
    "kblw/srip_pt_extended",
    "kblw/logical_pt_extended"
]

BATCH_SIZE = 32
LATENT_DIM = 1536
MODEL = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
MAX_EPOCHS = 50

if __name__ == "__main__":
    for entry in datasets:
        main(entry, MODEL, LATENT_DIM, BATCH_SIZE, MAX_EPOCHS)
