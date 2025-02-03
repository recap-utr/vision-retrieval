from finetuning import main

BATCH_SIZE = 16
LATENT_DIM = 1536
MODEL = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft"
MAX_EPOCHS = 500

datasets = [
    (MODEL, BATCH_SIZE, LATENT_DIM, MAX_EPOCHS, "VisionRetrievalPretraining/logical_pt_extended/logical_best", "kblw/logical_ft_arg"), 
    (MODEL, BATCH_SIZE, LATENT_DIM, MAX_EPOCHS, "VisionRetrievalPretraining/srip_pt_extended/srip_best", "kblw/srip_ft_arg"), 
            ]

if __name__ == "__main__":
    for entry in datasets:
        main(*entry)