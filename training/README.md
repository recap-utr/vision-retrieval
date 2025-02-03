# Training
In this directory, our code for training (pre-training, fine-tuning) our vision models can be found.
While `finetuning.py` and `pretraining.py` contain the actual model and training code, the training is configured and launched from the corresponding `launch_finetuning.py` and `launch_pretraining.py` scripts. The training is monitored using `wandb`; the wandb project name can be specified inside `finetuning.py` and `pretraining.py`.

## Pre-Training
Inside the `datasets` list, `launch_pretraining` expects a list of [datasets](https://huggingface.co/docs/datasets/en/index) for which a seperate vision model should be trained. The `BATCH_SIZE` is GPU-dependent; 32 fits into the VRAM of an Nvidia Tesla V100. `MODEL` specifies the base vision [transformer](https://huggingface.co/docs/transformers/en/index) to be trained. `LATENT_DIM` is the latent dimension of the base model.

With our default configuration, the pre-training took about 30h on 6 Nvidia Tesla V100s per model.

## Fine-Tuning
`launch_finetuning` expects the same base configuration parameters as `launch_pretraining`; however, since our contrastive fine-tuning implementation uses two images per sample, the `BATCH_SIZE` should be half of that used in the pre-training. Additionally, as our fine-tuning datasets are much smaller than our pre-training datasets, epochs run much faster and the number of trained epochs can be increased because of that. 

With our default configuration, the pre-training took about 5h on 6 Nvidia Tesla V100s per model.

**Note**: Instead of a simple string for the training dataset, `launch_finetuning` expects the entries of the `datasets` list to be tuples `(<base model>, <batch size>, <latent dimension>, <max epochs>, <pre-trained model>, <fine-tuning dataset>)`.
