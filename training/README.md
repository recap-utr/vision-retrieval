# Training
In this directory, our code for training (pre-training, fine-tuning) our vision models can be found.
While `finetuning.py` and `pretraining.py` contain the actual model and training code, the training is configured and launched from the corresponding `launch_finetuning.py` and `launch_pretraining.py` scripts. The training is monitored using `wandb`; the wandb project name can be specified inside `finetuning.py` and `pretraining.py`.

## Pre-Training
To launch the pre-training process, create a simple coordination script like this:
```python
from pretraining import main

datasets = ["path/to/pre-training/dataset"]

BATCH_SIZE = 32 
LATENT_DIM = 1536 
MODEL = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft" # base model
MAX_EPOCHS = 50

if __name__ == "__main__":
    for entry in datasets:
        main(entry, MODEL, LATENT_DIM, BATCH_SIZE, MAX_EPOCHS)

```
Inside the `datasets` list, `launch_pretraining` expects a list of [datasets](https://huggingface.co/docs/datasets/en/index) for which a seperate vision model should be trained. The `BATCH_SIZE` is GPU-dependent; 32 fits into the VRAM of an Nvidia Tesla V100. `MODEL` specifies the base vision [transformer](https://huggingface.co/docs/transformers/en/index) to be trained. `LATENT_DIM` is the latent dimension of the base model.

With our default configuration, the pre-training took about 30h on 6 Nvidia Tesla V100s per model.

## Fine-Tuning
To launch the fine-tuning process, create a simple coordination script like this:
```python
from finetuning import main

BATCH_SIZE = 16
LATENT_DIM = 1536
MODEL = "microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft" #base model
MAX_EPOCHS = 500

common_params = (MODEL, BATCH_SIZE, LATENT_DIM, MAX_EPOCHS)

datasets = [
    common_params
    + (
        "path/to/pre-trained/model",
        "path/to/fine-tuning/dataset",
    ),
]

if __name__ == "__main__":
    for entry in datasets:
        main(*entry)

```
`launch_finetuning` expects the same base configuration parameters as `launch_pretraining`; however, since our contrastive fine-tuning implementation uses two images per sample, the `BATCH_SIZE` should be half of that used in the pre-training. Additionally, as our fine-tuning datasets are much smaller than our pre-training datasets, epochs run much faster and the number of trained epochs can be increased because of that. 

With our default configuration, the pre-training took about 5h on 6 Nvidia Tesla V100s per model.

**Note**: Instead of a simple string for the training dataset, `launch_finetuning` expects the entries of the `datasets` list to be tuples `(<base model>, <batch size>, <latent dimension>, <max epochs>, <pre-trained model>, <fine-tuning dataset>)`.
