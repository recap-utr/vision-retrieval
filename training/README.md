# Training
In this directory, our code for training (pre-training, fine-tuning) our vision models can be found.
While `finetuning.py` and `pretraining.py` contain the actual model and training code, the training is configured and launched from the corresponding `launch_training.py` which provides a typer CLI to start the pre-training and fine-tuning processes.

## Parameters
To train a model (be this pre-training or fine-tuning) you need to provide a [dataset](https://huggingface.co/docs/datasets/en/index). The dataset is expected to have a `train` split and can optionally have a `test` split (if none is specified, 10% of the `train` split are used). 

Both training processes use early-stopping mechanisms, because of which only the maximum number of epochs and not the exact number of epochs can be specified.

Regarding **batch size**: On our configuration using 6 Nvidia Tesla V100s, we were able to set this to 32 for pre-training and 16 for fine-tuning, which maxes out the VRAM. Especially for fine-tuning a larger batch size is likely to be advantageous, because this provides more contrastive negatives with which an image is compared in a batch.

**Training duration**: With our default configuration, the pre-training took about 30h for 25 epochs on 6 Nvidia Tesla V100s per model. Fine-tuning took about 5h for 500 epochs per model.
