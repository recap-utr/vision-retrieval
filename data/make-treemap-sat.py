from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="pt-source/images", num_proc=6)
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
dataset.push_to_hub("kblw/treemap_sat")