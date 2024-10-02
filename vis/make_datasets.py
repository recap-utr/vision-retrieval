from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="../data/arg_finetune_treemaps")

ds.push_to_hub("kblw/treemaps_ft_arg")
