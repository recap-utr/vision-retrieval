from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="../data/random_logical_srip/srip2")

ds.push_to_hub("kblw/srip_pt_extended")
