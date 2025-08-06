from datasets import load_dataset

ds = load_dataset("zen-E/GSM8k-Aug")
ds.save_to_disk('./data/gsm8k_aug')