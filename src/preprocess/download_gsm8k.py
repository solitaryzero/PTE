from datasets import load_dataset

ds = load_dataset("openai/gsm8k", 'main')
ds.save_to_disk('./data/gsm8k')