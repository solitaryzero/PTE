import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
    
class LSVCollator:
    def __init__(self):
        pass
    
    def __call__(self, elements):
        batch = {
            'question': [e['query'] for e in elements],
            'step_results': [e['step_results'] for e in elements],
            'answer': [e['answer'] for e in elements],
        }

        return batch
    

class ProbeCollator:
    def __init__(self, pad_token_id, latent_encoder, tokenizer=None):
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.latent_encoder = latent_encoder

    def __call__(self, elements):
        tokenlist = [e["input_ids"] for e in elements]
        probe_mask_list = [e['probe_mask'] for e in elements]
        step_results = [e['step_results'] for e in elements]
        tokens_maxlen = max([len(t) for t in tokenlist])

        input_ids, probe_masks, labels, attention_masks = [], [], [], []
        for tokens, probe_mask, s_results in zip(tokenlist, probe_mask_list, step_results):
            pad_len = tokens_maxlen - len(tokens)
            input_ids.append(tokens + [self.pad_token_id]*pad_len)
            attention_masks.append([1]*len(tokens) + [0]*pad_len)

            try:
                assert len(s_results) == sum(probe_mask)
            except:
                print(self.tokenizer.convert_ids_to_tokens(tokens))
                print(s_results)
                print(probe_mask)
                quit()

            probe_masks.append(probe_mask + [0]*pad_len)
            count = 0
            label = [self.latent_encoder.get_dummy_latent() for _ in range(tokens_maxlen)]
            for i in range(len(probe_mask)):
                if (probe_mask[i] == 1):
                    label[i] = self.latent_encoder.encode_single(s_results[count])
                    count += 1

            labels.append(label)

        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "probe_mask": torch.tensor(probe_masks, dtype=torch.bool),
            "attention_mask": torch.tensor(attention_masks),
        }
        return batch


def build_dataloader(
    dataset,
    batch_size,
    **kwargs,
):
    collator = LSVCollator(**kwargs)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        **kwargs,
    )

    return dataloader

def build_probe_dataloader(
    dataset,
    batch_size,
    latent_encoder,
    pad_token_id,
    tokenizer=None,
    **kwargs,
):
    collator = ProbeCollator(
        latent_encoder=latent_encoder,
        pad_token_id=pad_token_id,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        **kwargs,
    )

    return dataloader


def get_optimizer(
    model,
    learning_rate=1e-5, 
    weight_decay=0.01
):
    params = [
        {"params": [p for n, p in model.named_parameters() if "latent_projector" not in n], "lr": learning_rate}
    ]

    optimizer = torch.optim.AdamW(
        params, betas=(0.9, 0.99), weight_decay=weight_decay
    )
    
    return optimizer

def get_probe_optimizer(
    model,
    learning_rate=1e-5, 
    weight_decay=0.01
):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay
    )
    
    return optimizer

def get_constant_scheduler(
    optimizer,
    last_epoch=-1
):
    return LambdaLR(optimizer, lambda x: 1, last_epoch=last_epoch)