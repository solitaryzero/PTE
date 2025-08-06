import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

class PTECollator:
    def __init__(self, pad_token_id, unk_token=None, unk_token_id=-1):
        self.pad_token_id = pad_token_id
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id

    def __call__(self, elements):
        tokenlist = [e["input_ids"] for e in elements]
        query_lengths = [len(e['query_ids']) for e in elements]
        tokens_maxlen = max([len(t) for t in tokenlist])

        input_ids, labels, attention_masks = [], [], []
        for tokens, query_len in zip(tokenlist, query_lengths):
            pad_len = tokens_maxlen - len(tokens)
            input_ids.append(tokens + [self.pad_token_id]*pad_len)
            attention_masks.append([1]*len(tokens) + [0]*pad_len)

            label = [-100]*query_len + tokens[query_len:] + [-100]*pad_len
            if (self.unk_token is not None):
                for i, t in enumerate(input_ids):
                    if (t == self.unk_token_id):
                        label[i] = -100
            labels.append(label)

        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks),
        }
        return batch
    

class PTELatentCollator:
    def __init__(self, pad_token_id, latent_token, latent_token_id, dummy_latent):
        self.pad_token_id = pad_token_id
        self.latent_token = latent_token
        self.latent_token_id = latent_token_id
        self.dummy_latent = dummy_latent

    def __call__(self, elements):
        tokenlist = [e["input_ids"] for e in elements]
        query_lengths = [len(e['query_ids']) for e in elements]
        tokens_maxlen = max([len(t) for t in tokenlist])
        latent_embeds = [e["latent_embeds"] for e in elements]

        input_ids, labels, attention_masks, full_latent_embeds, token_types = [], [], [], [], []
        for tokens, query_len, latent_embed in zip(tokenlist, query_lengths, latent_embeds):
            pad_len = tokens_maxlen - len(tokens)
            input_ids.append(tokens + [self.pad_token_id]*pad_len)
            attention_masks.append([1]*len(tokens) + [0]*pad_len)
            labels.append([-100]*query_len + tokens[query_len:] + [-100]*pad_len)

            # latent features
            full_latent_embed = []
            token_type = []
            current = 0
            for t in tokens:
                if (t == self.latent_token_id):
                    full_latent_embed.append(latent_embed[current])
                    current += 1
                    token_type.append(1)
                else:
                    full_latent_embed.append(self.dummy_latent)
                    token_type.append(0)
            for _ in range(pad_len):
                full_latent_embed.append(self.dummy_latent)
                token_type.append(0)
            full_latent_embeds.append(full_latent_embed)
            token_types.append(token_type)
            
        batch = {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_masks),
            "latent_embeds": torch.tensor(full_latent_embeds, dtype=torch.float32),
            "token_types": torch.tensor(token_types),
        }
        return batch
    

def build_dataloader(
    dataset,
    pad_token_id,
    batch_size,
    latent=False,
    unk_token=None,
    unk_token_id=-1,
    latent_token=None,
    latent_token_id=-1,
    dummy_latent=None,
    **kwargs,
):
    if (latent):
        collator = PTELatentCollator(
            pad_token_id=pad_token_id,
            latent_token=latent_token,
            latent_token_id=latent_token_id,
            dummy_latent=dummy_latent,
        )
    else:
        collator = PTECollator(
            pad_token_id=pad_token_id,
            unk_token=unk_token,
            unk_token_id=unk_token_id,   
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay
    )
    return optimizer

def get_constant_scheduler(
    optimizer,
    last_epoch=-1
):
    return LambdaLR(optimizer, lambda x: 1, last_epoch=last_epoch)