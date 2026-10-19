"""Text files -> one token stream per split; validation never overlaps training files."""
from __future__ import annotations
import random
from pathlib import Path
import torch


def load_text_files(paths, tokenizer, min_length=1):
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError('No corpus files provided')
    ids = []
    original_bytes = 0
    for path in paths:
        text = path.read_text(encoding='utf-8-sig')
        if len(text) < min_length:
            continue
        original_bytes += len(text.encode('utf-8'))
        ids.extend(tokenizer.encode(text))
        ids.append(tokenizer.eos_id)
    if len(ids) < 4:
        raise ValueError('Not enough input text')
    return torch.tensor(ids, dtype=torch.long), original_bytes


class TextBatches:
    def __init__(self, train_ids, valid_ids, seed=42):
        self.train = train_ids
        self.valid = valid_ids
        self.rng = random.Random(seed)

    def batch(self, split, batch_size, length, device):
        ids = self.train if split == 'train' else self.valid
        last = ids.numel() - length - 1
        if last < 0:
            raise ValueError(f'Split {split} is too short for context {length}')
        if split == 'train':
            starts = [self.rng.randrange(last+1) for _ in range(batch_size)]
        else:
            # deterministic sample of valid positions, diverse across eval batches
            starts = [((j * 9973) % (last+1)) for j in range(batch_size)]
        out = torch.stack([ids[i:i+length+1] for i in starts])
        return out.to(device, non_blocking=device.type == 'cuda')

    def eval_iterator(self, batch_size, length, device, batches=5):
        ids = self.valid
        last = ids.numel() - length - 1
        if last < 0:
            raise ValueError('Validation file is too short')
        count = batch_size * batches
        for batch in range(batches):
            starts = [round(last * (batch * batch_size + j)/(max(1,count-1)))
                      for j in range(batch_size)]
            yield torch.stack([ids[i:i+length+1] for i in starts]).to(device)
