"""Binary pretokenized corpus and bounded-RAM sequential samplers."""
from __future__ import annotations
from pathlib import Path
import os
import bisect
import random
import torch


def mmap_tokens(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size % 4:
        raise ValueError(f'Expected a uint32 token stream: {path}')
    count = path.stat().st_size // 4
    if count < 4:
        raise ValueError(f'Not enough tokens: {path}')
    # Platform is little-endian on supported Windows/Linux x64.
    return torch.from_file(str(path.resolve()), shared=False, size=count, dtype=torch.int32)


def mmap_doc_offsets(path):
    path=Path(path)
    if path.stat().st_size%8:raise ValueError('Invalid document index')
    return torch.from_file(str(path.resolve()),shared=False,size=path.stat().st_size//8,dtype=torch.int64)


class StreamSampler:
    def __init__(self, tokens, seed=42, doc_offsets=None):
        self.tokens = tokens
        self.rng = random.Random(seed)
        self.offsets = list(int(x) for x in doc_offsets) if doc_offsets is not None else [0,len(tokens)]
        if self.offsets[0]!=0 or self.offsets[-1]!=len(tokens) or any(
                b<=a for a,b in zip(self.offsets,self.offsets[1:])):
            raise ValueError('Document index offsets must increase from zero to corpus size')

    def chunks(self, batch_size, chunk_len, n_chunks, device, fixed_start=None):
        need = chunk_len * n_chunks + 1
        choices=[]
        docs=[]
        total=0
        for left,right in zip(self.offsets,self.offsets[1:]):
            positions=max(0,right-left-need+1)
            if positions:
                total+=positions
                choices.append(total)
                docs.append(left)
        if not total:
            raise ValueError(f'No document long enough for {need} tokens')
        starts=[]
        for i in range(batch_size):
            # Sample valid offsets uniformly, weighting longer documents by
            # available training-window positions rather than document count.
            choice=self.rng.randrange(total) if fixed_start is None else (
                (fixed_start*104729+i*9973)%total)
            j=bisect.bisect_right(choices,choice)
            previous=0 if j==0 else choices[j-1]
            starts.append(docs[j]+choice-previous)
        for part in range(n_chunks):
            off = part * chunk_len
            data = torch.stack([self.tokens[st+off:st+off+chunk_len+1] for st in starts]).long()
            yield data.to(device)
