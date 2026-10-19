"""Document-exhaustive, deterministic language-model evaluation (PCMN v0.6).

Unlike random-window validation, this evaluates every eligible target token in
independent held-out documents. Recurrent state resets BETWEEN documents and
persists BETWEEN chunks of the same document. No gradients and no future tokens
may influence a prediction. This evaluates next-token likelihood, not reasoning.
"""
from __future__ import annotations
import math
import time
import torch
from torch.nn import functional as F


def evaluate_corpus(model, tokens, doc_offsets, context=128, max_documents=None, device='cpu'):
    if context < 1 or (max_documents is not None and max_documents < 1):
        raise ValueError('context and max_documents must be positive')
    boundaries = [int(v) for v in doc_offsets]
    if boundaries[0] != 0 or boundaries[-1] != len(tokens) or any(
        b <= a for a,b in zip(boundaries, boundaries[1:])
    ):
        raise ValueError('Invalid document offsets')
    device = torch.device(device)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    count = 0
    evaluated = 0
    skipped = 0
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            for first, last in zip(boundaries, boundaries[1:]):
                if max_documents is not None and evaluated >= max_documents:
                    break
                if last - first < 2:
                    skipped += 1
                    continue
                state = None
                # positions first .. last-2 are inputs; last-1 is target only
                for begin in range(first, last-1, context):
                    end = min(begin+context, last-1)
                    inputs = tokens[begin:end].to(dtype=torch.long,device=device)[None,:]
                    targets = tokens[begin+1:end+1].to(dtype=torch.long,device=device)[None,:]
                    logits, state = model.forward_chunk(inputs,state)
                    loss = F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),
                                           targets.reshape(-1), reduction='sum')
                    total_loss += float(loss)
                    count += int(targets.numel())
                evaluated += 1
    finally:
        model.train(was_training)
    if count == 0:
        raise ValueError('No eligible prediction targets')
    nll = total_loss/count
    return {'validation_nll':round(nll,7), 'perplexity':round(math.exp(min(nll,50)),7),
            'targets':count, 'documents':evaluated, 'skipped_documents':skipped,
            'context_chunk':context, 'device':str(device),
            'elapsed_seconds':round(time.perf_counter()-start,3),
            'definition':'Exhaustive teacher-forced validation; resets state at document boundaries',
            'note':'Next-token likelihood is not generative/coding/tool-use quality'}


def checkpoint_metadata(checkpoint):
    return {'step':checkpoint.get('step',None),
            'architecture':checkpoint.get('architecture','pcmn'),
            'model_config':checkpoint.get('model_config',{}),
            'tokenizer':checkpoint.get('tokenizer',{}).get('type','unknown')}
