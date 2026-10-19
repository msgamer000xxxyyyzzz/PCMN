"""Instruction-tuning data: mask all non-assistant messages from the LM loss."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from pcmn.agent import tagged_message

ROLES={'system','user','assistant','tool'}


def read_chat_examples(path):
    items=[]
    for lineno,line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(),1):
        if not line.strip():continue
        obj=json.loads(line)
        if not isinstance(obj,dict) or not isinstance(obj.get('messages'),list):
            raise ValueError(f'Expected messages list at line {lineno}')
        msgs=obj['messages']
        if not msgs or msgs[-1].get('role')!='assistant':
            raise ValueError(f'Last message must be assistant at line {lineno}')
        for msg in msgs:
            if not isinstance(msg,dict) or msg.get('role') not in ROLES or not isinstance(msg.get('content'),str):
                raise ValueError(f'Bad chat message at line {lineno}')
        items.append(msgs)
    if not items:raise ValueError('No chat examples')
    return items


def encode_chat(messages, tokenizer, context):
    ids=[]
    target=[]
    for msg in messages:
        part=tokenizer.encode(tagged_message(msg['role'],msg['content']))
        ids.extend(part)
        target.extend(part if msg['role']=='assistant' else [-100]*len(part))
    if len(ids)>context+1:
        ids=ids[-(context+1):]
        target=target[-(context+1):]
    if len(ids)<2 or all(v==-100 for v in target[1:]):
        raise ValueError('Context too short to include supervised assistant text')
    return ids,target


def chat_batch(examples,tokenizer,context,device):
    batch=[encode_chat(msgs,tokenizer,context) for msgs in examples]
    maximum=max(len(a) for a,b in batch)
    x=torch.full((len(batch),maximum),tokenizer.eos_id,dtype=torch.long)
    y=torch.full((len(batch),maximum),-100,dtype=torch.long)
    for n,(ids,targets) in enumerate(batch):
        x[n,:len(ids)]=torch.tensor(ids)
        y[n,:len(targets)]=torch.tensor(targets)
    return x.to(device),y.to(device)
