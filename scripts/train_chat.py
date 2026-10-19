"""Supervised instruction/chat tuning from an existing PCMN v0.2 or v0.3 checkpoint.

Input JSONL rows: {"messages":[{"role":"user","content":"..."},...]}
Only assistant spans receive cross-entropy loss. This is not RLHF or a safe
production agent; a small local dataset proves the code path only.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import torch
from torch.nn import functional as F
from pcmn.text_core import PCMNTextLM,TextConfig
from pcmn.text_tokenizer import load_tokenizer
from pcmn.chat_data import read_chat_examples, chat_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--valid',type=Path,required=True)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--context',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=0.0001)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--device',default='auto')
    p.add_argument('--num-threads',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--out',type=Path,default=Path('runs/chat-v03'))
    args=p.parse_args()
    if args.train.resolve()==args.valid.resolve():raise SystemExit('Separate training and validation files required')
    if min(args.steps,args.context,args.batch_size,args.eval_every)<1:raise SystemExit('Positive values required')
    if args.num_threads:torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    rng=random.Random(args.seed)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False) # trusted checkpoint only
    if ck.get('architecture','pcmn')!='pcmn':raise SystemExit('PCMN checkpoint required')
    tok=load_tokenizer(ck['tokenizer'])
    model=PCMNTextLM(TextConfig(**ck['model_config'])).to(device)
    model.load_state_dict(ck['model'])
    train=read_chat_examples(args.train)
    valid=read_chat_examples(args.valid)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.02)
    args.out.mkdir(parents=True,exist_ok=True)

    def loss_fn(messages):
        x,y=chat_batch(messages,tok,args.context,device)
        logits=model(x[:,:-1])
        labels=y[:,1:]
        active=(labels!=-100).sum()
        if active.item()==0:raise ValueError('No assistant labels in batch')
        return F.cross_entropy(logits.float().reshape(-1,model.config.vocab_size),labels.reshape(-1),ignore_index=-100)

    def evaluate():
        model.eval()
        with torch.no_grad():
            scores=[float(loss_fn([item])) for item in valid]
        model.train()
        return sum(scores)/len(scores)

    model.train()
    for step in range(1,args.steps+1):
        sample=[train[rng.randrange(len(train))] for _ in range(args.batch_size)]
        loss=loss_fn(sample)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step==1 or step%args.eval_every==0 or step==args.steps:
            print(json.dumps({'step':step,'train_nll':round(float(loss.detach()),5),
                               'valid_nll':round(evaluate(),5),'parameters':model.n_parameters()}),flush=True)
    result={'model_config':asdict(model.config),'model':model.state_dict(),
            'tokenizer':ck['tokenizer'],'architecture':'pcmn','step':args.steps,
            'training_stage':'supervised_chat','parent_checkpoint':str(args.checkpoint)}
    filename=args.out/'last.pt'
    tmp=args.out/'last.tmp'
    torch.save(result,tmp)
    os.replace(tmp,filename)
    print('Saved:',filename)

if __name__=='__main__':main()
