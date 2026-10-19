"""Synthetic distant-key retrieval diagnostic; reports accuracy, not context guarantee.

The model receives a key at the beginning and a question at the end. It must
choose the correct next token among A and B. Exact score depends on the
training corpus and does not replace long-context benchmark suites.
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import torch
from pcmn.text_core import PCMNTextLM,TextConfig
from pcmn.text_tokenizer import load_tokenizer


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--distances',nargs='+',type=int,default=[128,512,2048])
    p.add_argument('--cases',type=int,default=4)
    p.add_argument('--prefill-chunk',type=int,default=128)
    p.add_argument('--device',default='auto')
    p.add_argument('--num-threads',type=int,default=2)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.num_threads:torch.set_num_threads(args.num_threads)
    dev=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else 'cpu' if args.device=='auto' else args.device)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False) # trusted only
    if ck.get('architecture','pcmn')!='pcmn':raise SystemExit('PCMN only')
    tok=load_tokenizer(ck['tokenizer'])
    a,b=tok.encode('A'),tok.encode('B')
    if len(a)!=1 or len(b)!=1 or a[0]==b[0]:raise SystemExit('Tokenizer must encode A and B as distinct single tokens')
    model=PCMNTextLM(TextConfig(**ck['model_config'])).to(dev).eval()
    model.load_state_dict(ck['model'])
    rng=random.Random(9944)
    results=[]
    for depth in args.distances:
        if depth<0:raise ValueError('distance must be >=0')
        cases=[]
        for i in range(args.cases):
            label=rng.choice('AB')
            start=f'The secret key is {label}. Remember the key.\n'
            filler='This paragraph contains ordinary unrelated historical text.\n'
            encoded=tok.encode(start)+ (tok.encode(filler)*(depth//max(1,len(tok.encode(filler)))+1))[:depth]
            encoded+=tok.encode('\nQuestion: What was the secret key? Answer: ')
            inp=torch.tensor(encoded,device=dev,dtype=torch.long).unsqueeze(0)
            with torch.inference_mode():
                logits,_=model.prefill(inp,chunk_size=args.prefill_chunk)
                ab_logits=logits[0,[a[0],b[0]]].float()
                probs=torch.softmax(ab_logits,dim=0)
                pred='A' if probs[0]>=probs[1] else 'B'
            cases.append({'expected':label,'predicted':pred,
                          'p_correct_conditional_AB':round(float(probs[0] if label=='A' else probs[1]),5)})
        result={'gap_tokens':depth,'cases':args.cases,
                'accuracy':round(sum(z['expected']==z['predicted'] for z in cases)/len(cases),4),
                'avg_p_correct_conditional_AB':round(sum(z['p_correct_conditional_AB'] for z in cases)/len(cases),5)}
        print(json.dumps(result),flush=True)
        results.append(result)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
