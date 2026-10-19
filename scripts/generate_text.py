"""Text generation with a checkpoint produced by scripts.train_text."""
from __future__ import annotations
import argparse
import torch
from pcmn.text_core import PCMNTextLM, TextConfig
from pcmn.text_baseline import TransformerBaseline
from pcmn.text_tokenizer import load_tokenizer
from pcmn.cpu_optim import dynamic_int8_cpu


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--prompt',default='Нейросеть ')
    p.add_argument('--tokens',type=int,default=120)
    p.add_argument('--temperature',type=float,default=0.9)
    p.add_argument('--top-k',type=int,default=40)
    p.add_argument('--greedy',action='store_true')
    p.add_argument('--device',default='auto')
    p.add_argument('--cpu-int8',action='store_true',help='Quantize linear layers at runtime on CPU')
    p.add_argument('--prefill-chunk',type=int,default=128)
    p.add_argument('--num-threads',type=int,default=2)
    args=p.parse_args()
    if args.num_threads:torch.set_num_threads(args.num_threads)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)  # ONLY trusted local checkpoints
    tok=load_tokenizer(ck['tokenizer'])
    arch=ck.get('architecture','pcmn')
    model=(PCMNTextLM(TextConfig(**ck['model_config'])) if arch == 'pcmn'
           else TransformerBaseline(TextConfig(**ck['model_config']))).to(device).eval()
    model.load_state_dict(ck['model'])
    if args.cpu_int8:
        if arch!='pcmn':raise SystemExit('--cpu-int8 is currently PCMN-only')
        model=dynamic_int8_cpu(model)
    prompt=torch.tensor([tok.encode(args.prompt) or [tok.bos_id]],device=device,dtype=torch.long)
    if arch == 'pcmn':
        output=model.generate(prompt,max_new_tokens=args.tokens,temperature=args.temperature,
                              top_k=args.top_k,greedy=args.greedy,eos_id=tok.eos_id,
                              prefill_chunk=args.prefill_chunk)
    else:
        with torch.no_grad():
            output=prompt
            logits, cache=model.prefill(prompt)
            for _ in range(args.tokens):
                if args.greedy:
                    next_tok=logits.argmax(-1,keepdim=True)
                else:
                    scaled=logits/args.temperature
                    if args.top_k:
                        cutoff=scaled.topk(min(args.top_k,scaled.size(-1)),-1).values[:,-1:]
                        scaled=scaled.masked_fill(scaled<cutoff,-torch.inf)
                    next_tok=torch.multinomial(torch.softmax(scaled,-1),1)
                output=torch.cat((output,next_tok),dim=1)
                if int(next_tok[0,0])==tok.eos_id:
                    break
                logits,cache=model.step(next_tok[:,0],cache,position=output.size(1)-1)
    print(tok.decode(output[0].tolist()))


if __name__=='__main__':
    main()
