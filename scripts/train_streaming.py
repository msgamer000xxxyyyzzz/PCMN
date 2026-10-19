"""PCMN v0.3: state-carrying truncated-BPTT training on mmap token streams."""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from pcmn.text_core import PCMNTextLM, TextConfig, build_config
from pcmn.text_tokenizer import load_tokenizer
from pcmn.stream_data import mmap_tokens, mmap_doc_offsets, StreamSampler


def args_parser():
    p=argparse.ArgumentParser()
    p.add_argument('--corpus',type=Path,required=True,help='Directory created with prepare_corpus')
    p.add_argument('--profile',choices=['smoke','mini','30m','100m'],default='smoke')
    p.add_argument('--memory-slots',type=int,default=4)
    p.add_argument('--addressable-slots',type=int,default=0)
    p.add_argument('--memory-base-half-life',type=int,default=32)
    p.add_argument('--scan-chunk',type=int,default=128)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--segments',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--grad-accum',type=int,default=1)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--lr',type=float,default=0.0003)
    p.add_argument('--min-lr',type=float,default=0.00003)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--eval-every',type=int,default=20)
    p.add_argument('--save-every',type=int,default=50)
    p.add_argument('--eval-batches',type=int,default=2)
    p.add_argument('--clip-grad',type=float,default=1.0)
    p.add_argument('--device',default='auto')
    p.add_argument('--precision',choices=['fp32','fp16'],default='fp32')
    p.add_argument('--cuda-scan',action='store_true',help='Experimental FP32 CUDA scan; compiles extension and validates gradients')
    p.add_argument('--cuda-arch',default='6.0',help='CUDA arch for --cuda-scan: P100=6.0')
    p.add_argument('--num-threads',type=int,default=0)
    p.add_argument('--out',type=Path,default=Path('runs/stream-v03'))
    p.add_argument('--resume',type=Path)
    p.add_argument('--pretrained',type=Path,help='v0.2 PCMN checkpoint, same tokenizer vocabulary')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=0,help='Stop after N non-improving validation checks; 0 disables')
    p.add_argument('--min-improvement',type=float,default=0.0,help='Minimum NLL improvement for best checkpoint')
    return p.parse_args()


def move_states(states):
    return PCMNTextLM.detach_states(states)


def stream_loss(model, chunk, cache):
    logits, cache=model.forward_chunk(chunk[:,:-1],cache)
    loss=F.cross_entropy(logits.float().reshape(-1,model.config.vocab_size),chunk[:,1:].reshape(-1))
    return loss,cache


@torch.no_grad()
def evaluate(model, sampler, args, device, amp):
    model.eval()
    losses=[]
    for i in range(args.eval_batches):
        cache=None
        for chunk in sampler.chunks(args.batch_size,args.context,args.segments,device,fixed_start=i+1):
            with amp():
                loss,cache=stream_loss(model,chunk,cache)
            cache=move_states(cache)
            losses.append(float(loss))
    model.train()
    return sum(losses)/len(losses)


def save(path, model, optimizer, scaler, step, tokenizer_info, args, sampler, best_val_nll=None, patience_bad=0):
    path.parent.mkdir(parents=True,exist_ok=True)
    ck={'model_config':asdict(model.config),'model':model.state_dict(),
        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict() if scaler else None,
        'step':step,'tokenizer':tokenizer_info,'architecture':'pcmn',
        'training_args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'torch_rng':torch.get_rng_state(),'cpu_data_rng':sampler.rng.getstate(),
        'best_val_nll':best_val_nll,'patience_bad':patience_bad}
    if torch.cuda.is_available():ck['cuda_rng']=torch.cuda.get_rng_state_all()
    tmp=path.with_suffix('.tmp')
    torch.save(ck,tmp)
    os.replace(tmp,path)


def main():
    args=args_parser()
    if min(args.context,args.segments,args.steps,args.grad_accum,args.batch_size,args.eval_every,args.save_every,args.eval_batches)<1 or args.patience<0 or args.min_improvement<0:
        raise SystemExit('Positive context, segments, steps, batches and frequencies required')
    if args.pretrained and args.resume:
        raise SystemExit('Choose either --pretrained or --resume')
    if args.num_threads:torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    if device.type!='cuda' and args.precision=='fp16':raise SystemExit('fp16 is CUDA-only')
    if args.cuda_scan:
        if device.type!='cuda':raise SystemExit('--cuda-scan requires CUDA')
        from pcmn.cuda_validation import build_extension,validate
        extension=build_extension(args.cuda_arch)
        verified=validate(extension,device)
        print('CUDA forward/backward check:',verified,flush=True)
        if not verified['passed']:raise SystemExit('CUDA scan verification FAILED')
    use_amp=device.type=='cuda' and args.precision=='fp16'
    def amp():return torch.autocast('cuda',dtype=torch.float16) if use_amp else nullcontext()
    corpus=json.loads((args.corpus/'corpus.json').read_text(encoding='utf-8'))
    tokenizer_info=json.loads((args.corpus/'tokenizer.json').read_text(encoding='utf-8'))
    tok=load_tokenizer(tokenizer_info)
    train=StreamSampler(mmap_tokens(args.corpus/'train.bin'),seed=args.seed,doc_offsets=mmap_doc_offsets(args.corpus/'train.idx'))
    valid=StreamSampler(mmap_tokens(args.corpus/'valid.bin'),seed=117,doc_offsets=mmap_doc_offsets(args.corpus/'valid.idx'))
    if args.resume:
        ck=torch.load(args.resume,map_location='cpu',weights_only=False)  # trusted local file only
        cfg=TextConfig(**ck['model_config'])
        if ck['tokenizer'] != tokenizer_info:
            raise SystemExit('Checkpoint tokenizer does not match corpus')
        start=ck['step']
    else:
        cfg=build_config(args.profile,tok.vocab_size,memory_slots=args.memory_slots,
                         memory_base_half_life=args.memory_base_half_life,scan_chunk=args.scan_chunk,
                         addressable_slots=args.addressable_slots)
        start=0
    model=PCMNTextLM(cfg).to(device)
    if args.resume:
        model.load_state_dict(ck['model'])
    elif args.pretrained:
        old=torch.load(args.pretrained,map_location='cpu',weights_only=False)
        if old.get('architecture','pcmn')!='pcmn' or old['model_config']['vocab_size']!=tok.vocab_size:
            raise SystemExit('Pretrained model architecture/tokenizer vocabulary mismatch')
        result=model.load_state_dict(old['model'],strict=False)
        if result.unexpected_keys:
            raise SystemExit(f'Unexpected pretrained weights: {result.unexpected_keys}')
        print('Loaded old parameters; new memory randomly initialized:',result.missing_keys,flush=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95),weight_decay=0.1)
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    if args.resume:
        optimizer.load_state_dict(ck['optimizer'])
        if ck.get('scaler'):scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['torch_rng'])
        train.rng.setstate(ck['cpu_data_rng'])
        if torch.cuda.is_available() and ck.get('cuda_rng'):torch.cuda.set_rng_state_all(ck['cuda_rng'])
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'run.json').write_text(json.dumps({'config':asdict(cfg),'parameters':model.n_parameters(),
        'device':str(device),'corpus':corpus},ensure_ascii=False,indent=2),encoding='utf-8')
    log=args.out/'metrics.jsonl'
    print('params:',model.n_parameters(),'device:',device,'state slots:',cfg.memory_slots,flush=True)
    model.train()
    clock=time.perf_counter()
    previous_log_step=start
    best_val_nll=ck.get('best_val_nll') if args.resume else None
    patience_bad=ck.get('patience_bad',0) if args.resume else 0
    for step in range(start+1,args.steps+1):
        if args.warmup and step<=args.warmup:lr=args.lr*step/args.warmup
        else:
            frac=max(0,min(1,(step-args.warmup)/max(1,args.steps-args.warmup)))
            lr=args.min_lr+0.5*(args.lr-args.min_lr)*(1+math.cos(math.pi*frac))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        total=0
        for _ in range(args.grad_accum):
            states=None
            for chunk in train.chunks(args.batch_size,args.context,args.segments,device):
                with amp(): loss,states=stream_loss(model,chunk,states)
                scaler.scale(loss/(args.grad_accum*args.segments)).backward()
                # Truncated BPTT: tokens influence later segments through carried
                # values, but gradients are intentionally detached here.
                states=move_states(states)
                total+=float(loss.detach())/(args.grad_accum*args.segments)
        scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Gradient is not finite')
        scaler.step(optimizer)
        scaler.update()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            val=evaluate(model,valid,args,device,amp)
            elapsed=max(time.perf_counter()-clock,1e-8)
            result={'step':step,'train_nll':round(total,6),'val_nll':round(val,6),
                'val_ppl':round(math.exp(min(50,val)),5),'tokens_per_sec_train_incl_eval':round(
                (args.context*args.segments*args.batch_size*args.grad_accum*(step-previous_log_step))/elapsed,2),
                'elapsed_since_last_log_s':round(elapsed,2),'lr':lr,'grad_norm':round(float(norm),4)}
            print(json.dumps(result),flush=True)
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
            clock=time.perf_counter()
            previous_log_step=step
            improved=(best_val_nll is None or val < best_val_nll-args.min_improvement)
            if improved:
                best_val_nll=val
                patience_bad=0
                save(args.out/'best.pt',model,optimizer,scaler,step,tokenizer_info,args,train,
                     best_val_nll,patience_bad)
            else:
                patience_bad+=1
        if step%args.save_every==0 or step==args.steps or (args.patience and patience_bad>=args.patience):
            save(args.out/'last.pt',model,optimizer,scaler,step,tokenizer_info,args,train,
                 best_val_nll,patience_bad)
        if args.patience and patience_bad >= args.patience:
            print(f'Early stopping at step {step}; best val NLL={best_val_nll:.6f}',flush=True)
            break
    print('Checkpoint:',args.out/'last.pt',flush=True)

if __name__=='__main__':main()
