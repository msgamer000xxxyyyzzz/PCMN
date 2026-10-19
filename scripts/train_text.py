"""Run: python -m scripts.train_text --train data/train.txt --valid data/valid.txt --steps 100"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import torch
from torch.nn import functional as F
from pcmn.text_core import TextConfig, PCMNTextLM, build_config
from pcmn.text_baseline import TransformerBaseline
from pcmn.text_tokenizer import ByteTokenizer, train_bpe, load_tokenizer
from pcmn.text_dataset import TextBatches, load_text_files


def add_args():
    p = argparse.ArgumentParser(description='PCMN text v0.2, training from scratch')
    p.add_argument('--train', nargs='+', required=True, help='UTF-8 training files')
    p.add_argument('--valid', nargs='+', required=True, help='UTF-8 validation files; must differ from training')
    p.add_argument('--profile', choices=['smoke','mini','30m','100m'], default='smoke')
    p.add_argument('--architecture', choices=['pcmn','transformer'],default='pcmn')
    p.add_argument('--scan-chunk', type=int, default=128)
    p.add_argument('--tokenizer', choices=['byte','bpe'], default='byte')
    p.add_argument('--vocab-size', type=int, default=16000)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--context', type=int, default=128)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--min-lr', type=float, default=0.00003)
    p.add_argument('--warmup', type=int, default=5)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--eval-every', type=int, default=25)
    p.add_argument('--eval-batches', type=int, default=3)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--out', type=Path, default=Path('runs/text-v02'))
    p.add_argument('--resume', type=Path)
    p.add_argument('--device', type=str, default='auto', help='auto | cpu | cuda | cuda:N')
    p.add_argument('--precision', choices=['fp32','fp16'], default='fp32')
    p.add_argument('--grad-checkpoint', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-threads', type=int, default=0, help='CPU torch threads, 0=default')
    return p.parse_args()


def evaluate(model, batches, n, batch_size, length, device, precision):
    model.eval()
    losses = []
    autocast = torch.autocast('cuda', dtype=torch.float16) if device.type == 'cuda' and precision=='fp16' else nullcontext()
    with torch.no_grad():
        for data in batches.eval_iterator(batch_size, length, device, n):
            with autocast:
                loss = model.loss(data)
            losses.append(float(loss))
    model.train()
    v = sum(losses)/len(losses)
    return {'val_nll':round(v,6), 'val_ppl':round(math.exp(min(v,50)),5)}


def checkpoint_save(path, model, optimizer, scaler, step, tok, args, data_rng_state):
    path.parent.mkdir(parents=True, exist_ok=True)
    info = dict(model_config=asdict(model.config), model=model.state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict() if scaler else None,
                step=step, tokenizer=tok.serialize(),
                architecture=args.architecture,
                training_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                torch_rng=torch.get_rng_state(),
                cpu_data_rng=data_rng_state)
    if torch.cuda.is_available():
        info['cuda_rng']=torch.cuda.get_rng_state_all()
    tmp = path.with_suffix('.tmp')
    torch.save(info, tmp)
    os.replace(tmp,path)


def main():
    args = add_args()
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    if args.steps < 1 or args.context < 2 or args.batch_size < 1 or args.grad_accum < 1:
        raise SystemExit('Steps, context, batch-size and grad-accum must be positive')
    train_files = [Path(p).resolve() for p in args.train]
    valid_files = [Path(p).resolve() for p in args.valid]
    if set(train_files).intersection(valid_files):
        raise SystemExit('Data leakage: training and validation paths overlap')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type != 'cuda' and args.precision=='fp16':
        raise SystemExit('fp16 mixed precision requires CUDA')
    if args.resume:
        # PyTorch checkpoint uses pickle! Never load an untrusted checkpoint.
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        tok = load_tokenizer(ck['tokenizer'])
        config = TextConfig(**ck['model_config'])
        args.architecture=ck.get('architecture','pcmn')
        start_step = ck['step']
    else:
        tok = ByteTokenizer() if args.tokenizer == 'byte' else train_bpe(train_files, args.vocab_size)
        config = build_config(args.profile, tok.vocab_size, grad_checkpoint=args.grad_checkpoint,
                              scan_chunk=args.scan_chunk)
        start_step = 0
    train_ids, train_bytes = load_text_files(train_files, tok)
    valid_ids, valid_bytes = load_text_files(valid_files, tok)
    if min(train_ids.numel(), valid_ids.numel()) <= args.context:
        raise SystemExit(f'Corpus too short for --context {args.context}')
    batches = TextBatches(train_ids, valid_ids, seed=args.seed)
    model = (PCMNTextLM(config) if args.architecture == 'pcmn' else TransformerBaseline(config)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                  betas=(0.9,0.95))
    use_amp = args.precision == 'fp16' and device.type=='cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    if args.resume:
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        if ck.get('scaler'):
            scaler.load_state_dict(ck['scaler'])
        torch.set_rng_state(ck['torch_rng'])
        if 'cuda_rng' in ck and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        batches.rng.setstate(ck['cpu_data_rng'])
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {'params':model.n_parameters(),'train_tokens':train_ids.numel(),
            'validation_tokens':valid_ids.numel(),'train_bytes':train_bytes,
            'validation_bytes':valid_bytes, 'device':str(device), 'model':asdict(config),
            'train_files':[str(p) for p in train_files], 'valid_files':[str(p) for p in valid_files],
            'tokenizer':tok.kind, 'architecture':args.architecture}
    (args.out/'run.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    log = args.out/'metrics.jsonl'
    print(json.dumps({'starting_step':start_step, **meta}, ensure_ascii=False), flush=True)
    model.train()
    clock = time.perf_counter()
    tokens_since = 0
    for step in range(start_step+1,args.steps+1):
        if step <= args.warmup and args.warmup:
            lr = args.lr * step/args.warmup
        else:
            fraction = min(1., max(0., (step-args.warmup)/max(1,args.steps-args.warmup)))
            lr = args.min_lr + 0.5*(args.lr-args.min_lr)*(1+math.cos(math.pi*fraction))
        for pg in optimizer.param_groups:
            pg['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        total_loss=0.
        for _ in range(args.grad_accum):
            data=batches.batch('train',args.batch_size,args.context,device)
            with torch.autocast('cuda',dtype=torch.float16) if use_amp else nullcontext():
                loss = model.loss(data) / args.grad_accum
            scaler.scale(loss).backward()
            total_loss += float(loss.detach())
            tokens_since += args.batch_size*args.context
        scaler.unscale_(optimizer)
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
        if not torch.isfinite(gn).item():
            raise FloatingPointError('Non-finite gradient: lower LR, turn off fp16 or inspect data')
        scaler.step(optimizer)
        scaler.update()
        if step==1 or step%args.eval_every==0 or step==args.steps:
            elapsed=max(0.001,time.perf_counter()-clock)
            stats={'step':step, 'train_nll':round(total_loss,6),
                   'lr':round(lr,9),'train_tokens_per_sec':round(tokens_since/elapsed,2),
                   'grad_norm':round(float(gn),4)}
            stats.update(evaluate(model,batches,args.eval_batches,args.batch_size,
                                  args.context,device,args.precision))
            print(json.dumps(stats,ensure_ascii=False),flush=True)
            with log.open('a',encoding='utf-8') as fh:
                fh.write(json.dumps(stats,ensure_ascii=False)+'\n')
            clock = time.perf_counter()
            tokens_since = 0
        if step%args.save_every==0 or step==args.steps:
            checkpoint_save(args.out/'last.pt',model,optimizer,scaler,step,tok,args,batches.rng.getstate())
    print('Saved:',args.out/'last.pt')


if __name__=='__main__':
    main()
