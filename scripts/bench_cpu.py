"""CPU throughput and first-token timing, including int8 and 0.2 compatibility.

Not a comparison against modern optimized inference engines; benchmarks here
exercise the unoptimized PyTorch reference implementation on the local CPU.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import statistics
import time
import torch
from pcmn.text_core import PCMNTextLM, TextConfig
from pcmn.cpu_optim import dynamic_int8_cpu
from pcmn.fast_scan import configure_cpu_backend


def measure(fn,repeat):
    with torch.inference_mode():
        for _ in range(2):fn()
        times=[]
        for _ in range(repeat):
            start=time.perf_counter()
            fn()
            times.append(time.perf_counter()-start)
    return statistics.median(times)


def run(path,context,generate,repeats,threads,quant):
    torch.set_num_threads(threads)
    ck=torch.load(path,map_location='cpu',weights_only=False) # trusted only
    if ck.get('architecture','pcmn')!='pcmn':raise ValueError('PCMN checkpoint required')
    model=PCMNTextLM(TextConfig(**ck['model_config'])).eval()
    model.load_state_dict(ck['model'])
    raw_params=model.n_parameters()
    if quant:model=dynamic_int8_cpu(model)
    vocab=model.config.vocab_size
    prompt=torch.randint(3,vocab,(1,context))
    chunk_size=128
    prefill_time=measure(lambda:model.prefill(prompt,chunk_size=chunk_size),repeats)

    def decode():
        _,states=model.prefill(prompt,chunk_size=chunk_size)
        x=prompt[:,-1]
        for _ in range(generate):
            logits,states=model.step(x,states)
            x=logits.argmax(-1)

    total_time=measure(decode,repeats)
    return dict(checkpoint=str(path),precision='dynamic_int8' if quant else 'fp32',
                params=raw_params,threads=threads,prompt_tokens=context,
                generated_tokens=generate,prefill_ms=round(prefill_time*1000,2),
                decode_including_prefill_ms=round(total_time*1000,2),
                isolated_decode_tokens_per_second=round(generate/max(1e-8,total_time-prefill_time),2),
                end_to_end_tokens_per_second=round(generate/max(1e-8,total_time),2),
                memory_slots=model.config.memory_slots)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--compare',type=Path,help='Optional old PCMN checkpoint')
    p.add_argument('--context',type=int,default=256)
    p.add_argument('--tokens',type=int,default=32)
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--int8',action='store_true')
    p.add_argument('--cpu-lib',type=Path,help='Optional compiled native scan library')
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    if args.cpu_lib: configure_cpu_backend(args.cpu_lib)
    if min(args.context,args.tokens,args.repeats,args.threads)<1:raise ValueError('positive values required')
    rows=[]
    for path in [args.checkpoint]+([args.compare] if args.compare else []):
        rows.append(run(path,args.context,args.tokens,args.repeats,args.threads,False))
        if args.int8:rows.append(run(path,args.context,args.tokens,args.repeats,args.threads,True))
    for row in rows:print(json.dumps(row,ensure_ascii=False),flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
