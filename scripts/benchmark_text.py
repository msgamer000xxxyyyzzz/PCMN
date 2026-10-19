"""Compute wall-clock inference with trained model pairs, identical device, fixed shapes."""
import argparse
import json
import statistics
import time
import torch
from pcmn.text_core import PCMNTextLM, TextConfig
from pcmn.text_baseline import TransformerBaseline


def load(path,dev):
    ck=torch.load(path,map_location='cpu',weights_only=False)  # ONLY trusted files
    cfg=TextConfig(**ck['model_config'])
    arch=ck.get('architecture','pcmn')
    model=(PCMNTextLM(cfg) if arch=='pcmn' else TransformerBaseline(cfg)).to(dev).eval()
    model.load_state_dict(ck['model'])
    return model,arch


def timed(fn,dev,repeats):
    with torch.no_grad():
        for _ in range(2): fn()
        if dev.type=='cuda':torch.cuda.synchronize(dev)
        timings=[]
        for _ in range(repeats):
            start=time.perf_counter()
            fn()
            if dev.type=='cuda':torch.cuda.synchronize(dev)
            timings.append(time.perf_counter()-start)
    return round(1000*statistics.median(timings),3)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pcmn',required=True)
    parser.add_argument('--transformer',required=True)
    parser.add_argument('--lengths',nargs='+',type=int,default=[32,128,512])
    parser.add_argument('--batch-size',type=int,default=1)
    parser.add_argument('--gen-tokens',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--num-threads',type=int,default=2)
    args=parser.parse_args()
    if args.num_threads:torch.set_num_threads(args.num_threads)
    dev=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                     'cpu' if args.device=='auto' else args.device)
    models={arch: model for model,arch in [load(args.pcmn,dev),load(args.transformer,dev)]}
    if set(models)!= {'pcmn','transformer'}:
        raise ValueError('Pass PCMN and Transformer checkpoints')
    counts={k:v.n_parameters() for k,v in models.items()}
    print(json.dumps({'device':str(dev),'params':counts}))
    for length in args.lengths:
        inputs=torch.randint(3,models['pcmn'].config.vocab_size,
                             (args.batch_size,length),device=dev)
        for arch,m in models.items():
            if m.config.vocab_size != models['pcmn'].config.vocab_size:
                raise ValueError('Tokenizers must share vocabulary size')
            forward_ms=timed(lambda:m(inputs),dev,args.repeats)
            if arch=='pcmn':
                prefill_ms=timed(lambda:m.prefill(inputs),dev,args.repeats)
                def gen():
                    logits,states=m.prefill(inputs)
                    for _ in range(args.gen_tokens):
                        logits,states=m.step(logits.argmax(-1),states)
            else:
                prefill_ms=timed(lambda:m.prefill(inputs),dev,args.repeats)
                def gen():
                    logits,states=m.prefill(inputs)
                    for pos in range(length,length+args.gen_tokens):
                        logits,states=m.step(logits.argmax(-1),states,position=pos)
            generation_ms=timed(gen,dev,args.repeats)
            print(json.dumps({'arch':arch,'length':length,
                 'forward_ms':forward_ms,'prefill_ms':prefill_ms,
                 'generation_ms_with_prefill':generation_ms,
                 'generation_tokens_per_sec_with_prefill':round(
                       args.gen_tokens*args.batch_size*1000/generation_ms,2)},
                 ensure_ascii=False),flush=True)

if __name__=='__main__':
    main()
