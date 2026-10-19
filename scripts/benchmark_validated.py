"""Same-input CPU/GPU validation and throughput between PCMN and Transformer.

Provides numbers, not fair scientific claims unless model scale, dataset and
training FLOPs are controlled. Distinct checkpoints may have different training.
"""
import argparse,json,math,statistics,time
from pathlib import Path
import torch
from torch.nn import functional as F
from pcmn.text_core import TextConfig,PCMNTextLM
from pcmn.text_baseline import TransformerBaseline
from pcmn.stream_data import mmap_tokens


def load(path,device):
    ck=torch.load(path,map_location='cpu',weights_only=False) # trusted only
    cfg=TextConfig(**ck['model_config'])
    kind=ck.get('architecture','pcmn')
    model=(PCMNTextLM(cfg) if kind=='pcmn' else TransformerBaseline(cfg)).to(device).eval()
    model.load_state_dict(ck['model'])
    return model,kind,ck.get('tokenizer')


def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def benchmark_one(m,kind,examples,device,gen=12,repeats=3):
    losses=[]
    with torch.inference_mode():
        for batch in examples:
            loss=m.loss(batch)
            losses.append(float(loss))
        def prefill():return m.prefill(examples[0])
        def gen_fn():
            logits,cache=m.prefill(examples[0])
            for pos in range(gen):
                if kind=='pcmn':
                    logits,cache=m.step(logits.argmax(-1),cache)
                else:
                    logits,cache=m.step(logits.argmax(-1),cache,position=examples[0].size(1)+pos)
        def timed(fn):
            for _ in range(2):fn()
            sync(device); samples=[]
            for _ in range(repeats):
                before=time.perf_counter();fn();sync(device)
                samples.append(time.perf_counter()-before)
            return statistics.median(samples)
        prefill_ms=timed(prefill)*1000
        full_ms=timed(gen_fn)*1000
    return {'nll':sum(losses)/len(losses),'perplexity':math.exp(min(50,sum(losses)/len(losses))),
            'prefill_ms':prefill_ms,'generation_including_prefill_ms':full_ms,
            'generation_tokens_per_sec_including_prefill':gen/max(full_ms/1000,1e-8),
            'parameters':m.n_parameters()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pcmn',required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--corpus',required=True,type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--eval-windows',type=int,default=8)
    p.add_argument('--gen',type=int,default=12)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(17)
    device=torch.device(a.device)
    first,k1,tok1=load(a.pcmn,device)
    second,k2,tok2=load(a.baseline,device)
    if k1!='pcmn' or k2!='transformer':raise ValueError('Checkpoint architecture mismatch')
    if tok1!=tok2:raise ValueError('Tokenizers differ; comparison invalid')
    data=mmap_tokens(a.corpus/'valid.bin')
    if len(data)<a.context+2:raise ValueError('Corpus too short')
    # Shared evenly spaced windows in one split (may overlap).
    starts=[int(i*(len(data)-a.context-1)/max(a.eval_windows-1,1)) for i in range(a.eval_windows)]
    examples=[data[s:s+a.context+1].long().unsqueeze(0).to(device) for s in starts]
    result={'device':str(device),'torch':torch.__version__,'context':a.context,'windows':a.eval_windows,
            'tokens_generated':a.gen,'same_tokenizer':True,'warning':'Different training budgets: not a scientific architecture-quality comparison',
            'pcmn':benchmark_one(first,k1,examples,device,a.gen),
            'transformer':benchmark_one(second,k2,examples,device,a.gen)}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
