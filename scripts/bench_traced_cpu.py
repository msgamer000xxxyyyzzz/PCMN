"""Reproducible traced vs eager PyTorch CPU *decode-only* comparisons."""
import argparse,json,statistics,time
from pathlib import Path
import torch
from pcmn.text_core import PCMNTextLM,TextConfig
from pcmn.traced_cpu import TracedCPUDecoder


def bench(model, decoder, t, length, trials):
    with torch.inference_mode():
        for _ in range(2):
            state=model.init_states(t.size(0))
            for _ in range(length):_,state=decoder.step(t,state)
        times=[]
        for _ in range(trials):
            states=model.init_states(t.size(0))
            start=time.perf_counter()
            for _ in range(length):
                _,states=decoder.step(t,states)
            times.append(time.perf_counter()-start)
    return length/statistics.median(times)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--tokens',type=int,default=50)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    torch.set_num_threads(a.threads)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False) # trusted local artifact
    m=PCMNTextLM(TextConfig(**ck['model_config'])).eval()
    m.load_state_dict(ck['model'])
    jit=TracedCPUDecoder(m)
    token=torch.tensor([min(4,m.config.vocab_size-1)])
    result={'checkpoint':str(a.checkpoint),'params':m.n_parameters(),'threads':a.threads,
            'length':a.tokens,'backend':'torchscript_trace_step','device':'cpu'}
    result['eager_tps']=round(bench(m,m,token,a.tokens,a.repeats),2)
    result['traced_tps']=round(bench(m,jit,token,a.tokens,a.repeats),2)
    result['speedup']=round(result['traced_tps']/result['eager_tps'],3)
    with torch.inference_mode():
        old=m.init_states(1);tr=m.init_states(1);diff=0
        for _ in range(a.tokens):
            x,old=m.step(token,old);y,tr=jit.step(token,tr)
            diff=max(diff,float((x-y).abs().max()))
    result['max_abs_logits_diff']=diff
    print(json.dumps(result,indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
