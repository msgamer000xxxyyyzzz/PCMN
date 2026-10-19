"""GPU correctness/performance smoke test; skips clearly when CUDA absent."""
import argparse,json,time
from pathlib import Path
import torch
from pcmn.gpu_runtime import capabilities,autocast_for
from pcmn.text_core import PCMNTextLM,TextConfig

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--precision',choices=['fp32','fp16','bf16'],default='fp16')
    p.add_argument('--context',type=int,default=64)
    p.add_argument('--tokens',type=int,default=16)
    a=p.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({'status':'NOT_TESTED','reason':'No CUDA device available',**capabilities()}))
        return
    dev=torch.device(a.device)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False) # trusted only
    m=PCMNTextLM(TextConfig(**ck['model_config'])).to(dev).eval()
    m.load_state_dict(ck['model'])
    inp=torch.randint(3,m.config.vocab_size,(1,a.context),device=dev)
    with torch.no_grad(),autocast_for(dev,a.precision):
        x,states=m.prefill(inp,chunk_size=32)
        torch.cuda.synchronize(dev)
        start=time.perf_counter()
        for _ in range(a.tokens):
            x,states=m.step(x.argmax(-1),states)
        torch.cuda.synchronize(dev)
    print(json.dumps({'status':'MEASURED','device':str(dev),'precision':a.precision,
       'tokens_per_sec':round(a.tokens/(time.perf_counter()-start),2),**capabilities()}))
if __name__=='__main__': main()
