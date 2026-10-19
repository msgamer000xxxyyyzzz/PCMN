"""One CPU/GPU gradient step verifies model-scale trainability, not quality."""
import argparse,json,time
from pathlib import Path
import torch
from pcmn.text_core import PCMNTextLM,build_config

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',choices=['mini','30m','100m'],default='30m')
    p.add_argument('--vocab',type=int,default=259)
    p.add_argument('--device',default='cpu')
    p.add_argument('--context',type=int,default=12)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--addressable-slots',type=int,default=0)
    p.add_argument('--out',type=Path)
    a=p.parse_args();torch.set_num_threads(a.threads)
    dev=torch.device(a.device)
    torch.manual_seed(3)
    model=PCMNTextLM(build_config(a.profile,a.vocab,memory_slots=2,
                                  addressable_slots=a.addressable_slots)).to(dev)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003)
    ids=torch.randint(3,a.vocab,(1,a.context+1),device=dev)
    t=time.perf_counter()
    loss_before=model.loss(ids)
    loss_before.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    if dev.type=='cuda':torch.cuda.synchronize(dev)
    result={'profile':a.profile,'params':model.n_parameters(),'device':str(dev),
            'loss_before_update':float(loss_before.detach()),'gradient_norm':float(norm),
            'finite':bool(torch.isfinite(loss_before).item()),
            'one_training_step_seconds':time.perf_counter()-t,'full_training_complete':False}
    if a.out:a.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
if __name__=='__main__':main()
