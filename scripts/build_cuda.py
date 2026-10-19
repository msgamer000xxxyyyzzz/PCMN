"""Build and check experimental v0.5 training-capable CUDA extension.

Requires nvcc, CUDA-compatible PyTorch and the target GPU. The kernel is
parallel over features but sequential over tokens; benchmark before adoption.
"""
import argparse,json,torch
from pcmn.cuda_validation import build_extension,validate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arch',default='6.0',help='Tesla P100=6.0; RTX 4060=8.9')
    p.add_argument('--device',default='cuda:0')
    args=p.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({'status':'NOT_TESTED','reason':'CUDA not available'}))
        return
    module=build_extension(args.arch)
    result=validate(module,args.device)
    print(json.dumps({'status':'TESTED','arch':args.arch,'device':args.device,**result},indent=2))
    if not result['passed']:
        raise SystemExit('CUDA kernel failed gradient equivalence')
if __name__=='__main__':main()
