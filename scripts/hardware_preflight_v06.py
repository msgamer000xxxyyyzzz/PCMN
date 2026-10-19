"""No-assumption host report for CPU, CUDA, architecture and safe precision."""
import argparse,json,os,platform,shutil,subprocess
from pathlib import Path
import torch


def inspect():
    data={'python':platform.python_version(), 'system':platform.platform(),
          'machine':platform.machine(),'torch':torch.__version__,
          'cpu_threads_logical':os.cpu_count(), 'torch_threads':torch.get_num_threads(),
          'cuda_available':torch.cuda.is_available(),'torch_cuda_version':torch.version.cuda,
          'nvcc_found':bool(shutil.which('nvcc')),'devices':[]}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            major,minor=torch.cuda.get_device_capability(i)
            prop=torch.cuda.get_device_properties(i)
            data['devices'].append({'index':i,'name':prop.name,'compute_capability':f'{major}.{minor}',
                                    'vram_gib':round(prop.total_memory/(1024**3),2),
                                    'fp16_possible':major>=5,
                                    'bf16_supported':major>=8,
                                    'experimental_scan_arch':f'{major}.{minor}'})
    data['recommendations']=['CPU mode: --device cpu --precision fp32',
        'Use a supported PyTorch/CUDA build for the actual GPU; verify sm_60 for Tesla P100.',
        'Tesla P100: FP16 can be tested, BF16 is unsupported.',
        'GPU kernel must be COMPILED and VALIDATED locally before any speed claims.']
    return data


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    r=inspect()
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
