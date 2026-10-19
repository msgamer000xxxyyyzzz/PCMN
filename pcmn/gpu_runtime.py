"""Honest runtime checks for CUDA; all CUDA gradients use PyTorch reference.

No claim of fused CUDA speedup until a backwards-enabled kernel is implemented.
"""
import torch


def capabilities():
    available=torch.cuda.is_available()
    devices=[]
    if available:
        for idx in range(torch.cuda.device_count()):
            p=torch.cuda.get_device_properties(idx)
            devices.append({'index':idx,'name':p.name,'compute_capability':f'{p.major}.{p.minor}',
                            'vram_bytes':p.total_memory,
                            'fp16_autocast':True,
                            'bf16_recommended':p.major>=8})
    return {'cuda_available':available,'torch_version':torch.__version__,
            'cuda_runtime':torch.version.cuda,'devices':devices}


def autocast_for(device,precision='fp32'):
    device=torch.device(device)
    if device.type!='cuda' and precision!='fp32':
        raise ValueError('Mixed precision requested without CUDA')
    if precision not in ('fp32','fp16','bf16'):
        raise ValueError('precision must be fp32/fp16/bf16')
    if device.type=='cuda' and precision=='bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('BF16 not supported on selected CUDA environment')
    from contextlib import nullcontext
    return (torch.autocast('cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16)
            if device.type=='cuda' and precision!='fp32' else nullcontext())
