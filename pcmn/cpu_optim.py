"""Optional CPU-only dynamic INT8 for linear layers.

Runtime conversion is convenient but does NOT produce a portable GGUF model,
nor guarantee a speedup. PyTorch's legacy eager quantization APIs are in
transition; this is an opt-in, version-dependent inference path.
"""
import torch
from torch import nn


def dynamic_int8_cpu(model):
    if next(model.parameters()).device.type!='cpu':
        raise ValueError('Dynamic int8 path is CPU-only')
    model.eval()
    try:
        from torch.ao.quantization import quantize_dynamic
    except ImportError as exc:
        raise RuntimeError('This PyTorch build does not support dynamic quantization') from exc
    return quantize_dynamic(model,{nn.Linear},dtype=torch.qint8,inplace=False)
