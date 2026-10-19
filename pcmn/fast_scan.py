"""Optional CPU scan kernel for inference; PyTorch default preserves autograd.

No unsafe auto-loading: enable explicitly with configure_cpu_backend(path).
Compatibility: only float32 contiguous CPU tensors and inference_mode/no_grad.
"""
from __future__ import annotations
import ctypes
from pathlib import Path
import torch

_lib = None
_cuda_extension = None


def configure_cpu_backend(path=None):
    global _lib
    if path is None:
        _lib = None
        return False
    lib = ctypes.CDLL(str(Path(path).resolve(strict=True)))
    f = lib.pcmn_scan_f32
    f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    f.restype = ctypes.c_int
    _lib = lib
    return True


def cpu_scan(a, b, initial=None):
    if _lib is None or torch.is_grad_enabled() or a.device.type != 'cpu' or a.dtype != torch.float32:
        return None
    if b.device.type != 'cpu' or b.dtype != torch.float32 or a.ndim != 3 or a.shape != b.shape:
        return None
    batch, length, channels = a.shape
    if length == 0:
        return None
    if initial is not None and (initial.shape != (batch,channels) or initial.device.type != 'cpu'
                                or initial.dtype != torch.float32):
        return None
    a = a.contiguous(); b = b.contiguous()
    initial = None if initial is None else initial.contiguous()
    out = torch.empty_like(b)
    ptr = lambda v: None if v is None else v.data_ptr()
    status = _lib.pcmn_scan_f32(ptr(a), ptr(b), ptr(initial), ptr(out), batch, length, channels)
    if status:
        raise RuntimeError('PCMN CPU kernel rejected scan dimensions')
    return out


def configure_cuda_backend(extension=None):
    """Enable prebuilt torch CUDA extension. No autograd provided."""
    global _cuda_extension
    if extension is not None and not hasattr(extension, 'scan'):
        raise TypeError('CUDA extension needs a scan(a,b,initial) method')
    _cuda_extension=extension
    return extension is not None


class _CudaScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b,initial):
        y=_cuda_extension.scan(a.contiguous(),b.contiguous(),initial.contiguous())
        ctx.save_for_backward(a,initial,y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        a,initial,y=ctx.saved_tensors
        da,db,dinitial=_cuda_extension.scan_backward(a.contiguous(),y.contiguous(),
                                                    initial.contiguous(),grad_y.contiguous())
        return da,db,dinitial if initial.numel() else None


def cuda_scan(a,b,initial=None):
    if (_cuda_extension is None or a.device.type != 'cuda' or a.dtype != torch.float32
            or b.dtype != torch.float32 or b.shape!=a.shape or b.device!=a.device or a.ndim!=3):
        return None
    if initial is not None and (initial.shape!=(a.size(0),a.size(2)) or
                                initial.device!=a.device or initial.dtype!=torch.float32):
        return None
    init=(torch.empty(0,device=a.device,dtype=torch.float32) if initial is None
          else initial.contiguous())
    if torch.is_grad_enabled() and (a.requires_grad or b.requires_grad or init.requires_grad):
        if not hasattr(_cuda_extension,'scan_backward'):
            return None  # older v0.4 inference-only binary; keep PyTorch autograd
        return _CudaScanFunction.apply(a,b,init)
    return _cuda_extension.scan(a.contiguous(),b.contiguous(),init)
