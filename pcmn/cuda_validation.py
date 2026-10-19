"""Compile / validate optional experimental CUDA FP32 scan on user's CUDA host."""
from pathlib import Path
import os
import torch
from .fast_scan import configure_cuda_backend
from .text_core import chunked_affine_scan


def build_extension(arch='6.0'):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA runtime or GPU not available')
    from torch.utils.cpp_extension import load
    os.environ['TORCH_CUDA_ARCH_LIST'] = arch
    root = Path(__file__).resolve().parents[1]
    module = load(name='pcmn_cuda_scan_v05',
                  sources=[str(root/'native/cuda_scan.cpp'),str(root/'native/cuda_scan.cu')],
                  extra_cuda_cflags=['-O3'],verbose=True)
    configure_cuda_backend(module)
    return module


def validate(extension,device='cuda:0',atol=3e-4,rtol=3e-4):
    """Compare independent gradients including initial-state derivatives."""
    device=torch.device(device)
    if device.type!='cuda': raise ValueError('CUDA device required')
    torch.manual_seed(910)
    def inputs():
        aa=(torch.rand((2,41,37),device=device)*0.8+0.1).requires_grad_()
        bb=torch.randn_like(aa).requires_grad_()
        hh=torch.randn((2,37),device=device).requires_grad_()
        return aa,bb,hh
    g=torch.randn((2,41,37),device=device)
    configure_cuda_backend(None)
    a,b,h=inputs()
    expected=chunked_affine_scan(a,b,17,h)
    reference=torch.autograd.grad((expected*g).sum(),(a,b,h))
    configure_cuda_backend(extension)
    aa=a.detach().clone().requires_grad_()
    bb=b.detach().clone().requires_grad_()
    hh=h.detach().clone().requires_grad_()
    actual=chunked_affine_scan(aa,bb,17,hh)
    got=torch.autograd.grad((actual*g).sum(),(aa,bb,hh))
    metrics={'forward_max_abs':float((actual-expected).abs().max()),
             'grad_a_max_abs':float((got[0]-reference[0]).abs().max()),
             'grad_b_max_abs':float((got[1]-reference[1]).abs().max()),
             'grad_h0_max_abs':float((got[2]-reference[2]).abs().max())}
    metrics['passed']=bool(torch.allclose(actual,expected,atol=atol,rtol=rtol) and
                        all(torch.allclose(x,y,atol=atol,rtol=rtol) for x,y in zip(got,reference)))
    return metrics
