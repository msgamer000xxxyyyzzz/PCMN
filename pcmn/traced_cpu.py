"""Optional TorchScript traced *decoder step* for CPU inference.

Trace requires a fixed cache layout and fixed batch size. All checkpoint
parameters and model behavior stay unchanged. No persistent JIT cache is saved.
Not a standalone C++/AVX engine and may be slower than eager on some CPUs.
"""
from __future__ import annotations
import torch
from torch import nn


class _Step(nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model=model
        self.schema=[len(item) for item in model.init_states(1,'cpu')]

    def forward(self, token, *flat):
        states=[];at=0
        for count in self.schema:
            states.append(tuple(flat[at:at+count]))
            at+=count
        logits, result=self.model.step(token,states)
        return (logits,*(v for state in result for v in state))


class TracedCPUDecoder:
    def __init__(self, model, batch=1):
        if model.embedding.weight.device.type!='cpu' or model.training:
            raise ValueError('CPU eval model required')
        self.model=model
        self.batch=batch
        wrapper=_Step(model)
        example=torch.zeros(batch,dtype=torch.long)
        states=model.init_states(batch,'cpu')
        with torch.inference_mode():
            self.traced=torch.jit.trace(wrapper,(example,*(v for s in states for v in s)),
                                        check_trace=False,strict=True)
        self.schema=wrapper.schema

    def step(self, token, states):
        if token.shape!=(self.batch,) or token.device.type!='cpu':
            raise ValueError('Invalid token batch/device for traced model')
        if [len(s) for s in states]!=self.schema:
            raise ValueError('Invalid state layout')
        out=self.traced(token,*(v for s in states for v in s))
        next_states=[];j=1
        for count in self.schema:
            next_states.append(tuple(out[j:j+count]));j+=count
        return out[0],next_states
