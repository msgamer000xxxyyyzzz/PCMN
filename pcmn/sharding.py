"""Inference-only layer sharding across CPU and CUDA devices.

Per-layer states stay on their owners. The tied LM head is copied to the last
compute device; its weights are not trained through this wrapper.
Cross-device activation transfer can exceed saved VRAM benefits.
"""
import torch
from torch.nn import functional as F


class ShardedPCMN:
    def __init__(self, model, devices):
        if not devices:
            raise ValueError('At least one device required')
        self.model = model.eval()
        self.devices = [torch.device(x) for x in devices]
        if len(self.devices) != len(model.blocks):
            raise ValueError('Exactly one device entry required per network layer')
        self.input_device = self.devices[0]
        self.output_device = self.devices[-1]
        model.embedding.to(self.input_device)
        for block, device in zip(model.blocks, self.devices):
            block.to(device)
        model.final_norm.to(self.output_device)
        # Tied head and input embedding cannot both own different devices.
        self.head_weight = model.embedding.weight.detach().to(self.output_device).clone()

    def init_states(self, batch):
        cfg=self.model.config
        out=[]
        for b,dev in zip(self.model.blocks,self.devices):
            d=cfg.dim
            dtype=next(b.parameters()).dtype
            row=(torch.zeros(batch,d,device=dev,dtype=torch.float32),
                 torch.zeros(batch,cfg.conv_kernel-1,d,device=dev,dtype=dtype))
            if b.memory_slots:
                row+=(torch.zeros(batch,b.memory_slots,d,device=dev,dtype=torch.float32),)
            if b.addressable_slots:
                k=torch.zeros(batch,b.addressable_slots,d,device=dev,dtype=torch.float32)
                row+=(k,torch.zeros_like(k),torch.zeros(batch,device=dev,dtype=torch.long))
            out.append(row)
        return out

    @torch.inference_mode()
    def forward_chunk(self, tokens, states=None):
        if states is None:states=self.init_states(tokens.size(0))
        x=self.model.embedding(tokens.to(self.input_device))
        new=[]
        for b,dev,old in zip(self.model.blocks,self.devices,states):
            x,ns=b.forward_with_state(x.to(dev),old)
            new.append(ns)
        return F.linear(self.model.final_norm(x),self.head_weight),new

    @torch.inference_mode()
    def prefill(self,tokens,chunk_size=128):
        if tokens.size(1)<1:raise ValueError('Empty prompt')
        states=None
        for start in range(0,tokens.size(1),chunk_size):
            logits,states=self.forward_chunk(tokens[:,start:start+chunk_size],states)
        return logits[:,-1],states

    @torch.inference_mode()
    def step(self,tokens,states):
        x=self.model.embedding(tokens.to(self.input_device))
        new=[]
        for b,dev,old in zip(self.model.blocks,self.devices,states):
            x,ns=b.step(x.to(dev),old)
            new.append(ns)
        return F.linear(self.model.final_norm(x),self.head_weight),new
