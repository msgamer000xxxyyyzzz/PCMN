"""Causal Transformer comparison baseline; NOT part of the proposed PCMN core."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .text_core import TextConfig


class TransformerBaseline(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.config = cfg
        if cfg.dim%8:
            raise ValueError('Transformer baseline requires dimension divisible by 8')
        self.embedding=nn.Embedding(cfg.vocab_size,cfg.dim)
        self.scale=math.sqrt(cfg.dim)
        block=nn.TransformerEncoderLayer(d_model=cfg.dim,nhead=8,
            dim_feedforward=round(cfg.dim*cfg.ff_mult*1.5), dropout=cfg.dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.encoder=nn.TransformerEncoder(block,num_layers=cfg.layers,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(cfg.dim)
        self.head=nn.Linear(cfg.dim,cfg.vocab_size,bias=False)
        self.head.weight=self.embedding.weight
        nn.init.normal_(self.embedding.weight,mean=0.,std=cfg.dim**-0.5)

    def _embed(self, tokens, position=0):
        _,T=tokens.shape
        x=self.embedding(tokens)*self.scale
        pos=torch.arange(position,position+T,device=tokens.device,dtype=torch.float32)
        freq=torch.exp(torch.arange(0,self.config.dim,2,device=tokens.device,dtype=torch.float32)
                       *(-math.log(10000.0)/self.config.dim))
        angles=pos[:,None]*freq[None,:]
        encoding=torch.stack([torch.sin(angles),torch.cos(angles)],-1).reshape(T,self.config.dim)
        return x+encoding.to(x.dtype)[None,:,:]

    def forward(self, tokens):
        _,T=tokens.shape
        x=self._embed(tokens)
        mask=torch.ones(T,T,dtype=torch.bool,device=tokens.device).triu(1)
        return self.head(self.norm(self.encoder(x,mask=mask,is_causal=True)))

    @staticmethod
    def _split_qkv(layer, normed):
        q,k,v=F.linear(normed,layer.self_attn.in_proj_weight,
                        layer.self_attn.in_proj_bias).chunk(3,dim=-1)
        b,t,d=q.shape
        heads=layer.self_attn.num_heads
        size=d//heads
        return tuple(z.view(b,t,heads,size).transpose(1,2) for z in (q,k,v))

    @torch.no_grad()
    def prefill(self, tokens):
        """Returns last logits and KV cache. Positions are causal."""
        x=self._embed(tokens)
        t=tokens.size(1)
        mask=torch.ones(t,t,dtype=torch.bool,device=tokens.device).triu(1)
        states=[]
        for layer in self.encoder.layers:
            _,k,v=self._split_qkv(layer,layer.norm1(x))
            states.append((k,v))
            x=layer(x,src_mask=mask,is_causal=True)
        return self.head(self.norm(x[:,-1])),states

    @torch.no_grad()
    def step(self, tokens, states, position):
        """Cached one-token generation, no full-prompt recomputation."""
        if tokens.ndim != 1 or len(states) != len(self.encoder.layers):
            raise ValueError('Tokens [batch] and KV cache per layer required')
        x=self._embed(tokens[:,None],position)
        result=[]
        for layer,(prev_k,prev_v) in zip(self.encoder.layers,states):
            q,k,v=self._split_qkv(layer,layer.norm1(x))
            k=torch.cat((prev_k,k),dim=2)
            v=torch.cat((prev_v,v),dim=2)
            y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.,is_causal=False)
            y=y.transpose(1,2).contiguous().view(tokens.size(0),1,self.config.dim)
            x=x+layer.dropout1(layer.self_attn.out_proj(y))
            x=x+layer._ff_block(layer.norm2(x))
            result.append((k,v))
        return self.head(self.norm(x[:,0])),result

    def loss(self,tokens):
        logits=self(tokens[:,:-1])
        return F.cross_entropy(logits.float().reshape(-1,self.config.vocab_size), tokens[:,1:].reshape(-1))

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())
