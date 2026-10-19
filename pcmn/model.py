"""Transformer-free experimental neural language-model backbone.

Training/prefill uses a parallel associative scan over an elementwise affine recurrence.
Decoding with .step() has constant recurrent-state size (w.r.t. sequence length).

WARNING: Pure PyTorch scan is an educational O(T log T) work implementation.
It does not attain the fused O(T) work kernels of Mamba/SSD, nor is it expected
necessarily to beat torch.nn.TransformerEncoder on real GPUs.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def parallel_affine_scan(a: torch.Tensor, b: torch.Tensor):
    """Inclusive scan of h[t] = a[t] * h[t-1] + b[t] with h[-1]=0.

    a, b: (batch, time, channels), 0 <= a <= 1 (not mathematically necessary).
    Outputs h (batch, time, channels). Uses log2(time) parallel doubling stages.
    Each doubling composes (a_t,b_t) with its shifted predecessor.
    """
    if a.shape != b.shape or a.ndim != 3:
        raise ValueError('a and b must both have shape [batch,time,channels]')
    if a.size(1) == 0:
        return b
    aa, bb = a, b
    k = 1
    while k < a.size(1):
        # For the k-shifted prefix: (a_i,b_i) ∘ (a_{i-k},b_{i-k}).
        new_a = torch.cat((aa[:, :k], aa[:, k:] * aa[:, :-k]), dim=1)
        new_b = torch.cat((bb[:, :k], bb[:, k:] + aa[:, k:] * bb[:, :-k]), dim=1)
        aa, bb = new_a, new_b
        k *= 2
    return bb


class ScanBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drive = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * expand), nn.SiLU(),
                                nn.Linear(dim * expand, dim))
        # Per-channel retain bias initially encourages several time scales.
        retain = torch.linspace(0.5, 4.5, dim)
        self.retain_bias = nn.Parameter(retain)
        nn.init.constant_(self.gate.bias, 0.0)

    def _ab(self, x):
        q = self.norm(x)
        a = torch.sigmoid(self.gate(q) + self.retain_bias)
        b = (1.0 - a) * torch.tanh(self.drive(q))
        return a, b

    def forward(self, x):
        a, b = self._ab(x)
        state_seq = parallel_affine_scan(a, b)
        h = x + self.output(state_seq)
        return h + self.ff(self.ff_norm(h))

    def step(self, x, previous_state):
        """x: [B,D], previous_state: [B,D], returns output and new state."""
        a, b = self._ab(x)
        state = a * previous_state + b
        h = x + self.output(state)
        return h + self.ff(self.ff_norm(h)), state


class PCMNCore(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 4, expand: int = 2):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([ScanBlock(dim, expand) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x [batch,time,dim]."""
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)

    def init_states(self, batch: int, device, dtype):
        return [torch.zeros(batch, self.dim, device=device, dtype=dtype)
                for _ in self.layers]

    def step(self, x, states):
        """x [batch,dim]."""
        if len(states) != len(self.layers):
            raise ValueError('Incorrect number of recurrent states')
        next_states = []
        for layer, previous in zip(self.layers, states):
            x, state = layer.step(x, previous)
            next_states.append(state)
        return self.final_norm(x), next_states


class PCMNLanguageModel(nn.Module):
    def __init__(self, vocab_size: int = 256, dim: int = 128,
                 layers: int = 4, expand: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, dim)
        self.core = PCMNCore(dim=dim, layers=layers, expand=expand)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, tokens):
        return self.head(self.core(self.embed(tokens)))

    def loss(self, tokens):
        if tokens.size(1) < 2:
            raise ValueError('At least two tokens are needed')
        logits = self.forward(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, self.vocab_size),
                               tokens[:, 1:].reshape(-1))

    @torch.no_grad()
    def prefill(self, tokens):
        """Return logits and states of the last prompt token, recurrence exact.

        This reference prefill loops over tokens to calculate the states.
        For high throughput use a compiled/chunked scan that exposes prefixes.
        """
        states = self.core.init_states(tokens.size(0), tokens.device, self.embed.weight.dtype)
        for i in range(tokens.size(1)):
            y, states = self.core.step(self.embed(tokens[:, i]), states)
        return self.head(y), states

    @torch.no_grad()
    def sample(self, prompt, new_tokens=40, temperature=1.0):
        if temperature <= 0:
            raise ValueError('temperature must be positive')
        logits, states = self.prefill(prompt)
        output = [prompt]
        for _ in range(new_tokens):
            probabilities = torch.softmax(logits / temperature, dim=-1)
            tok = torch.multinomial(probabilities, num_samples=1)
            output.append(tok)
            y, states = self.core.step(self.embed(tok[:, 0]), states)
            logits = self.head(y)
        return torch.cat(output, dim=1)
