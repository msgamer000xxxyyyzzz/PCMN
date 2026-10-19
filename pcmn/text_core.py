"""PCMN v0.3: experimental Transformer-free language backbone.

Causal local depthwise convolution + diagonal input-selective recurrent scan
+ gated residual feed-forward. All forward operations strictly causal.
Reference scan is O(T log T) in work/memory; NOT a fused optimized kernel.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .model import parallel_affine_scan
from .fast_scan import cpu_scan, cuda_scan


@dataclass
class TextConfig:
    vocab_size: int = 259
    dim: int = 64
    layers: int = 2
    ff_mult: int = 3
    conv_kernel: int = 4
    dropout: float = 0.0
    grad_checkpoint: bool = False
    scan_chunk: int = 128
    memory_slots: int = 0  # 0 keeps v0.2 checkpoint compatibility; >=1 enables multi-timescale memory
    memory_base_half_life: int = 32
    addressable_slots: int = 0  # v0.4 opt-in dynamic key/value ring memory

    def __post_init__(self):
        if self.vocab_size < 4 or self.dim < 8 or self.layers < 1:
            raise ValueError('Invalid vocab, dim or layer count')
        if self.ff_mult < 1 or self.conv_kernel < 1 or self.scan_chunk < 1 or self.memory_slots < 0 or self.addressable_slots < 0 or self.memory_base_half_life < 2 or not 0 <= self.dropout < 1:
            raise ValueError('Invalid feedforward, convolution or dropout configuration')


def chunked_affine_scan(a, b, chunk_size, initial=None):
    """Exact differentiable scan in sequential chunks, parallel within each chunk.

    Saves intermediate allocations vs log2(total_length) full-tensor doubling,
    but sequential dependence between chunks limits training parallelism.
    """
    if a.size(1) == 0:
        return b
    native = cpu_scan(a, b, initial)
    if native is None:
        native = cuda_scan(a, b, initial)
    if native is not None:
        return native
    if a.size(1) <= chunk_size:
        result = parallel_affine_scan(a, b)
        return result if initial is None else result + torch.cumprod(a, dim=1) * initial.unsqueeze(1)
    carry = torch.zeros_like(b[:, 0]) if initial is None else initial
    results = []
    for start in range(0, a.size(1), chunk_size):
        aa = a[:, start:start+chunk_size]
        bb = b[:, start:start+chunk_size]
        part = parallel_affine_scan(aa, bb)
        # Prefix product propagates state entering this chunk.
        part = part + torch.cumprod(aa, dim=1) * carry.unsqueeze(1)
        carry = part[:, -1]
        results.append(part)
    return torch.cat(results, dim=1)


class TextBlock(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        d = cfg.dim
        self.conv_kernel = cfg.conv_kernel
        self.scan_chunk = cfg.scan_chunk
        self.memory_slots = cfg.memory_slots
        self.addressable_slots = cfg.addressable_slots
        if self.addressable_slots:
            self.addr_key = nn.Linear(d, d, bias=False)
            self.addr_query = nn.Linear(d, d, bias=False)
            self.addr_value = nn.Linear(d, d, bias=False)
            self.addr_out = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.addr_out.weight)
        if cfg.memory_slots:
            # Long, fixed exponentially decaying timescales. This is a compressed
            # memory, NOT lossless recall or proof of arbitrary context comprehension.
            lives = cfg.memory_base_half_life * 4.0 ** torch.arange(cfg.memory_slots)
            self.register_buffer('memory_decays', torch.exp(-math.log(2) / lives))
            self.memory_write = nn.Linear(d, d, bias=False)
            self.memory_select = nn.Linear(d, cfg.memory_slots)
            self.memory_out = nn.Linear(d, d, bias=False)
            nn.init.zeros_(self.memory_out.weight)
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, cfg.conv_kernel, groups=d, bias=True)
        self.drive = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.read_gate = nn.Linear(d, d)
        self.read_proj = nn.Linear(d, d, bias=False)
        self.ff_norm = nn.LayerNorm(d)
        inner = d * cfg.ff_mult
        self.ff_a = nn.Linear(d, inner, bias=False)
        self.ff_b = nn.Linear(d, inner, bias=False)
        self.ff_out = nn.Linear(inner, d, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.retain_bias = nn.Parameter(torch.linspace(-0.5, 4.5, d))
        nn.init.normal_(self.conv.weight, std=0.06)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.gate.bias)
        nn.init.constant_(self.read_gate.bias, -1.0)

    def _gates(self, normalized, local):
        mixed = normalized + F.silu(local)
        retain = torch.sigmoid(self.gate(mixed) + self.retain_bias)
        add = (1 - retain) * torch.tanh(self.drive(mixed))
        read = torch.sigmoid(self.read_gate(mixed))
        return retain, add, read

    def _memory_sequence(self, normalized, initial=None):
        """Exact parallel update for multiple exponential memory scales."""
        batch, length, dim = normalized.shape
        slots = self.memory_slots
        decays = self.memory_decays.float().view(1, 1, slots, 1)
        write = torch.tanh(self.memory_write(normalized)).float()
        a = decays.expand(batch, length, slots, dim).reshape(batch, length, slots*dim)
        b = ((1.0 - decays) * write.unsqueeze(2)).reshape(batch, length, slots*dim)
        init = None if initial is None else initial.reshape(batch, slots*dim)
        states = chunked_affine_scan(a, b, self.scan_chunk, init).reshape(batch,length,slots,dim)
        weights = torch.softmax(self.memory_select(normalized).float(), dim=-1)
        selected = (states * weights.unsqueeze(-1)).sum(dim=2)
        return selected, states[:, -1]

    def _address_step(self, q, past):
        """Dynamic key/value memory. Read before overwriting oldest ring entry.

        O(slots * dimension) per token; storage bounded regardless of sequence length.
        Attention is limited to the memory bank, not quadratic sequence attention.
        """
        keys, vals, cursor = past
        batch, slots, dim = keys.shape
        query = F.normalize(self.addr_query(q).float(), dim=-1)
        similarity = (keys * query.unsqueeze(1)).sum(-1) * 6.0
        # Mask unfilled slots; avoid softmax(all -inf) for an empty cache.
        valid = torch.arange(slots, device=keys.device).view(1, slots) < cursor.view(-1, 1)
        similarity = similarity.masked_fill(~valid, -1e4)
        probs = similarity.softmax(dim=1)
        read = (probs.unsqueeze(-1) * vals).sum(dim=1)
        fresh_key = F.normalize(self.addr_key(q).float(), dim=-1)
        fresh_val = torch.tanh(self.addr_value(q).float())
        mask = F.one_hot(cursor.remainder(slots), num_classes=slots).to(keys.dtype).unsqueeze(-1)
        new_keys = keys * (1.0 - mask) + fresh_key.unsqueeze(1) * mask
        new_vals = vals * (1.0 - mask) + fresh_val.unsqueeze(1) * mask
        return read, (new_keys, new_vals, cursor + 1)

    def _address_sequence(self, normalized, old):
        reads = []
        for t in range(normalized.size(1)):
            read, old = self._address_step(normalized[:, t, :], old)
            reads.append(read)
        return torch.stack(reads, dim=1), old

    def _output(self, x, state, read, memory_read=None, addr_read=None):
        h = x + self.dropout(self.read_proj(state.to(x.dtype)) * read)
        if memory_read is not None:
            h = h + self.dropout(self.memory_out(memory_read.to(x.dtype)))
        if addr_read is not None:
            h = h + self.dropout(self.addr_out(addr_read.to(x.dtype)))
        q = self.ff_norm(h)
        return h + self.dropout(self.ff_out(F.silu(self.ff_a(q)) * self.ff_b(q)))

    def _sequence(self, x, with_state=False, old=None):
        normalized = self.norm(x)
        history_length = self.conv_kernel - 1
        if old is None:
            history = normalized.new_zeros(normalized.size(0), history_length, normalized.size(-1))
            old_state = None
            old_memory = None
            old_address = None
        else:
            old_state, history = old[:2]
            old_memory = old[2] if self.memory_slots else None
            old_address = old[2 + int(bool(self.memory_slots)):] if self.addressable_slots else None
        if self.addressable_slots and old_address is None:
            slots = self.addressable_slots
            old_address = (normalized.new_zeros(normalized.size(0), slots, normalized.size(-1)).float(),
                           normalized.new_zeros(normalized.size(0), slots, normalized.size(-1)).float(),
                           torch.zeros(normalized.size(0), dtype=torch.long, device=normalized.device))
        conv_input = torch.cat((history, normalized), dim=1)
        local = self.conv(conv_input.transpose(1,2)).transpose(1,2)
        a, b, r = self._gates(normalized, local)
        state = chunked_affine_scan(a.float(), b.float(), self.scan_chunk, old_state)
        memory_read = None
        memory_state = None
        if self.memory_slots:
            memory_read, memory_state = self._memory_sequence(normalized, old_memory)
        address_read, address_state = (self._address_sequence(normalized, old_address)
                                       if self.addressable_slots else (None, None))
        output = self._output(x, state, r, memory_read, address_read)
        if not with_state:
            return output
        next_history = conv_input[:, -history_length:, :] if history_length else normalized[:, :0, :]
        next_state = (state[:, -1], next_history)
        if self.memory_slots:
            next_state += (memory_state,)
        if self.addressable_slots:
            next_state += address_state
        return output, next_state

    def forward(self, x):
        return self._sequence(x)

    def forward_with_state(self, x, old=None):
        return self._sequence(x, with_state=True, old=old)

    def step(self, x, old):
        """x [B,D]; old state length 2 (v0.2) or 3 (with slow memory)."""
        state, history = old[:2]
        q = self.norm(x)
        window = torch.cat([history, q.unsqueeze(1)], dim=1)
        local = self.conv(window.transpose(1, 2)).squeeze(-1)
        a, b, r = self._gates(q, local)
        state = a.float() * state + b.float()
        next_history = window[:, 1:, :]
        slow_read = None
        slow_state = None
        if self.memory_slots:
            decay = self.memory_decays.float().view(1,-1,1)
            slow_state = decay * old[2] + (1-decay) * torch.tanh(self.memory_write(q)).float().unsqueeze(1)
            probs = torch.softmax(self.memory_select(q).float(), dim=-1)
            slow_read = (slow_state * probs.unsqueeze(-1)).sum(1)
        cache = (state, next_history) if not self.memory_slots else (state, next_history, slow_state)
        addr_read = None
        if self.addressable_slots:
            offset = 2 + int(bool(self.memory_slots))
            addr_read, nxt = self._address_step(q, old[offset:offset+3])
            cache += nxt
        return self._output(x, state, r, slow_read, addr_read), cache


class PCMNTextLM(nn.Module):
    def __init__(self, cfg: TextConfig):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([TextBlock(cfg) for _ in range(cfg.layers)])
        self.final_norm = nn.LayerNorm(cfg.dim)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        nn.init.normal_(self.embedding.weight, mean=0, std=cfg.dim**-0.5)

    def forward(self, tokens):
        if tokens.ndim != 2:
            raise ValueError('Expected token ids [batch,time]')
        x = self.embedding(tokens)
        for block in self.blocks:
            if self.config.grad_checkpoint and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.lm_head(self.final_norm(x))

    def loss(self, tokens):
        if tokens.size(1) < 2:
            raise ValueError('At least two tokens required')
        logits = self(tokens[:, :-1])
        return F.cross_entropy(logits.float().reshape(-1, self.config.vocab_size),
                               tokens[:, 1:].reshape(-1))

    def init_states(self, batch: int, device=None):
        device = device if device is not None else self.embedding.weight.device
        dtype = self.embedding.weight.dtype
        states = []
        for block in self.blocks:
            state = (torch.zeros(batch, self.config.dim, device=device, dtype=torch.float32),
                     torch.zeros(batch, self.config.conv_kernel-1, self.config.dim, device=device, dtype=dtype))
            if block.memory_slots:
                state += (torch.zeros(batch, block.memory_slots, self.config.dim, device=device, dtype=torch.float32),)
            if block.addressable_slots:
                k = torch.zeros(batch, block.addressable_slots, self.config.dim, device=device, dtype=torch.float32)
                v = torch.zeros_like(k)
                ptr = torch.zeros(batch, device=device, dtype=torch.long)
                state += (k, v, ptr)
            states.append(state)
        return states

    def forward_chunk(self, tokens, states=None):
        """State-carrying segment forward for truncated-BPTT or long prefill.

        Gradients flow through input and returned states; call detach_states()
        deliberately at sequence-boundaries to limit training memory.
        """
        if tokens.ndim != 2 or tokens.size(1) < 1:
            raise ValueError('Expected nonempty tokens [batch,time]')
        if states is None:
            states = self.init_states(tokens.shape[0], tokens.device)
        if len(states) != len(self.blocks):
            raise ValueError('Expected one state per block')
        x = self.embedding(tokens)
        next_states = []
        for block, old in zip(self.blocks,states):
            x, state = block.forward_with_state(x, old)
            next_states.append(state)
        return self.lm_head(self.final_norm(x)), next_states

    @staticmethod
    def detach_states(states):
        return [tuple(v.detach() for v in layer) for layer in states]

    def step(self, token, states):
        if token.ndim != 1 or len(states) != len(self.blocks):
            raise ValueError('Expected tokens [batch] and one state per block')
        x = self.embedding(token)
        next_states = []
        for block, old in zip(self.blocks, states):
            x, new_state = block.step(x, old)
            next_states.append(new_state)
        return self.lm_head(self.final_norm(x)), next_states

    @torch.no_grad()
    def prefill(self, prompt, chunk_size=None):
        """Prefill, optionally bounded in token chunks to prevent memory spikes.

        With chunk_size=None uses the original parallel full prompt algorithm.
        CPU chunked mode reduces peak activations but adds chunk boundaries.
        """
        if prompt.ndim != 2 or prompt.size(1) == 0:
            raise ValueError('Expected nonempty prompt [batch, tokens]')
        if chunk_size is not None:
            if chunk_size < 1:
                raise ValueError('chunk_size must be >=1')
            states = self.init_states(prompt.size(0), prompt.device)
            for start in range(0, prompt.size(1), chunk_size):
                logits, states = self.forward_chunk(prompt[:,start:start+chunk_size], states)
            return logits[:, -1], states
        x = self.embedding(prompt)
        states = []
        for block in self.blocks:
            x, state = block.forward_with_state(x)
            states.append(state)
        return self.lm_head(self.final_norm(x[:, -1])), states

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=0.8, top_k=40,
                 eos_id=None, greedy=False, prefill_chunk=None):
        if temperature <= 0 or max_new_tokens < 0 or top_k < 0:
            raise ValueError('Invalid sampling options')
        was_training = self.training
        self.eval()
        try:
            logits, states = self.prefill(prompt, chunk_size=prefill_chunk)
            generated = []
            done = torch.zeros(prompt.size(0), device=prompt.device, dtype=torch.bool)
            for _ in range(max_new_tokens):
                if greedy:
                    tok = logits.argmax(dim=-1)
                else:
                    scaled = logits / temperature
                    if top_k:
                        threshold = torch.topk(scaled, min(top_k, scaled.size(-1)), dim=-1).values[:, -1:]
                        scaled = scaled.masked_fill(scaled < threshold, -torch.inf)
                    tok = torch.multinomial(torch.softmax(scaled, dim=-1), 1).squeeze(-1)
                if eos_id is not None:
                    tok = torch.where(done, torch.full_like(tok, eos_id), tok)
                    done |= (tok == eos_id)
                generated.append(tok)
                if bool(done.all()) and eos_id is not None:
                    break
                logits, states = self.step(tok, states)
            return torch.cat([prompt, torch.stack(generated, 1)], dim=1) if generated else prompt
        finally:
            self.train(was_training)

    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())


PROFILES = {
    'smoke': dict(dim=64, layers=2, ff_mult=2),
    'mini': dict(dim=192, layers=4, ff_mult=3),
    '30m': dict(dim=448, layers=10, ff_mult=3),
    '100m': dict(dim=704, layers=16, ff_mult=3),
}


def build_config(profile='smoke', vocab_size=259, **overrides):
    if profile not in PROFILES:
        raise ValueError(f'Unknown profile {profile!r}; choose {list(PROFILES)}')
    vals = dict(PROFILES[profile], vocab_size=vocab_size)
    vals.update({k: v for k, v in overrides.items() if v is not None})
    return TextConfig(**vals)
