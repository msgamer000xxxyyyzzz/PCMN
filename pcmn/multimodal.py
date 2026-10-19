"""Multimodal interface: runnable *untrained* 4-modality codecs and PCMN backbone.

All generated output arrays are raw, random predictions until the codecs and
joint model are trained on paired text/audio/image/video data. Not a TTS,
vocoder, image diffusion generator, or video foundation model.

Input formats:
  text:  int64 [B,L]
  image: float32 [B,3,H,W], H,W multiples of 8
  audio: float32 [B,1,S], S multiple of 16
  video: float32 [B,3,T,H,W], T multiple of 2, H,W multiples of 8
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import PCMNCore


class ImageCodec(nn.Module):
    def __init__(self, dim=128, patch=8):
        super().__init__()
        self.patch = patch
        self.enc = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.dec = nn.ConvTranspose2d(dim, 3, kernel_size=patch, stride=patch)

    def encode(self, x):
        if x.ndim != 4 or x.size(1) != 3 or x.size(2) % self.patch or x.size(3) % self.patch:
            raise ValueError('Expected [B,3,H,W] with H and W divisible by patch size')
        return self.enc(x).flatten(2).transpose(1, 2)

    def decode(self, tokens, hw):
        h, w = hw
        if h % self.patch or w % self.patch or tokens.size(1) != h * w // self.patch**2:
            raise ValueError('Image tokens / dimensions do not match')
        return self.dec(tokens.transpose(1, 2).reshape(tokens.size(0), -1, h // self.patch, w // self.patch))


class AudioCodec(nn.Module):
    def __init__(self, dim=128, patch=16):
        super().__init__()
        self.patch = patch
        self.enc = nn.Conv1d(1, dim, kernel_size=patch, stride=patch)
        self.dec = nn.ConvTranspose1d(dim, 1, kernel_size=patch, stride=patch)

    def encode(self, x):
        if x.ndim != 3 or x.size(1) != 1 or x.size(2) % self.patch:
            raise ValueError('Expected [B,1,S], S divisible by patch size')
        return self.enc(x).transpose(1, 2)

    def decode(self, tokens):
        return self.dec(tokens.transpose(1, 2))


class VideoCodec(nn.Module):
    def __init__(self, dim=128, patch=(2, 8, 8)):
        super().__init__()
        self.patch = patch
        self.enc = nn.Conv3d(3, dim, kernel_size=patch, stride=patch)
        self.dec = nn.ConvTranspose3d(dim, 3, kernel_size=patch, stride=patch)

    def encode(self, x):
        if x.ndim != 5 or x.size(1) != 3 or any(x.size(i + 2) % self.patch[i] for i in range(3)):
            raise ValueError('Expected [B,3,T,H,W], divisible by video patch sizes')
        return self.enc(x).flatten(2).transpose(1, 2)

    def decode(self, tokens, thw):
        t, h, w = thw
        p0, p1, p2 = self.patch
        if t % p0 or h % p1 or w % p2 or tokens.size(1) != (t // p0) * (h // p1) * (w // p2):
            raise ValueError('Video tokens / dimensions do not match')
        block = tokens.transpose(1, 2).reshape(tokens.size(0), -1, t // p0, h // p1, w // p2)
        return self.dec(block)


class PCMNMultimodal(nn.Module):
    """Training-friendly fusion and modality-specific reconstruction heads.

    forward(data): returns available-output predictions for supplied modalities.
    synthesize(data, modality, shape): basic conditional output-shape demo, *untrained*.
    """
    modality_names = ('text', 'image', 'audio', 'video')

    def __init__(self, vocab_size=256, dim=128, layers=3, max_query_tokens=4096):
        super().__init__()
        self.vocab_size, self.dim = vocab_size, dim
        self.embed = nn.Embedding(vocab_size, dim)
        self.image = ImageCodec(dim)
        self.audio = AudioCodec(dim)
        self.video = VideoCodec(dim)
        self.modality_offsets = nn.Parameter(torch.zeros(4, dim))
        self.core = PCMNCore(dim, layers)
        self.text_head = nn.Linear(dim, vocab_size)
        self.query_pos = nn.Embedding(max_query_tokens, dim)
        self.max_query_tokens = max_query_tokens

    def _tokens(self, modality, data):
        if modality == 'text':
            return self.embed(data)
        if modality == 'image':
            return self.image.encode(data)
        if modality == 'audio':
            return self.audio.encode(data)
        if modality == 'video':
            return self.video.encode(data)
        raise ValueError('Unknown modality')

    def _flatten_inputs(self, data):
        chunks, slices = [], {}
        begin = 0
        for i, name in enumerate(self.modality_names):
            if name not in data or data[name] is None:
                continue
            t = self._tokens(name, data[name]) + self.modality_offsets[i]
            chunks.append(t)
            slices[name] = slice(begin, begin + t.size(1))
            begin += t.size(1)
        if not chunks:
            raise ValueError('Supply at least one modality')
        batch_sizes = {v.size(0) for v in chunks}
        if len(batch_sizes) != 1:
            raise ValueError('All modalities must have the same batch size')
        return torch.cat(chunks, dim=1), slices

    def forward(self, data):
        z, segments = self._flatten_inputs(data)
        out = self.core(z)
        result = {}
        for name, sl in segments.items():
            tokens = out[:, sl]
            if name == 'text':
                result['text_logits'] = self.text_head(tokens)
            elif name == 'image':
                result['image'] = self.image.decode(tokens, data['image'].shape[-2:])
            elif name == 'audio':
                result['audio'] = self.audio.decode(tokens)
            elif name == 'video':
                result['video'] = self.video.decode(tokens, data['video'].shape[-3:])
        return result

    def _prefill(self, data):
        z, _ = self._flatten_inputs(data)
        states = self.core.init_states(z.size(0), z.device, z.dtype)
        for i in range(z.size(1)):
            last, states = self.core.step(z[:, i], states)
        return last, states

    @torch.no_grad()
    def generate_text(self, conditioning, new_tokens=32, temperature=1.0):
        """Sample text tokens conditioned on any supplied input modality.

        Output is random / low quality until training on paired modalities.
        """
        if new_tokens < 1 or temperature <= 0:
            raise ValueError('new_tokens must be >=1, temperature >0')
        last, states = self._prefill(conditioning)
        outputs = []
        for _ in range(new_tokens):
            scores = self.text_head(last) / temperature
            tok = torch.multinomial(torch.softmax(scores, -1), 1)
            outputs.append(tok)
            next_input = self.embed(tok.squeeze(-1)) + self.modality_offsets[0]
            last, states = self.core.step(next_input, states)
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def synthesize(self, conditioning, modality, shape):
        """Return an *untrained* cross-modal prediction of requested shape.

        For image: shape=(H,W), audio: (samples,), video: (T,H,W).
        Shape dimensions must be multiples of codec patch sizes.
        Uses learned query positions and recurrent state from conditioning.
        """
        if modality not in ('image', 'audio', 'video'):
            raise ValueError('Only image/audio/video raw-output synthesis is defined')
        if modality == 'image':
            h, w = shape
            if h % 8 or w % 8:
                raise ValueError('Image H,W must be multiples of 8')
            count = (h // 8) * (w // 8)
        elif modality == 'audio':
            (samples,) = shape
            if samples % 16:
                raise ValueError('Audio sample count must be divisible by 16')
            count = samples // 16
        else:
            t, h, w = shape
            if t % 2 or h % 8 or w % 8:
                raise ValueError('Video shape T,H,W must divide patch sizes 2,8,8')
            count = (t // 2) * (h // 8) * (w // 8)
        if not 1 <= count <= self.max_query_tokens:
            raise ValueError('Too many or too few generated patches')
        last, states = self._prefill(conditioning)
        pos = torch.arange(count, device=last.device)
        # Extend modality-conditioned state with query tokens; no cross-modal output
        # semantics are learned until paired training of the model is performed.
        query = self.query_pos(pos) + self.modality_offsets[self.modality_names.index(modality)]
        outputs = []
        for i in range(count):
            last, states = self.core.step(query[i].unsqueeze(0).expand(last.size(0), -1), states)
            outputs.append(last)
        latent = torch.stack(outputs, dim=1)
        if modality == 'image':
            return self.image.decode(latent, shape)
        if modality == 'audio':
            return self.audio.decode(latent)
        return self.video.decode(latent, shape)
