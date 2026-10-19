"""Local per-model timing with proper device sync, NOT a rigorously matched evaluation."""
import argparse
import time
import torch
from torch import nn
from pcmn import PCMNLanguageModel


class BaselineTransformer(nn.Module):
    def __init__(self, vocab, dim, layers, heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(8192, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, dropout=0)
        self.core = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, tokens):
        t = tokens.size(1)
        x = self.embed(tokens) + self.pos(torch.arange(t, device=tokens.device))[None]
        # Causal mask for a fair next-token information constraint.
        mask = torch.triu(torch.ones(t,t,device=tokens.device, dtype=torch.bool), diagonal=1)
        return self.head(self.core(x, mask=mask))


def measure(model, data, loops, device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    with torch.inference_mode():
        for _ in range(3):
            model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(loops):
            model(data)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    return (time.perf_counter()-t0)/loops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--length', type=int, default=64)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--loops', type=int, default=5)
    args = parser.parse_args()
    device_name = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(device_name)
    data = torch.randint(0, 128, (args.batch, args.length), device=device)
    for label, model in [('PCMN scan (reference)', PCMNLanguageModel(128, args.dim, args.layers)),
                         ('PyTorch Transformer', BaselineTransformer(128, args.dim, args.layers))]:
        model = model.to(device).eval()
        seconds = measure(model, data, args.loops, device)
        parameters = sum(p.numel() for p in model.parameters())
        print(f'{label}: {parameters:,} parameters; '
              f'{seconds*1000:.2f} ms/batch; '
              f'{data.numel()/seconds:,.0f} input tokens/second')
    print('Caution: models are not matched for parameter count, accuracy or optimized kernels.')


if __name__ == '__main__':
    main()
