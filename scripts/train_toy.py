"""Tiny diagnostic training, NOT meaningful language quality or speed evidence."""
import argparse
import sys
import torch
from pcmn import PCMNLanguageModel


def batch_pattern(batch, time, vocab, device):
    starts = torch.randint(0, vocab, (batch, 1), device=device)
    shift = torch.arange(time, device=device).unsqueeze(0)
    return (starts + shift) % vocab


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=80)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--length', type=int, default=48)
    p.add_argument('--dim', type=int, default=64)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--device', default='auto')
    args = p.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.manual_seed(123)
    vocab = 32
    model = PCMNLanguageModel(vocab, args.dim, args.layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    model.train()
    for step in range(1, args.steps + 1):
        tokens = batch_pattern(args.batch, args.length, vocab, device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(tokens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 1 or step % max(1, args.steps//5) == 0:
            print(f'step {step:4} / {args.steps}, next-token loss={loss.item():.4f}')
    model.eval()
    print('Example sampled IDs:', model.sample(batch_pattern(1, 5, vocab, device), 10)[0].tolist())


if __name__ == '__main__':
    main()
