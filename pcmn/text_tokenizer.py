"""Deterministic byte tokenizer and optional trainable ByteLevel BPE tokenizer."""
from __future__ import annotations
import json
from pathlib import Path


class ByteTokenizer:
    vocab_size = 259
    bos_id = 1
    eos_id = 2
    kind = 'byte'

    def encode(self, text: str):
        return [int(c) + 3 for c in text.encode('utf-8')]

    def decode(self, ids, skip_special=True):
        bb = []
        for token in ids:
            i = int(token)
            if 3 <= i <= 258:
                bb.append(i - 3)
            elif not skip_special:
                bb.extend(f'<{i}>'.encode())
        return bytes(bb).decode('utf-8', errors='replace')

    def serialize(self):
        return {'type': self.kind}


class BPETokenizer:
    kind = 'bpe'
    def __init__(self, hf_tokenizer):
        self.tk = hf_tokenizer
        self.vocab_size = self.tk.get_vocab_size()
        self.bos_id = self.tk.token_to_id('[BOS]')
        self.eos_id = self.tk.token_to_id('[EOS]')

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special=True):
        return self.tk.decode([int(x) for x in ids], skip_special_tokens=skip_special)

    def serialize(self):
        return {'type':'bpe','tokenizer_json':self.tk.to_str()}


def train_bpe(paths, vocab_size=16000, min_frequency=2, text_iterator=None):
    try:
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, normalizers
    except ImportError as exc:
        raise SystemExit('BPE requires: pip install tokenizers') from exc
    tk = Tokenizer(models.BPE(unk_token='[UNK]'))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_frequency,
                  special_tokens=['[PAD]', '[BOS]', '[EOS]', '[UNK]'],
                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    if text_iterator is None:
        tk.train([str(p) for p in paths], trainer)
    else:
        # Streaming iterator: only text is used, never JSONL metadata or keys.
        tk.train_from_iterator(text_iterator, trainer=trainer)
    return BPETokenizer(tk)


def load_tokenizer(data):
    if data['type'] == 'byte':
        return ByteTokenizer()
    if data['type'] == 'bpe':
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise SystemExit('Install tokenizers to use BPE checkpoints') from exc
        return BPETokenizer(Tokenizer.from_str(data['tokenizer_json']))
    raise ValueError('Unknown tokenizer format')
