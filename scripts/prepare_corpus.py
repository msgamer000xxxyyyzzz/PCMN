"""Create mmap-friendly token streams from independent UTF-8 text files / JSONL.

Input train and valid paths must be intentionally disjoint. Documents are not
mixed across these splits; duplicate SHA256 document hashes are rejected.
JSONL rows have {"text":"..."}. This is not a web crawler or license checker.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from pcmn.text_tokenizer import ByteTokenizer, train_bpe


def documents(paths):
    for path in paths:
        path=Path(path)
        if path.suffix.lower()=='.jsonl':
            with path.open(encoding='utf-8-sig') as fd:
                for lineno,line in enumerate(fd,1):
                    if not line.strip():continue
                    value=json.loads(line)
                    if not isinstance(value,dict) or not isinstance(value.get('text'),str):
                        raise ValueError(f'Invalid JSONL text at {path}:{lineno}')
                    yield value['text']
        else:
            # Bound preprocessing memory for very large source text files.
            # Segments of ~1 MiB are treated as independent documents.
            parts=[]
            size=0
            with path.open(encoding='utf-8-sig') as fd:
                for line in fd:
                    if len(line)>1_048_576:
                        # Extremely long single lines must also be streamed.
                        lines=[line[i:i+1_048_576] for i in range(0,len(line),1_048_576)]
                    else:
                        lines=[line]
                    for segment in lines:
                        parts.append(segment)
                        size+=len(segment)
                        if size>=1_048_576:
                            yield ''.join(parts)
                            parts=[]
                            size=0
            if parts:yield ''.join(parts)


def create(train, valid, out, tokenizer, vocab_size=16000):
    train=[Path(x).resolve() for x in train]
    valid=[Path(x).resolve() for x in valid]
    if not train or not valid or set(train)&set(valid):
        raise ValueError('Must provide different, nonempty train and valid file paths')
    out=Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer=='bpe':
        # JSONL and TXT can be mixed; train on document text only, not metadata.
        # Generator keeps training text out of Python lists for large corpora.
        tok=train_bpe(train,vocab_size=vocab_size,text_iterator=documents(train))
    else:
        tok=ByteTokenizer()
    seen={}
    counts={}
    for split, paths in [('train',train),('valid',valid)]:
        n=0
        ndocs=0
        with (out/f'{split}.bin').open('wb') as out_file, (out/f'{split}.idx').open('wb') as idx_file:
            idx_file.write(struct.pack('<Q',0))
            for doc in documents(paths):
                if not doc.strip(): continue
                digest=hashlib.sha256(doc.encode('utf-8')).hexdigest()
                if digest in seen:
                    if seen[digest]!=split:
                        raise ValueError('Exact duplicate document across train / validation')
                    continue
                seen[digest]=split
                ids=tok.encode(doc)+[tok.eos_id]
                for start in range(0,len(ids),16384):
                    seg=ids[start:start+16384]
                    out_file.write(struct.pack('<'+str(len(seg))+'I',*seg))
                n+=len(ids)
                ndocs+=1
                idx_file.write(struct.pack('<Q',n))
        counts[split]={'tokens':n,'documents':ndocs}
        if n < 8:
            raise ValueError(f'Empty corpus split: {split}')
    (out/'tokenizer.json').write_text(json.dumps(tok.serialize(),ensure_ascii=False),encoding='utf-8')
    info={'format':'PCMN-u32le-v1','tokenizer':tok.kind,'vocab_size':tok.vocab_size,
          'train_paths':[str(x) for x in train],'valid_paths':[str(x) for x in valid],
          **counts}
    (out/'corpus.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return info


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',nargs='+',required=True)
    p.add_argument('--valid',nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--tokenizer',choices=['byte','bpe'],default='byte')
    p.add_argument('--vocab-size',type=int,default=16000)
    args=p.parse_args()
    print(json.dumps(create(**vars(args)),ensure_ascii=False))

if __name__=='__main__':main()
