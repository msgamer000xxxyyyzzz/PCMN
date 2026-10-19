"""Evaluate ALL eligible tokens in the held-out corpus without random sampling."""
import argparse,json
from pathlib import Path
import torch
from pcmn.text_core import PCMNTextLM,TextConfig
from pcmn.stream_data import mmap_tokens,mmap_doc_offsets
from pcmn.quality_eval import evaluate_corpus,checkpoint_metadata


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--corpus',required=True,type=Path)
    p.add_argument('--context',type=int,default=128)
    p.add_argument('--max-documents',type=int)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.threads<1:raise ValueError('threads must be positive')
    torch.set_num_threads(a.threads)
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False) # ONLY trusted local checkpoints
    corpus_info=json.loads((a.corpus/'corpus.json').read_text(encoding='utf-8'))
    tokenizer_info=json.loads((a.corpus/'tokenizer.json').read_text(encoding='utf-8'))
    if ck['tokenizer'] != tokenizer_info:
        raise SystemExit('Mismatch: model and validation corpus tokenizers differ')
    model=PCMNTextLM(TextConfig(**ck['model_config'])).to(a.device)
    model.load_state_dict(ck['model'],strict=True)
    result={'checkpoint':str(a.checkpoint),'corpus':str(a.corpus),
            'checkpoint_metadata':checkpoint_metadata(ck),'corpus_metadata':corpus_info}
    result.update(evaluate_corpus(model,mmap_tokens(a.corpus/'valid.bin'),
                                  mmap_doc_offsets(a.corpus/'valid.idx'),a.context,
                                  a.max_documents,a.device))
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
