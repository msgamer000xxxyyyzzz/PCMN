"""Training/validation contamination audit on curated JSONL files."""
import argparse,json
from pathlib import Path
from pcmn.corpus_audit import audit

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--train',type=Path,required=True)
    p.add_argument('--valid',type=Path,required=True)
    p.add_argument('--sample',type=int,default=200)
    p.add_argument('--threshold',type=float,default=0.85)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    result=audit(a.train,a.valid,a.sample,a.threshold)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
