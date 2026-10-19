"""License-aware, provenance-preserving deterministic corpus preparation.

Input JSONL documents:
{"text":"...","source":"...","license":"CC-BY-4.0","category":"code"}
Never auto-accept unknown licenses. This filters EXACT duplicates, not paraphrases,
and does not certify copyright, consent, privacy or training rights.
"""
import argparse,hashlib,json,re
from collections import Counter
from pathlib import Path

EMAIL=re.compile(r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])')

def curate(paths,out,allowed,valid_percent=5,minimum=40,maximum=1_000_000,redact_email=True,split_unit="document"):
    if not 0<valid_percent<50:raise ValueError('valid_percent must be between 0 and 50')
    if not allowed:raise ValueError('Explicit allowed license list required')
    if split_unit not in {'document','source'}:raise ValueError('split_unit must be document or source')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    seen=set();stats=Counter();cats=Counter(); sources=set()
    files=[Path(p) for p in paths]
    # Avoid loading whole corpus into RAM; hashed provenance and dedup signatures
    # are kept in memory for now, and must be sharded for huge datasets.
    with (out/'train.jsonl').open('w',encoding='utf-8') as train, (out/'valid.jsonl').open('w',encoding='utf-8') as valid:
        for file in files:
            with file.open(encoding='utf-8') as fd:
                for i,line in enumerate(fd,1):
                    if not line.strip():continue
                    stats['seen']+=1
                    doc=json.loads(line)
                    if not isinstance(doc,dict) or not all(isinstance(doc.get(k),str) for k in ('text','source','license')):
                        stats['bad_metadata']+=1;continue
                    if doc['license'] not in allowed:
                        stats['rejected_license']+=1;continue
                    text=doc['text']
                    if redact_email:text=EMAIL.sub('[EMAIL]',text)
                    if not minimum<=len(text)<=maximum:
                        stats['bad_length']+=1;continue
                    digest=hashlib.sha256(text.strip().encode('utf-8')).hexdigest()
                    if digest in seen:
                        stats['duplicate']+=1;continue
                    seen.add(digest)
                    # Deterministic split prevents exact duplicate split leakage.
                    split_hash=(digest if split_unit=='document' else hashlib.sha256(doc['source'].encode('utf-8')).hexdigest())
                    split=int(split_hash[:8],16)%10000 < valid_percent*100
                    target=valid if split else train
                    payload={'text':text,'source':doc['source'],'license':doc['license'],
                             'category':doc.get('category','general'),'sha256':digest}
                    target.write(json.dumps(payload,ensure_ascii=False)+'\n')
                    stats['valid' if split else 'train']+=1
                    cats[payload['category']]+=1;sources.add(doc['source'])
    report={'counts':dict(stats),'categories':dict(cats),'distinct_sources':len(sources),
            'allowed_licenses':sorted(allowed),'seed_policy':f'sha256({split_unit}) deterministic', 'split_unit':split_unit,
            'limitations':['exact-duplicate-only','licenses_not_legally_verified','email_redaction_not_full_PII']}
    (out/'curation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if stats['train']<1 or stats['valid']<1:
        raise ValueError('Not enough accepted data in both splits; adjust input size')
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',nargs='+',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--allow-license',action='append',required=True)
    p.add_argument('--valid-percent',type=int,default=5)
    p.add_argument('--split-unit',choices=['document','source'],default='document')
    a=p.parse_args()
    print(json.dumps(curate(a.input,a.out,set(a.allow_license),a.valid_percent,split_unit=a.split_unit),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
