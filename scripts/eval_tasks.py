"""Held-out deterministic synthetic evaluation. No claims about general intelligence."""
import argparse,json,re,random,math
from pathlib import Path
from collections import defaultdict
from pcmn.task_suite import make_items,check_code
from pcmn.serving import TextEngine

def run(engine,n,seed,gen_tokens):
    rows=[]
    for item in make_items(n,seed):
        prediction=engine.complete(item['question']+'\nОтвет:',max_tokens=gen_tokens,greedy=True)['text'].strip()
        if item['category']=='code_generation':
            name=re.search(r'def (\w+)\(',item['answer']).group(1)
            add=int(re.search(r'\+ (\d+)',item['answer']).group(1))
            correct=check_code(prediction,name,17,17+add)
        else:
            match=re.match(r'\s*(-?\d+)\b',prediction)
            correct=bool(match and match.group(1)==item['answer'])
        rows.append({'category':item['category'],'correct':correct,'target':item['answer'],
                    'prediction':prediction[:200]})
    by=defaultdict(list)
    for row in rows:by[row['category']].append(row['correct'])
    return {'count':n,'accuracy':sum(r['correct'] for r in rows)/n,
            'category_accuracy':{k:sum(v)/len(v) for k,v in by.items()},'rows':rows,
            'note':'Small generated exact-match tasks; not a coding benchmark like HumanEval'}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--count',type=int,default=20)
    p.add_argument('--seed',type=int,default=999)
    p.add_argument('--tokens',type=int,default=64)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    if not 1<=a.count<=1000:raise ValueError('Invalid count')
    import torch
    torch.set_num_threads(2)
    result=run(TextEngine(a.checkpoint),a.count,a.seed,a.tokens)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
