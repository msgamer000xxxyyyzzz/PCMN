import argparse,json
from pathlib import Path
from pcmn.task_suite import make_items,write_chat

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--out',type=Path,default=Path('data/reasoning'))
 p.add_argument('--train-size',type=int,default=1000)
 p.add_argument('--valid-size',type=int,default=100)
 p.add_argument('--seed',type=int,default=42)
 a=p.parse_args()
 if min(a.train_size,a.valid_size)<1:raise ValueError('Sizes must be positive')
 a.out.mkdir(parents=True,exist_ok=True)
 write_chat(make_items(a.train_size,a.seed),a.out/'train.jsonl')
 write_chat(make_items(a.valid_size,a.seed+1024),a.out/'valid.jsonl')
 print(json.dumps({'train':a.train_size,'valid':a.valid_size,'note':'Synthetic examples only'}))
if __name__=='__main__':main()
