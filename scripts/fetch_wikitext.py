"""Download WikiText-2 / 103 into separate train and validation UTF-8 files.

Requires 'datasets' package and internet; data is not redistributed in the archive.
Source: https://huggingface.co/datasets/Salesforce/wikitext
Observe dataset license and attribution when redistributing derived datasets.
"""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--size',choices=['2','103'],default='2')
    parser.add_argument('--out',default='data/wikitext')
    args=parser.parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit('pip install datasets') from exc
    data=load_dataset('Salesforce/wikitext',f'wikitext-{args.size}-raw-v1')
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    for name in ('train','validation','test'):
        target=out/f'{name}.txt'
        with target.open('w',encoding='utf-8') as fh:
            for item in data[name]:
                if item['text'].strip():
                    fh.write(item['text']+'\n')
        print(name,target,target.stat().st_size,'bytes')


if __name__ == '__main__':
    main()
