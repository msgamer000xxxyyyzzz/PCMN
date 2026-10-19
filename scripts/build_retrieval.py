"""Build offline BM25 source-attributed memory from JSONL documents."""
import argparse,json
from pathlib import Path
from pcmn.retrieval import RetrievalIndex

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--chunk-words',type=int,default=180)
    p.add_argument('--overlap',type=int,default=40)
    p.add_argument('--max-docs',type=int,default=100000)
    a=p.parse_args()
    index=RetrievalIndex(max_docs=a.max_docs)
    count=0
    with a.input.open(encoding='utf-8') as fd:
        for no,line in enumerate(fd,1):
            if not line.strip():continue
            doc=json.loads(line)
            if not isinstance(doc,dict) or not isinstance(doc.get('text'),str):
                raise ValueError(f'Invalid document at line {no}')
            index.add_document(doc['text'],str(doc.get('source',f'line:{no}')),
                               a.chunk_words,a.overlap)
            count+=1
    index.save(a.out)
    print(json.dumps({'documents':count,'passages':len(index.docs),'path':str(a.out)}))
if __name__=='__main__':main()
