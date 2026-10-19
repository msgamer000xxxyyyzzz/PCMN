"""Document retrieval regression: factual recall without an LLM or training."""
import argparse,json,random,time
from pathlib import Path
from pcmn.retrieval import RetrievalIndex


def run(lengths=(2048,8192,32768),n=12,seed=49):
    rng=random.Random(seed)
    outputs=[]
    for length in lengths:
        correct=0;latency=[]
        for case in range(n):
            key=f'объект орбита{case}цель{length}'
            answer=f'код{length}{case}пример'
            # distractors share general vocabulary, but not the target entity.
            noise=[f'Объект эксперимент{j} получил параметр {j+11}.' for j in range(length)]
            index=RetrievalIndex()
            pos=rng.randrange(length)
            noise[pos]=f'Для {key} секретный испытательный маркер {answer}.'
            index.add_document(' '.join(noise),source=f'test:{length}:{case}',chunk_words=128,overlap=16)
            query=f'Какой испытательный маркер для {key}?'
            t0=time.perf_counter()
            hits=index.search(query,k=3)
            latency.append(time.perf_counter()-t0)
            correct+=int(any(answer in hit.text for hit in hits))
        outputs.append({'distractor_sentences':length,'tests':n,'retrieval_hit_at_3':correct/n,
                        'median_query_ms':round(sorted(latency)[len(latency)//2]*1000,3),
                        'limitation':'Lexical recall only; NO LLM answer or comprehension test'})
    return outputs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--examples',type=int,default=12)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    rows=run(n=a.examples)
    print(json.dumps(rows,ensure_ascii=False,indent=2))
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True)
        a.out.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
