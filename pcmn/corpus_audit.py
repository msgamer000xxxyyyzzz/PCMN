"""Audits corpus leakage and approximate duplicates with bounded sample memory.

SHA-256 exact matches, source overlap, and shingle Jaccard near duplicates.
Near-duplicate audit is sampled / not exhaustive: this is NOT license vetting.
"""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

WORD = re.compile(r"[^\W_]+",re.UNICODE)


def shingles(text, width=5):
    words = WORD.findall(text.casefold())
    if not words:return set()
    if len(words) < width:return {tuple(words)}
    return set(tuple(words[i:i+width]) for i in range(len(words)-width+1))


def _read(path, maximum):
    sources=set()
    exact=set()
    sketches=[]
    count=0
    with Path(path).open('r',encoding='utf-8-sig') as f:
        for lineno,line in enumerate(f,1):
            if not line.strip():continue
            obj=json.loads(line)
            if not isinstance(obj,dict) or not isinstance(obj.get('text'),str):
                raise ValueError(f'Invalid text at {path}:{lineno}')
            text=obj['text'].strip()
            if not text:continue
            exact.add(hashlib.sha256(text.encode('utf-8')).hexdigest())
            if isinstance(obj.get('source'),str):sources.add(obj['source'])
            if len(sketches)<maximum:sketches.append(shingles(text))
            count+=1
    return count,sources,exact,sketches


def audit(train,valid,max_sample=200,threshold=0.85):
    if not 1<=max_sample<=5000 or not 0<threshold<=1:
        raise ValueError('Invalid audit sample count or threshold')
    ntrain,ts,th,tsh=_read(train,max_sample)
    nvalid,vs,vh,vsh=_read(valid,max_sample)
    near=0
    near_examples=[]
    # Deliberately bounded quadratic comparison of samples only.
    for i,a in enumerate(tsh):
        for j,b in enumerate(vsh):
            if not a or not b:continue
            similarity=len(a & b)/len(a | b)
            if similarity>=threshold:
                near+=1
                if len(near_examples)<5:
                    near_examples.append({'train_sample':i,'valid_sample':j,
                                          'shingle_jaccard':round(similarity,4)})
    exact=len(th & vh)
    shared=sorted(ts & vs)
    return {'train_documents':ntrain,'valid_documents':nvalid,
            'exact_duplicate_hashes_across_splits':exact,
            'sources_present_in_both_splits':len(shared),
            'example_shared_sources':shared[:10],
            'approx_near_duplicate_sample_pairs':near,
            'approx_near_duplicate_examples':near_examples,
            'sampled_per_split':max_sample,'shingle_jaccard_threshold':threshold,
            'exact_leakage_check_passed':exact==0,
            'source_independence_check_passed':len(shared)==0,
            'warning':'Near-duplicate test samples first N docs per split only; false negatives possible'}
