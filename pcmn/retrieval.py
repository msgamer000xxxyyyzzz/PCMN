"""Auditable CPU lexical memory (BM25) for long-document question answering.

This is a separate, non-neural retrieval index, NOT a trained differentiable
memory or proof that the language model understands 32K tokens. It provides a
cheap exact-term retrieval baseline, including Unicode text and source IDs.
Never execute retrieved text as commands or interpret it as tool authority.
"""
from __future__ import annotations
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re

WORDS = re.compile(r'[^\W_]+(?:_[^\W_]+)*', re.UNICODE)


def terms(text: str) -> list[str]:
    return [t.casefold() for t in WORDS.findall(text)]


@dataclass(frozen=True)
class Hit:
    id: str
    source: str
    text: str
    score: float


class RetrievalIndex:
    def __init__(self, passages=None, max_docs=100_000):
        self.max_docs = max_docs
        self.docs = []
        self.postings = defaultdict(dict)
        self.lengths = []
        self.keys = set()
        self.total_len = 0
        for item in (passages or []):
            self.add(item['text'], source=item['source'], id=item.get('id'))

    def add(self, text: str, source: str, id: str | None = None) -> bool:
        if not isinstance(text, str) or not text.strip() or len(text) > 32768:
            raise ValueError('Expected nonempty passage of at most 32768 characters')
        if not isinstance(source, str) or not source or len(source) > 1024:
            raise ValueError('source required (at most 1024 characters)')
        if len(self.docs) >= self.max_docs:
            raise ValueError('Index full')
        digest = hashlib.sha256((source+'\0'+text).encode('utf-8')).hexdigest()
        if digest in self.keys:
            return False
        if id is None:
            id = digest[:16]
        if not isinstance(id,str) or not id or len(id)>256 or any(d['id']==id for d in self.docs):
            raise ValueError('Duplicate/invalid passage ID')
        tokens=terms(text)
        if not tokens:
            return False
        idx=len(self.docs)
        self.docs.append({'id':id,'source':source,'text':text})
        self.keys.add(digest)
        self.lengths.append(len(tokens))
        self.total_len += len(tokens)
        for word, freq in Counter(tokens).items():
            self.postings[word][idx] = freq
        return True

    def add_document(self, text: str, source: str, chunk_words=180, overlap=40):
        """Chunk along word offsets without copying the entire document N times."""
        if not isinstance(text,str):raise TypeError('text must be str')
        if not 8 <= chunk_words <= 2048 or not 0 <= overlap < chunk_words:
            raise ValueError('Invalid chunk/overlap size')
        matches=list(WORDS.finditer(text))
        count=0
        step=chunk_words-overlap
        for start in range(0,len(matches),step):
            end=min(len(matches),start+chunk_words)
            if end <= start:break
            passage=text[matches[start].start():matches[end-1].end()]
            if self.add(passage,source=source):count+=1
            if end==len(matches):break
        return count

    def search(self, query: str, k=5) -> list[Hit]:
        if not isinstance(query,str) or not 1 <= k <= 100:
            raise ValueError('Invalid query or top-k')
        words=set(terms(query))
        if not words or not self.docs:
            return []
        n=len(self.docs)
        avg=max(self.total_len/n,1)
        scores=defaultdict(float)
        for term in words:
            posting=self.postings.get(term)
            if not posting:continue
            df=len(posting)
            idf=math.log(1+(n-df+0.5)/(df+0.5))
            for doc,tf in posting.items():
                norm=1.2*(0.25+0.75*self.lengths[doc]/avg)
                scores[doc]+=idf*tf*2.2/(tf+norm)
        indices=sorted(scores, key=lambda i:(-scores[i],self.docs[i]['id']))[:k]
        return [Hit(**self.docs[i],score=round(scores[i],6)) for i in indices]

    def save(self, path):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp')
        with tmp.open('w',encoding='utf-8') as fd:
            for item in self.docs:
                fd.write(json.dumps(item,ensure_ascii=False)+'\n')
        tmp.replace(path)

    @classmethod
    def load(cls,path):
        index=cls()
        with Path(path).open(encoding='utf-8') as fd:
            for line in fd:
                if line.strip():
                    data=json.loads(line)
                    index.add(data['text'],data['source'],data.get('id'))
        return index

    def render(self, query, k=4, max_chars=3000):
        if max_chars<0:raise ValueError('max_chars must be nonnegative')
        out=[];size=0
        for hit in self.search(query,k):
            snippet=f'[SOURCE {hit.id} / {hit.source}]\n{hit.text}\n'
            if size+len(snippet)>max_chars:
                break
            out.append(snippet);size+=len(snippet)
        return ''.join(out)
