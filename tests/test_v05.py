"""Tests for v0.5 independent retrieval, text-data and traced inference."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from pcmn.retrieval import RetrievalIndex,terms
from pcmn.text_core import PCMNTextLM,build_config
from pcmn.traced_cpu import TracedCPUDecoder
from pcmn.text_tokenizer import ByteTokenizer
from scripts.prepare_corpus import create


class RetrievalV05(unittest.TestCase):
    def test_unicode_exact_hit_and_persistence(self):
        self.assertIn('україна',terms('УКРАЇНА'))
        idx=RetrievalIndex()
        idx.add('PCMN развивается в Норвегии. Пример тестового параметра AVX2.',source='russian')
        idx.add('Вычислительная архитектура на других процессорах.',source='other')
        self.assertEqual(idx.search('Какой тестовый параметр PCMN AVX2?')[0].source,'russian')
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'memory.jsonl';idx.save(f)
            clone=RetrievalIndex.load(f)
            self.assertEqual(clone.search('PCMN AVX2')[0].source,'russian')
            self.assertEqual(len(clone.docs),len(idx.docs))

    def test_dedup_and_chunk_bounds(self):
        idx=RetrievalIndex()
        self.assertTrue(idx.add('sample unique test text',source='file-a'))
        self.assertFalse(idx.add('sample unique test text',source='file-a'))
        with self.assertRaises(ValueError):idx.add(' ',source='foo')
        self.assertGreater(idx.add_document(' '.join(f'word{i}' for i in range(700)),
                                            source='big',chunk_words=64,overlap=16),5)
        result=idx.render('word300',max_chars=1200)
        self.assertLessEqual(len(result),1200)
        self.assertIn('word300',result)

    def test_no_irrelevant_memories(self):
        idx=RetrievalIndex()
        idx.add('синтетический проект орбитального компьютера',source='one')
        self.assertEqual(idx.search('фотосинтез апельсина'),[])

    def test_long_document_retrieval(self):
        idx=RetrievalIndex()
        text=' '.join(f'Тестовый фрагмент номер {i}.' for i in range(3200))
        text+=' Уникальный объект альфапроект имеет значение фиолетовый.'
        idx.add_document(text,source='stress',chunk_words=96,overlap=16)
        result=idx.search('альфапроект фиолетовый',k=3)
        self.assertTrue(result)
        self.assertIn('альфапроект',result[0].text)


class BackendV05(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_traced_decode_matches_eager_with_cache(self):
        torch.manual_seed(72)
        for memory_slots,addressable_slots in [(0,0),(2,0),(2,3)]:
            model=PCMNTextLM(build_config('smoke',memory_slots=memory_slots,
                                          addressable_slots=addressable_slots)).eval()
            trace=TracedCPUDecoder(model)
            state=model.init_states(1);tstate=model.init_states(1)
            with torch.inference_mode():
                for _ in range(11):
                    token=torch.randint(3,259,(1,))
                    a,state=model.step(token,state)
                    b,tstate=trace.step(token,tstate)
                    self.assertTrue(torch.allclose(a,b,atol=1e-5))
                    for s,t in zip(state,tstate):
                        for x,y in zip(s,t):
                            self.assertTrue(torch.allclose(x,y,atol=1e-5))

    def test_pytorch_cuda_reference_gradient_closed_form(self):
        from pcmn.text_core import chunked_affine_scan
        torch.manual_seed(99)
        a=torch.rand(1,13,3,dtype=torch.double,requires_grad=True)*0.7+0.1
        b=torch.randn(1,13,3,dtype=torch.double,requires_grad=True)
        init=torch.randn(1,3,dtype=torch.double,requires_grad=True)
        weights=torch.randn_like(b)
        states=chunked_affine_scan(a,b,4,init)
        grads=torch.autograd.grad((weights*states).sum(),(a,b,init))
        ga=torch.empty_like(a);gb=torch.empty_like(b)
        adj=torch.zeros_like(init)
        for t in range(12,-1,-1):
            adj=weights[:,t]+(a[:,t+1]*adj if t<12 else 0)
            gb[:,t]=adj
            ga[:,t]=adj*(init if t==0 else states[:,t-1])
        self.assertTrue(torch.allclose(ga,grads[0],atol=1e-9))
        self.assertTrue(torch.allclose(gb,grads[1],atol=1e-9))
        self.assertTrue(torch.allclose(a[:,0]*adj,grads[2],atol=1e-9))


class CorpusV05(unittest.TestCase):
    def test_bpe_jsonl_iterator_uses_only_text(self):
        # Does not need optional `tokenizers` dependency installed.
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,text in [('train','Этот корпус содержит исследовательский текст. '*30),
                              ('valid','Контрольная выборка содержит другие данные. '*30)]:
                (root/f'{name}.jsonl').write_text(json.dumps({'text':text,'source':'INTERNAL-META-NEVER-TOKENIZE'},
                                                            ensure_ascii=False)+'\n',encoding='utf-8')
            captured=[]
            def fake_train(paths,vocab_size,text_iterator):
                captured.extend(text_iterator)
                return ByteTokenizer()
            with patch('scripts.prepare_corpus.train_bpe',side_effect=fake_train):
                result=create([root/'train.jsonl'],[root/'valid.jsonl'],root/'out','bpe',2048)
            self.assertEqual(result['train']['documents'],1)
            self.assertTrue(captured)
            self.assertNotIn('INTERNAL-META-NEVER-TOKENIZE',''.join(captured))

    def test_v04_model_strict_load(self):
        ck=Path(__file__).resolve().parents[1]/'runs'/'v04_addressable_demo'/'last.pt'
        if not ck.exists():self.skipTest('checkpoint not bundled')
        from pcmn.text_core import TextConfig
        data=torch.load(ck,map_location='cpu',weights_only=False)
        model=PCMNTextLM(TextConfig(**data['model_config']))
        model.load_state_dict(data['model'],strict=True)
        self.assertTrue(torch.isfinite(model(torch.randint(3,model.config.vocab_size,(1,5)))).all())

if __name__=='__main__':unittest.main()
