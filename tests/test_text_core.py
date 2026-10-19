import tempfile
import unittest
from pathlib import Path
import torch
from pcmn.text_core import PCMNTextLM, build_config, TextConfig
from pcmn.text_tokenizer import ByteTokenizer
from pcmn.text_dataset import load_text_files, TextBatches


class TextCoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)

    def test_forward_and_token_steps_exact(self):
        cfg=build_config('smoke',vocab_size=259,conv_kernel=4)
        model=PCMNTextLM(cfg).eval()
        tokens=torch.randint(0,259,(2,16))
        with torch.no_grad():
            forward=model(tokens)
            states=model.init_states(2)
            parts=[]
            for t in range(tokens.size(1)):
                logits, states = model.step(tokens[:,t],states)
                parts.append(logits)
        self.assertTrue(torch.allclose(forward,torch.stack(parts,dim=1),atol=1e-5,rtol=1e-5))
        self.assertTrue(torch.allclose(model.prefill(tokens)[0], forward[:,-1],atol=1e-5))

    def test_causality_future_token_cannot_change_past(self):
        model=PCMNTextLM(build_config('smoke')).eval()
        t=torch.randint(0,259,(2,16))
        u=t.clone()
        u[:,8:]=torch.randint(0,259,(2,8))
        with torch.no_grad():
            a=model(t)
            b=model(u)
        self.assertTrue(torch.allclose(a[:,:8],b[:,:8],atol=1e-5))

    def test_training_and_grads(self):
        model=PCMNTextLM(build_config('smoke',dropout=0.1))
        x=torch.randint(0,259,(2,21))
        loss=model.loss(x)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad,msg=name)
            self.assertTrue(torch.isfinite(param.grad).all(),msg=name)
        out=model.generate(x[:,:2],max_new_tokens=4,greedy=True,eos_id=None)
        self.assertEqual(out.shape,(2,6))

    def test_zero_conv_history_kernel_one(self):
        model=PCMNTextLM(build_config('smoke',conv_kernel=1)).eval()
        tokens=torch.randint(0,259,(1,5))
        with torch.no_grad():
            logits=model(tokens)
            states=model.init_states(1)
            parts=[]
            for i in range(tokens.shape[1]):
                y,states=model.step(tokens[:,i],states)
                parts.append(y)
        self.assertTrue(torch.allclose(logits,torch.stack(parts,1),atol=1e-5))

    def test_tokenizer_and_no_data_leakage(self):
        tokenizer=ByteTokenizer()
        sample='Привет! こんにちは\n123🙂'
        self.assertEqual(tokenizer.decode(tokenizer.encode(sample)),sample)
        with tempfile.TemporaryDirectory() as tmp:
            train=Path(tmp)/'train.txt'
            valid=Path(tmp)/'valid.txt'
            train.write_text(sample*10,encoding='utf-8')
            valid.write_text('Другой текст\n'*20,encoding='utf-8')
            tr,_=load_text_files([train],tokenizer)
            va,_=load_text_files([valid],tokenizer)
            b=TextBatches(tr,va)
            self.assertEqual(b.batch('train',2,12,torch.device('cpu')).shape,(2,13))
            self.assertEqual(next(b.eval_iterator(2,12,torch.device('cpu'))).shape,(2,13))


if __name__=='__main__':
    unittest.main()

# Extra checks are collected by unittest discovery in class below.
class ChunkedAndBaselineTests(unittest.TestCase):
    def test_chunked_scan_matches_dense_and_gradient(self):
        from pcmn.text_core import chunked_affine_scan
        from pcmn.model import parallel_affine_scan
        torch.manual_seed(0)
        a=(torch.rand(2,15,4)*0.7+0.2).requires_grad_()
        b=torch.randn(2,15,4,requires_grad=True)
        expected=parallel_affine_scan(a,b)
        actual=chunked_affine_scan(a,b,5)
        self.assertTrue(torch.allclose(actual,expected,atol=1e-6))
        g1=torch.autograd.grad(expected.sum(),(a,b),retain_graph=True)
        g2=torch.autograd.grad(actual.sum(),(a,b))
        self.assertTrue(torch.allclose(g1[0],g2[0],atol=1e-5))
        self.assertTrue(torch.allclose(g1[1],g2[1],atol=1e-5))

    def test_transformer_causal_baseline(self):
        from pcmn.text_baseline import TransformerBaseline
        cfg=build_config('smoke')
        model=TransformerBaseline(cfg).eval()
        x=torch.randint(0,259,(1,12))
        y=x.clone()
        y[:,6:]=torch.randint(0,259,(1,6))
        with torch.no_grad():
            out=model(x)
            alt=model(y)
        self.assertTrue(torch.allclose(out[:,:6],alt[:,:6],atol=1e-5))

class PrefillTests(unittest.TestCase):
    def test_pcmn_prefill_cache_matches_step_beyond_chunk(self):
        m=PCMNTextLM(build_config('smoke',scan_chunk=5,conv_kernel=7)).eval()
        x=torch.randint(0,259,(2,13))
        with torch.no_grad():
            logits,cache=m.prefill(x)
            states=m.init_states(2)
            for t in range(x.size(1)):
                expected,states=m.step(x[:,t],states)
            self.assertTrue(torch.allclose(logits,expected,atol=1e-5))
            for (s1,c1),(s2,c2) in zip(cache,states):
                self.assertTrue(torch.allclose(s1,s2,atol=1e-5))
                self.assertTrue(torch.allclose(c1,c2,atol=1e-5))

    def test_transformer_kv_cache_matches_full_forward(self):
        from pcmn.text_baseline import TransformerBaseline
        m=TransformerBaseline(build_config('smoke')).eval()
        x=torch.randint(0,259,(2,14))
        with torch.no_grad():
            full=m(x)
            logit,cache=m.prefill(x[:,:10])
            self.assertTrue(torch.allclose(logit,full[:,9],atol=1e-4,rtol=1e-4))
            for i in range(10,14):
                logit,cache=m.step(x[:,i],cache,position=i)
                self.assertTrue(torch.allclose(logit,full[:,i],atol=1e-4,rtol=1e-4),
                                msg=f'token {i}, maxdiff={float((logit-full[:,i]).abs().max())}')
