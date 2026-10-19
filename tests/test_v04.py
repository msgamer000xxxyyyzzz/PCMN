"""PCMN v0.4 incremental regression tests."""
import json, os, tempfile, unittest, sys
from pathlib import Path
import torch
from pcmn.fast_scan import configure_cpu_backend
from pcmn.text_core import PCMNTextLM,build_config,chunked_affine_scan
from pcmn.sharding import ShardedPCMN
from pcmn.task_suite import make_items,check_code,write_chat
from pcmn.dev_tools import DevTools
from scripts.curate_corpus import curate

class V04Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_native_scan_exact_forward(self):
        base=Path(__file__).resolve().parents[1]/'native'/'build'
        suffix='.dll' if sys.platform=='win32' else '.so'
        files=list(base.rglob('*pcmn_cpu*'+suffix))
        if not files:self.skipTest('Native library not built')
        a=torch.rand(2,91,13)*.8;b=torch.randn_like(a);h=torch.randn(2,13)
        with torch.no_grad():
            default=chunked_affine_scan(a,b,17,h)
            configure_cpu_backend(files[0])
            try:
                accelerated=chunked_affine_scan(a,b,17,h)
                self.assertLess((default-accelerated).abs().max().item(),2e-5)
            finally:
                configure_cpu_backend(None)

    def test_grad_still_uses_autograd(self):
        a=torch.rand(2,12,4,requires_grad=True);b=torch.rand(2,12,4,requires_grad=True)
        out=chunked_affine_scan(a,b,7)
        out.sum().backward()
        self.assertGreater(float(a.grad.abs().sum()),0)

    def test_address_chunk_matches_full_step(self):
        torch.manual_seed(32)
        m=PCMNTextLM(build_config('smoke',memory_slots=2,addressable_slots=4,scan_chunk=9)).eval()
        tokens=torch.randint(3,259,(2,27))
        with torch.no_grad():
            full=m(tokens)
            cached=None;parts=[]
            for s,e in [(0,5),(5,12),(12,19),(19,27)]:
                logits,cached=m.forward_chunk(tokens[:,s:e],cached)
                parts.append(logits)
            self.assertTrue(torch.allclose(full,torch.cat(parts,1),atol=1e-5,rtol=1e-5))
            states=m.init_states(2)
            for i in range(27):
                last,states=m.step(tokens[:,i],states)
            self.assertTrue(torch.allclose(last,full[:,-1],atol=2e-5))

    def test_address_memory_causal(self):
        m=PCMNTextLM(build_config('smoke',addressable_slots=4)).eval()
        x=torch.randint(3,259,(1,23));z=x.clone();z[:,12:]=(z[:,12:]+7)%259
        with torch.no_grad():
            self.assertTrue(torch.allclose(m(x)[:,:12],m(z)[:,:12],atol=2e-5))

    def test_address_memory_receives_grad(self):
        m=PCMNTextLM(build_config('smoke',addressable_slots=4))
        optim=torch.optim.AdamW(m.parameters(),lr=0.02)
        t=torch.randint(3,259,(1,18))
        for _ in range(2):
            l=m.loss(t);l.backward();optim.step();optim.zero_grad(set_to_none=True)
        m.loss(t).backward()
        self.assertGreater(float(m.blocks[0].addr_query.weight.grad.abs().sum()),0)

    def test_load_v03_checkpoint(self):
        ck=Path(__file__).resolve().parents[1]/'runs'/'v03_smoke'/'last.pt'
        if not ck.exists():self.skipTest('v0.3 checkpoint not bundled')
        obj=torch.load(ck,map_location='cpu',weights_only=False)
        from pcmn.text_core import TextConfig
        model=PCMNTextLM(TextConfig(**obj['model_config']))
        model.load_state_dict(obj['model'],strict=True)
        with torch.no_grad():self.assertTrue(torch.isfinite(model(torch.randint(3,259,(1,8)))).all())

    def test_shard_matches_reference(self):
        torch.manual_seed(33)
        m=PCMNTextLM(build_config('smoke',addressable_slots=2,memory_slots=2)).eval()
        t=torch.randint(3,259,(1,24))
        with torch.inference_mode():
            r,rs=m.prefill(t,chunk_size=7)
            r2,rs2=ShardedPCMN(m,['cpu','cpu']).prefill(t,chunk_size=7)
            self.assertTrue(torch.allclose(r,r2,atol=3e-5))

    def test_curation_license_exact_dedup(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);inp=root/'data.jsonl';doc='A research note of sufficient length for the corpus.'
            data=[{'text':doc,'license':'CC-BY-4.0','source':'a'},
                  {'text':doc,'license':'CC-BY-4.0','source':'b'},
                  {'text':'No permission '+doc,'license':'Restricted','source':'c'}]
            for i in range(60):data.append({'text':f'{doc} record {i}', 'license':'CC-BY-4.0','source':str(i)})
            inp.write_text('\n'.join(json.dumps(x) for x in data),encoding='utf-8')
            report=curate([inp],root/'out',{'CC-BY-4.0'},20)
            self.assertEqual(report['counts']['duplicate'],1)
            self.assertEqual(report['counts']['rejected_license'],1)
            self.assertGreater(report['counts']['train'],0)
            self.assertGreater(report['counts']['valid'],0)

    def test_suite_generation_and_syntax(self):
        items=make_items(24)
        self.assertEqual(set(i['category'] for i in items),{'arithmetic','causal_do','state_tracking','code_generation'})
        code=[i for i in items if i['category']=='code_generation'][0]
        import re
        fn=re.search(r'def (\w+)\(',code['answer']).group(1)
        constant=int(re.search(r'\+ (\d+)',code['answer']).group(1))
        self.assertTrue(check_code(code['answer'],fn,21,21+constant))
        self.assertFalse(check_code('import os\nos.system("echo foo")',fn,21,21+constant))

    def test_devtools_default_denied(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'tests').mkdir()
            tool=DevTools(root)
            with self.assertRaises(PermissionError):
                tool.execute('write_file',{'path':'test.txt','content':'x'})
            with self.assertRaises(PermissionError):
                tool.execute('run_tests',{'path':'tests'})
            allowed=DevTools(root,approval=lambda n,args:n=='write_file')
            allowed.execute('write_file',{'path':'hello.py','content':'print(1)'})
            self.assertEqual((root/'hello.py').read_text(),'print(1)')
            with self.assertRaises(PermissionError):
                allowed.execute('write_file',{'path':'../bad.py','content':'x'})

if __name__=='__main__':unittest.main()
