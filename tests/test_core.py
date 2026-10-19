import unittest
import torch
from pcmn.model import parallel_affine_scan, PCMNCore, PCMNLanguageModel
from pcmn.multimodal import PCMNMultimodal
from pcmn.causal import OrderedLinearSCM
from pcmn.mirror import MirrorLearning
from pcmn.probability import DiagonalGaussian


class PCMNTests(unittest.TestCase):
    def test_scan_and_gradients(self):
        torch.manual_seed(7)
        a = (0.1 + 0.8 * torch.rand(2, 17, 8)).requires_grad_()
        b = torch.randn(2, 17, 8).requires_grad_()
        target = []
        h = torch.zeros_like(b[:, 0])
        for i in range(a.size(1)):
            h = a[:, i] * h + b[:, i]
            target.append(h)
        expected = torch.stack(target, dim=1)
        actual = parallel_affine_scan(a, b)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))
        grad_ref = torch.autograd.grad(expected.sum(), [a, b], retain_graph=True)
        grad_scan = torch.autograd.grad(actual.sum(), [a, b])
        for x, y in zip(grad_ref, grad_scan):
            self.assertTrue(torch.allclose(x, y, atol=1e-4))

    def test_forward_matches_step(self):
        torch.manual_seed(11)
        net = PCMNCore(dim=16, layers=2).eval()
        x = torch.randn(2, 19, 16)
        whole = net(x)
        states = net.init_states(2, x.device, x.dtype)
        results = []
        for i in range(x.size(1)):
            o, states = net.step(x[:, i], states)
            results.append(o)
        stepped = torch.stack(results, 1)
        self.assertTrue(torch.allclose(whole, stepped, atol=1e-5))

    def test_language_loss_and_generation(self):
        net = PCMNLanguageModel(40, 24, 2)
        seq = torch.randint(0, 40, (2, 13))
        loss = net.loss(seq)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertTrue(torch.isfinite(net.core.layers[0].drive.weight.grad).all().item())
        net.eval()
        out = net.sample(seq[:, :4], new_tokens=3)
        self.assertEqual(tuple(out.shape), (2, 7))

    def test_multimodal_inputs_outputs(self):
        torch.manual_seed(1)
        mm = PCMNMultimodal(100, dim=16, layers=1, max_query_tokens=64)
        x = {'text':torch.randint(0, 100, (2, 4)),
             'image':torch.rand(2, 3, 16, 16),
             'audio':torch.rand(2, 1, 64),
             'video':torch.rand(2, 3, 2, 16, 16)}
        o = mm(x)
        self.assertEqual(tuple(o['text_logits'].shape), (2, 4, 100))
        for name in ('image', 'audio', 'video'):
            self.assertEqual(tuple(o[name].shape), tuple(x[name].shape))
        with torch.no_grad():
            img = mm.synthesize({'text':x['text']}, 'image', (16,16))
            generated = mm.generate_text({'image':x['image']}, new_tokens=3)
        self.assertEqual(tuple(img.shape), (2,3,16,16))
        self.assertEqual(tuple(generated.shape), (2,3))

    def test_causal_intervention(self):
        model = OrderedLinearSCM(3)
        with torch.no_grad():
            model.weights[1,0] = 2.0
            model.weights[2,1] = 3.0
        noise = torch.zeros(2,3)
        self.assertTrue(torch.allclose(model(noise), noise))
        intervention = model(noise, {0:2.0})
        self.assertTrue(torch.allclose(intervention[0], torch.tensor([2.0, 4.0, 12.0])))
        intervention2 = model(noise, {1:5.0})
        self.assertTrue(torch.allclose(intervention2[0], torch.tensor([0.0,5.0,15.0])))

    def test_mirror_probability(self):
        mirror = MirrorLearning(5,4,8)
        loss = mirror.paired_loss(torch.randn(3,5),torch.randn(3,4))
        self.assertTrue(torch.isfinite(loss).item())
        gaussian = DiagonalGaussian(5, 4)
        mu, logvar = gaussian(torch.randn(3,5))
        nll = gaussian.negative_log_likelihood(torch.randn(3,4), mu, logvar)
        self.assertTrue(torch.isfinite(nll).item())


if __name__ == '__main__':
    unittest.main()
