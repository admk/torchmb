import sys
import unittest

import torch

sys.path.append('.')
from torchmb import BatchModule


class TestBatchModule(unittest.TestCase):
    def setUp(self):
        self.batch = 7

    def test_lenet(self):
        from lenet import LeNet
        modules = [LeNet() for _ in range(self.batch)]
        xs = torch.randn(self.batch, 64, 1, 28, 28)
        # single batch module
        x = xs[0]
        r = modules[0](x)
        batch_module = BatchModule(modules[0], 1, False)
        xb = x.unsqueeze(0)
        rb = batch_module(xb)
        self.assertTrue(torch.allclose(r, rb.squeeze(0)))
        # multiple modules
        rs = torch.stack([m(x) for m, x in zip(modules, xs)])
        batch_module = BatchModule(modules[0], xs.size(0), False)
        batch_module.load_state_dicts(m.state_dict() for m in modules)
        rb = batch_module(xs)
        self.assertTrue(torch.allclose(rs, rb, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
