import unittest
from typing import List, Tuple, Sequence, Mapping
import functools

import torch
from torch import nn, Tensor

from torchmb.batch import (
    AbstractBatchModule, BatchModule,
    BatchLinear, BatchConv2d, BatchBatchNorm2d)


StateDict = Mapping[str, Tensor]


class TestBase(unittest.TestCase):
    rtoi = 1e-4
    atoi = 1e-4
    model_batch = 17
    image_batch = 64
    xs = None
    batch_module: AbstractBatchModule
    modules: List[nn.Module] = []
    lr = 0.01
    momentum = 0.5
    weight_decay = 1e-3

    def forward(self, xs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        xs = xs.clone().detach().requires_grad_()
        xb = xs.clone().detach().requires_grad_()
        ss = [m.state_dict() for m in self.modules]
        self.batch_module.load_state_dicts(ss)
        rs = torch.stack([m(x) for m, x in zip(self.modules, xs)])
        rb = self.batch_module.merge_batch(xb)
        if isinstance(self.batch_module, BatchModule):
            rb = self.batch_module(rb, merge=False, split=False)
        else:
            rb = self.batch_module(rb)
        rb = self.batch_module.split_batch(rb)
        return xs, xb, rs, rb

    def grad(self, t: Tensor) -> Tensor:
        self.assertIsNotNone(t.grad)
        return t.grad

    def assertStatesAllClose(
        self, batch_module: AbstractBatchModule, modules: Sequence[nn.Module]
    ) -> None:
        sb = batch_module.state_dicts()
        ss = [m.state_dict() for m in modules]
        self.assertEquals(len(sb), len(ss))
        for a, b in zip(sb, ss):
            keys = set(a.keys()) | set(b.keys())
            for k in keys:
                self.assertTrue(
                    torch.allclose(a[k], b[k], self.rtoi, self.atoi),
                    f'Key {k!r} mismatch.')

    def test_state_dict(self) -> None:
        if not self.modules:
            return
        ss = [m.state_dict() for m in self.modules]
        self.batch_module.load_state_dicts(ss)
        self.assertStatesAllClose(self.batch_module, self.modules)

    def test_forward_backward(self) -> None:
        # FIXME this test fails 5% of the time for conv.
        if self.xs is None:
            return
        # forward
        xs, xb, rs, rb = self.forward(self.xs)
        self.assertTrue(torch.allclose(rs, rb, self.rtoi, self.atoi))
        # backward
        rb.sum().backward()
        rs.sum().backward()
        pb = {
            k: self.grad(p)
            for k, p in self.batch_module.named_parameters() if p.requires_grad
        }
        states = [dict(m.named_parameters()) for m in self.modules]
        # param grads all close
        for k in pb.keys():
            psk = torch.stack([self.grad(s[k]) for s in states])
            self.assertTrue(torch.allclose(psk, pb[k], self.rtoi, self.atoi))
        # input grads all close
        self.assertTrue(
            torch.allclose(self.grad(xs), self.grad(xb), self.rtoi, self.atoi))

    def test_optimize(self) -> None:
        if self.xs is None:
            return
        opt_func = functools.partial(
            torch.optim.SGD, lr=self.lr, momentum=self.momentum,
            weight_decay=self.weight_decay)
        ob = opt_func(self.batch_module.parameters())
        os = [opt_func(m.parameters()) for m in self.modules]
        xs, xb, rs, rb = self.forward(self.xs)
        # backward
        rb.sum().backward()
        rs.sum().backward()
        ob.step()
        for o in os:
            o.step()
        self.assertStatesAllClose(self.batch_module, self.modules)


class TestLinear(TestBase):
    def setUp(self):
        in_features = 100
        out_features = 200
        self.xs = torch.randn(
            self.model_batch, self.image_batch, in_features)
        self.modules = [
            nn.Linear(in_features, out_features, True)
            for _ in range(self.model_batch)]
        self.batch_module = BatchLinear(
            in_features, out_features, True, self.model_batch)


class TestConv2d(TestBase):
    def setUp(self) -> None:
        in_features = 100
        out_features = 200
        image_size = 8
        kernel_size = 3
        self.xs = torch.randn(
            self.model_batch, self.image_batch,
            in_features, image_size, image_size)
        self.modules = [
            nn.Conv2d(in_features, out_features, kernel_size)
            for _ in range(self.model_batch)]
        self.batch_module = BatchConv2d(
            in_features, out_features, kernel_size, batch=self.model_batch)


class TestBatchNorm2d(TestBase):
    def setUp(self) -> None:
        features = 100
        image_size = 8
        self.xs = torch.randn(
            self.model_batch, self.image_batch,
            features, image_size, image_size)
        self.modules = [
            nn.BatchNorm2d(features) for _ in range(self.model_batch)]
        for m in self.modules:
            torch.nn.init.normal_(m.running_mean)
            torch.nn.init.normal_(m.running_var)
            torch.nn.init.normal_(m.weight)
            torch.nn.init.normal_(m.bias)
        self.batch_module = BatchBatchNorm2d(features, batch=self.model_batch)

    def test_stats(self) -> None:
        xs, xb, rs, rb = self.forward(self.xs)
        self.assertStatesAllClose(self.batch_module, self.modules)


class TestLeNet(TestBase):
    def setUp(self) -> None:
        from lenet import LeNet
        self.xs = torch.randn(self.model_batch, self.image_batch, 1, 28, 28)
        self.modules = [LeNet() for _ in range(self.model_batch)]
        self.batch_module = BatchModule(LeNet(), self.model_batch, False)


if __name__ == '__main__':
    unittest.main()
