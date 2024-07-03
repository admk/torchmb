import random
import functools
from typing import List, Tuple, Mapping, Sequence

import numpy as np
import torch
from torch import nn, Tensor

from torchmb.batch import (
    AbstractBatchModule, BatchModule,
    BatchLinear, BatchConv2d, BatchBatchNorm2d,
    merge_batch, split_batch)

from .base import TestBase


StateDict = Mapping[str, Tensor]


class TestLayerBase(TestBase):
    model_batch = 17
    image_batch = 64
    xs = None
    batch_module: AbstractBatchModule
    modules: List[nn.Module] = []
    lr = 1
    momentum = 0.5
    weight_decay = 1

    def forward(self, xs: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        xs = xs.clone().detach().requires_grad_()
        xb = xs.clone().detach().requires_grad_()
        ss = [m.state_dict() for m in self.modules]
        self.batch_module.load_state_dicts(ss)
        rs = torch.stack([m(x) for m, x in zip(self.modules, xs)]).contiguous()
        rb = merge_batch(xb)
        if isinstance(self.batch_module, BatchModule):
            rb = self.batch_module(rb, merge=False, split=False)
        else:
            rb = self.batch_module(rb)
        rb = split_batch(rb, self.model_batch)
        return xs, xb, rs, rb

    def assertStatesAllClose(
        self, batch_module: AbstractBatchModule, modules: Sequence[nn.Module]
    ) -> None:
        sb = batch_module.state_dicts()
        ss = [m.state_dict() for m in modules]
        self.assertEqual(len(sb), len(ss))
        for a, b in zip(sb, ss):
            keys = set(a.keys()) | set(b.keys())
            for k in keys:
                msg = f'Key {k!r} mismatch, {(a[k] - b[k]).abs().max()=}.'
                self.assertAllClose(a[k], b[k], msg)


    def test_state_dict(self) -> None:
        if not self.modules:
            return
        ss = [m.state_dict() for m in self.modules]
        self.batch_module.load_state_dicts(ss)
        self.assertStatesAllClose(self.batch_module, self.modules)

    def test_forward(self) -> None:
        if self.xs is None:
            return
        for m in self.modules:
            m.eval()
        self.batch_module.eval()
        with torch.no_grad():
            _, _, rs, rb = self.forward(self.xs)
        self.assertAllClose(rs, rb, f'{(rs - rb).abs().max()=}.')

    def test_backward(self) -> None:
        # FIXME this test fails 5% of the time for conv.
        if self.xs is None:
            return
        # forward
        xs, xb, rs, rb = self.forward(self.xs)
        self.assertAllClose(rs, rb, f'{(rs - rb).abs().max()=}')
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
            msg = (
                f'Gradient for key {k} does not match, '
                f'{(psk - pb[k]).abs().max()=}')
            self.assertAllClose(psk, pb[k], msg)
        # input grads all close
        gs, gb = self.grad(xs), self.grad(xb)
        msg = f'Gradient for input does not match, {(gs - gb).abs().max()=}'
        self.assertAllClose(gs, gb, msg)

    def test_independent(self) -> None:
        if self.xs is None:
            return
        xs, xb, rs, rb = self.forward(self.xs)
        m = random.randint(0, self.model_batch - 1)
        rs[m].sum().backward(retain_graph=True)
        rb[m].sum().backward(retain_graph=True)
        # input gradients
        self.assertAllClose(
            self.grad(xs)[m], self.grad(xb)[m],
            f'Backprop to the input of the {m}-th module failed.')
        zero_idx = np.array([i != m for i in range(self.model_batch)])
        self.assertEqual(
            self.grad(xs)[zero_idx].abs().max(), 0,
            f'Input gradient leaked to modules other than the {m}-th.')
        # parameter gradients
        parameters = [dict(m.named_parameters()) for m in self.modules]
        for k, pb in self.batch_module.named_parameters():
            ps = parameters[m][k]
            gs = self.grad(ps)
            gb = self.grad(pb)[m]
            self.assertAllClose(
                gs, gb,
                f'Gradient does not match for parameter {k!r} '
                f'on the {m}-th module, {(gs - gb).abs().max()=}.')
            gb = self.grad(pb)[zero_idx]
            self.assertEqual(
                gb.abs().max(), 0.0,
                f'Parameter gradient leaked to modules other than the {m}-th, '
                f'{gb.abs().max()=}.')

    def test_optimize(self) -> None:
        if self.xs is None:
            return
        for m in self.modules:
            m.train()
        self.batch_module.train()
        opt_func = functools.partial(
            torch.optim.SGD, lr=self.lr, momentum=self.momentum, nesterov=True,
            weight_decay=self.weight_decay)
        ob = opt_func(self.batch_module.parameters())
        os = [opt_func(m.parameters()) for m in self.modules]
        _, _, rs, rb = self.forward(self.xs)
        # backward
        rb.sum().backward()
        rs.sum().backward()
        ob.step()
        for o in os:
            o.step()
        self.assertStatesAllClose(self.batch_module, self.modules)


class TestLinear(TestLayerBase):
    rtoi = 1e-5
    atoi = 1e-5

    def setUp(self):
        in_features = 100
        out_features = 200
        device = torch.device('cpu')
        self.xs = torch.randn(
            self.model_batch, self.image_batch, in_features).to(device)
        self.modules = [
            nn.Linear(in_features, out_features, True).to(device)
            for _ in range(self.model_batch)]
        self.batch_module = BatchLinear(
            in_features, out_features, True, self.model_batch).to(device)


class TestConv2d(TestLayerBase):
    rtoi = 1e-4
    atoi = 1e-4

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


class TestBatchNorm2d(TestLayerBase):
    def setUp(self) -> None:
        features = 100
        image_size = 8
        self.xs = torch.randn(
            self.model_batch, self.image_batch,
            features, image_size, image_size)
        self.modules = [
            nn.BatchNorm2d(features) for _ in range(self.model_batch)]
        for m in self.modules:
            torch.nn.init.uniform_(m.running_var, 1, 2)
            torch.nn.init.normal_(m.running_mean)
            torch.nn.init.uniform_(m.weight, 1, 2)
            torch.nn.init.normal_(m.bias)
        self.batch_module = BatchBatchNorm2d(features, batch=self.model_batch)

    def test_stats(self) -> None:
        self.forward(self.xs)
        self.assertStatesAllClose(self.batch_module, self.modules)


class TestLeNet(TestLayerBase):
    rtoi = 1e-5
    atoi = 1e-5

    def setUp(self) -> None:
        from .lenet import LeNet
        self.xs = torch.randn(self.model_batch, self.image_batch, 1, 28, 28)
        self.modules = [LeNet() for _ in range(self.model_batch)]
        self.batch_module = BatchModule(LeNet(), self.model_batch, False)


class TestResNet(TestLayerBase):
    rtoi = 1e-4
    atoi = 1e-4
    model_batch = 5
    image_batch = 7

    def setUp(self) -> None:
        from .resnet import ResNet18
        self.xs = torch.randn(self.model_batch, self.image_batch, 3, 32, 32)
        self.modules = [ResNet18() for _ in range(self.model_batch)]
        self.batch_module = BatchModule(ResNet18(), self.model_batch, False)
