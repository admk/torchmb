from typing import Tuple, Callable, Sequence, List

import torch
from torch import nn, Tensor

from torchmb.batch import BatcherTracer
from torchmb.functional import (
    batch_accuracy, batch_loss, Reduction, to_batch_func)

from .base import TestBase


class TestBatchFunctionalBase(TestBase):
    model_batch = 17
    image_batch = 64
    num_classes = 10

    def setUp(self) -> None:
        self.inputs = torch.randn(
            self.model_batch, self.image_batch, self.num_classes)
        self.targets = torch.randint(
            0, self.num_classes - 1, (self.model_batch, self.image_batch))


class TestAutoBatchIndependent(TestBatchFunctionalBase):
    # WIP
    nonelementwise_funcs = {
        # torch.nn.functional.softmax: [3],
    }

    def enumerate_forward(self, func, inputs):
        outputs = []
        for i in range(self.model_batch):
            outputs.append(func(inputs[i]))
        return torch.stack(outputs)

    def batch_forward(self, func, inputs):
        return to_batch_func(func, self.model_batch)(inputs)

    def test_forward(self):
        for func in self.nonelementwise_funcs:
            outputs = self.enumerate_forward(func, self.inputs)
            batch_outputs = self.batch_forward(func, self.inputs)
            self.assertAllClose(batch_outputs, outputs)

    def test_backward(self):
        for func in self.nonelementwise_funcs:
            inputs = self.inputs.clone().detach().requires_grad_()
            batch_inputs = self.inputs.clone().detach().requires_grad_()
            outputs = self.enumerate_forward(func, inputs)
            batch_outputs = self.batch_forward(func, batch_inputs)
            outputs.sum().backward()
            batch_outputs.sum().backward()
            igrad = inputs.grad
            bgrad = batch_inputs.grad
            self.assertAllClose(
                igrad, bgrad,
                f'Gradient mismatch, {(igrad - bgrad).abs().max()=}')


class TestBatchAccuracy(TestBatchFunctionalBase):
    k = (1, 5)

    def accuracy(
        self, inputs: Tensor, targets: Tensor,
        k: Sequence[int] = (1, ), count: bool = False
    ) -> List[float]:
        _, pred = inputs.topk(max(k), 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        batch = 1 if count else targets.size(0)
        return [float(correct[:i].sum()) / batch for i in k]

    def test_forward(self):
        topks = torch.tensor([
            self.accuracy(i, t, self.k)
            for i, t in zip(self.inputs, self.targets)]).t()
        batch_topks = batch_accuracy(
            self.inputs, self.targets, self.model_batch, self.k)
        self.assertAllClose(
            topks, batch_topks, f'Value not matched:\n{topks}\n{batch_topks}')


class TestBatchLoss(TestBatchFunctionalBase):
    rtoi = 1e-6
    atoi = 1e-6

    def _test_forward(
        self, loss_func: Callable[..., Tensor], reduction: Reduction
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        inputs = self.inputs.clone().detach()
        inputs.requires_grad_()
        targets = self.targets.clone().detach()
        losses = torch.stack([
            loss_func(inputs[b], targets[b], reduction=reduction)
            for b in range(self.model_batch)])
        batch_inputs = self.inputs.clone().detach()
        batch_inputs.requires_grad_()
        batch_targets = self.targets.clone().detach()
        batch_losses = batch_loss(
            batch_inputs, batch_targets, self.model_batch,
            loss_func, reduction)
        self.assertAllClose(
            losses, batch_losses,
            f'Loss mismatch, {(losses - batch_losses).abs().max()=}.')
        return inputs, losses, batch_inputs, batch_losses

    def _test_backward(
        self, inputs: Tensor, losses: Tensor,
        batch_inputs: Tensor, batch_losses: Tensor,
    ) -> None:
        losses.sum().backward()
        batch_losses.sum().backward()
        igrad = self.grad(inputs)
        bgrad = self.grad(batch_inputs)
        self.assertAllClose(
            igrad, bgrad,
            f'Gradient mismatch, {(igrad - bgrad).abs().max()=}.')

    def _test_loss(
        self, loss_func: Callable[..., Tensor], reduction: Reduction
    ):
        self._test_backward(*self._test_forward(loss_func, reduction))

    def test_cross_entropy_loss(self):
        return self._test_loss(nn.functional.cross_entropy, 'mean')
