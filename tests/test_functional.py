from typing import Tuple, Callable, Sequence, List

import torch
from torch import nn, Tensor

from torchmb.functional import batch_topk, batch_loss, Reduction

from base import TestBase


class TestBatchFunctionalBase(TestBase):
    model_batch = 17
    image_batch = 64
    num_classes = 10

    def setUp(self) -> None:
        self.inputs = torch.randn(
            self.model_batch, self.image_batch, self.num_classes)
        self.targets = torch.randint(
            0, self.num_classes - 1, (self.model_batch, self.image_batch))


class TestBatchTopk(TestBatchFunctionalBase):
    k = (1, 5)

    def topk(
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
            self.topk(i, t, self.k)
            for i, t in zip(self.inputs, self.targets)]).t()
        batch_topks = batch_topk(
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
