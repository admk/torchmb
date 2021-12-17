from typing import Callable, Any, Literal, Sequence

import torch
from torch import nn, Tensor

from .base import merge_batch, split_batch, inner_batch_size


Reduction = Literal['none', 'mean', 'sum']


def batch_loss(
    batch_inputs: Tensor, batch_targets: Tensor, batch: int,
    loss_func: Callable[..., Tensor] = nn.functional.cross_entropy,
    reduction: Reduction = 'mean', **kwargs: Any
) -> Tensor:
    batch_inputs = merge_batch(batch_inputs)
    batch_targets = merge_batch(batch_targets)
    loss = loss_func(
        batch_inputs, batch_targets, **kwargs, reduction='none')
    loss = split_batch(loss, batch)
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean(1)
    if reduction == 'sum':
        return loss.sum(1)
    raise ValueError(f'Unknown reduction method {reduction!r}.')


def batch_topk(
    batch_inputs: Tensor, batch_targets: Tensor, batch: int,
    k: Sequence[int] = (1, ), count: bool = False
) -> Tensor:
    batch_inputs = merge_batch(batch_inputs)
    batch_targets = merge_batch(batch_targets)
    pred = batch_inputs.topk(max(k), 1, True, True)[1]
    correct = pred == batch_targets.unsqueeze(1)
    correct = torch.cumsum(split_batch(correct, batch), 2)
    indices = torch.tensor(k, device=correct.device)
    correct = correct.index_select(2, indices - 1)
    correct = torch.einsum('gbc -> cg', correct)
    if count:
        return correct
    batch_size = inner_batch_size(batch_inputs, batch)
    return correct / batch_size
