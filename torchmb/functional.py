from typing import Callable, Any, Literal, Sequence, Tuple
import functools

import torch
from torch import fx, nn, Tensor
import einops

from .types import DataOrder, ELEMENTWISE_FUNCS, BATCH_INDEPENDENT_FUNCS


def inner_batch_size(x: Tensor, batch: int) -> int:
    if x.shape[0] % batch:
        raise RuntimeError(
            f'Tensor with shape {x.shape} '
            f'cannot be split into {batch} batches.')
    return x.shape[0] // batch


def merge_batch(x: Tensor, data_order: DataOrder = 'g b') -> Tensor:
    return einops.rearrange(x, f'{data_order} ... -> (b g) ...')


def split_batch(
    x: Tensor, batch: int, data_order: DataOrder = 'g b'
) -> Tensor:
    return einops.rearrange(x, f'(b g) ... -> {data_order} ...', g=batch)


def _recursive_apply(
    func: Callable[..., Tensor], *args: Any, **kwargs: Any
) -> Tuple[Tuple[Any, ...], dict]:
    largs = list(args)
    for i, arg in enumerate(args):
        if isinstance(arg, Tensor):
            largs[i] = func(arg)
        elif isinstance(arg, (list, tuple)):
            largs[i] = _recursive_apply(func, *arg)
    for k, v in kwargs.items():
        kwargs[k] = _recursive_apply(func, v)
    return tuple(largs), kwargs


def to_batch_func(
    node: fx.Node, batch: int, data_order: DataOrder = 'g b'
) -> Callable:
    func = node.target
    if func in ELEMENTWISE_FUNCS:
        return func
    if func in BATCH_INDEPENDENT_FUNCS:
        def bif(*args, **kwargs):
            args, kwargs = _recursive_apply(
                lambda x: merge_batch(x, data_order), *args, **kwargs)
            args = func(*args, **kwargs)
            if not isinstance(args, tuple):
                args = (args, )
            args, _ = _recursive_apply(
                lambda x: split_batch(x, batch, data_order), *args)
            if isinstance(args, tuple) and len(args) == 1:
                return args[0]
            return args
        bif = functools.wraps(func)(bif)
        bif.__name__ = f'batch_{func.__name__}'
        bif.__qualname__ = f'batch_{func.__qualname__}'
        return bif
    raise NotImplementedError(f'Function {func} not supported.')


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
    loss = split_batch(loss, batch=batch)
    if reduction == 'none':
        return loss
    if reduction == 'mean':
        return loss.mean(1)
    if reduction == 'sum':
        return loss.sum(1)
    raise ValueError(f'Unknown reduction method {reduction!r}.')


def batch_accuracy(
    batch_inputs: Tensor, batch_targets: Tensor, batch: int,
    k: Sequence[int] = (1, ), count: bool = False
) -> Tensor:
    batch_inputs = merge_batch(batch_inputs)
    batch_targets = merge_batch(batch_targets)
    pred = batch_inputs.topk(max(k), 1, True, True)[1]
    correct = pred == batch_targets.unsqueeze(1)
    correct = torch.cumsum(split_batch(correct, batch=batch), 2)
    indices = torch.tensor(k, device=correct.device)
    correct = correct.index_select(2, indices - 1)
    correct = torch.einsum('gbc -> cg', correct)
    if count:
        return correct
    batch_size = inner_batch_size(batch_inputs, batch)
    return correct / batch_size
