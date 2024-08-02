import functools
from typing import Type, TypeVar, Dict, Callable, Tuple, Any, Union

from torch import nn, fx, Tensor

from .base import AbstractBatchModule
from .types import (
    DataOrder, ELEMENTWISE_FUNCS, BATCH_INDEPENDENT_FUNCS,
    PARAMETER_FREE_ELEMENTWISE_MODULES, BATCH_DEPENDENT_MODULES)
from .functional import merge_batch, split_batch


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


BATCH_MODULES: Dict[Type[nn.Module], Type[AbstractBatchModule]] = {}


def to_batch_module(
    node: fx.Node, module: nn.Module, batch: int
) -> Union[nn.Module, AbstractBatchModule]:
    if isinstance(module, PARAMETER_FREE_ELEMENTWISE_MODULES):
        return module
    if not isinstance(module, BATCH_DEPENDENT_MODULES):
        if all(not p.requires_grad for p in module.parameters()):
            # Skip non-learnable batch-independent modules
            return module
    try:
        batch_func = BATCH_MODULES[type(module)]
    except KeyError as e:
        raise NotImplementedError(
            f'Batch module for {type(module)} is not registered.') from e
    return batch_func.from_module(module, batch)


BatchModuleClass = TypeVar('BatchModuleClass', bound=AbstractBatchModule)


def register_batch_module(
    batch_module: Type[BatchModuleClass]
) -> Type[BatchModuleClass]:
    BATCH_MODULES[batch_module.base_class] = batch_module
    return batch_module
