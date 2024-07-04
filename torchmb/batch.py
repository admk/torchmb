import copy
from typing import (
    Dict, OrderedDict, Iterator, Tuple, Type, Sequence, List, Union)

import torch
from torch import nn, fx, Tensor

from .types import (
    Size, StateDict, DataOrder, TensorOrTensors,
    ELEMENTWISE_FUNCS, PARAMETER_FREE_ELEMENTWISE_MODULES)
from .base import AbstractBatchModule
from .layers import BatchLinear, BatchConv2d, BatchBatchNorm2d
from .functional import merge_batch, split_batch, to_batch_func


BATCH_MODULES: Dict[Type[nn.Module], Type[AbstractBatchModule]] = {
    nn.Linear: BatchLinear,
    nn.Conv2d: BatchConv2d,
    nn.BatchNorm2d: BatchBatchNorm2d,
}


def register_batch_module(batch_module: Type[AbstractBatchModule]) -> None:
    BATCH_MODULES[batch_module.base_class] = batch_module


def to_batch_module(
    module: nn.Module, batch: int
) -> Union[nn.Module, AbstractBatchModule]:
    if isinstance(module, PARAMETER_FREE_ELEMENTWISE_MODULES):
        return module
    try:
        batch_func = BATCH_MODULES[type(module)]
    except KeyError as e:
        raise NotImplementedError(
            f'Batch module for {type(module)} is not registered.') from e
    return batch_func.from_module(module, batch)


class BatcherTracer(fx.Tracer):
    def symbolic_trace(self, module: nn.Module) -> fx.GraphModule:
        graph = super().trace(module)
        return fx.GraphModule(module, graph, module.__class__.__name__)

    def is_leaf_module(
        self, m: nn.Module, module_qualified_name: str
    ) -> bool:
        if any(isinstance(m, module_cls) for module_cls in BATCH_MODULES):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class BatchModule(AbstractBatchModule):
    def __init__(self, model: nn.Module, batch: int = 1, inplace: bool = True):
        super().__init__(batch)
        self._module = self._create_batch_module(model, inplace)
        self.load_state_dicts(model.state_dict())

    def _match_replace(
        self, node: fx.Node, modules: Dict[str, fx.GraphModule],
        shared_buffers: List[str]
    ) -> None:
        if len(node.args) == 0:
            return
        if not isinstance(node, fx.Node):
            return
        if node.op == 'call_function':
            if node.target in ELEMENTWISE_FUNCS:
                return
            try:
                setattr(node, 'target', to_batch_func(node, self.batch))
            except NotImplementedError:
                print(
                    f'Node: {node.format_node()}: '
                    f'Function {node.target} not supported, '
                    'please check model-sample confluence.')
        if node.op != 'call_module':
            return
        if not isinstance(node.target, str):
            return
        module = modules[node.target]
        try:
            batch_module = to_batch_module(module, self.batch)
        except KeyError:
            return
        *parent, name = node.target.rsplit('.', 1)
        parent_name = parent[0] if parent else ''
        setattr(modules[parent_name], name, batch_module)
        if hasattr(batch_module, 'shared_buffers'):
            for n in batch_module.shared_buffers:
                shared_buffers.append(f'{node.target}.{n}')

    def _create_batch_module(
        self, model: nn.Module, inplace: bool = True
    ) -> nn.Module:
        if not inplace:
            model = copy.deepcopy(model)
        fx_model = BatcherTracer().symbolic_trace(model)
        modules = dict(fx_model.named_modules())
        graph = copy.deepcopy(fx_model.graph)
        for node in graph.nodes:
            self._match_replace(node, modules, self._shared_buffers)
        module = fx.GraphModule(fx_model, graph)
        return module

    def load_state_dict(
        self, state_dict: StateDict, strict: bool = True
    ) -> None:
        self._module.load_state_dict(OrderedDict(state_dict), strict=strict)

    def state_dict(self) -> StateDict:
        return self._module.state_dict()

    def named_parameters(
        self, prefix: str = '', recurse: bool = True
    ) -> Iterator[Tuple[str, nn.Parameter]]:
        return self._module.named_parameters(prefix=prefix, recurse=recurse)

    def forward(
        self, *x: Tensor, merge: bool = True, split: bool = True,
        data_order: DataOrder = 'g b'
    ) -> Union[Tensor, TensorOrTensors]:
        if merge:
            x = tuple(merge_batch(t, data_order) for t in x)
        x = self._module(*x)
        if split:
            if isinstance(x, Tensor):
                x = (split_batch(x, self.batch, data_order), )
            elif isinstance(x, tuple):
                x = tuple(split_batch(t, self.batch, data_order) for t in x)
            else:
                raise NotImplementedError(
                    f'Unsupported return type {type(x)}.')
        return x[0] if len(x) == 1 else x

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, '


def assert_all_close(
    a: Tensor, b: Tensor, rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    assert a.size() == b.size(), f'Size mismatch, "{a.size()} != "{b.size()}".'
    msg = f'Values mismatch, {(a - b).abs().max()=}.'
    assert torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True), msg


def test_batch_module(
    module: nn.Module, input_shapes: Sequence[Size], batch: int = 3,
    rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    state = module.state_dict()
    states = []
    for _ in range(batch):
        rand_state = {}
        for k, v in state.items():
            rand_state[k] = torch.rand_like(v)
        states.append(rand_state)
    batch_module = BATCH_MODULES[type(module)].from_module(module, batch)
    batch_module.load_state_dicts(states)
    batch_inputs = [
        torch.randn((batch, ) + tuple(shape))
        for shape in input_shapes]
    batch_outputs = batch_module(*[merge_batch(i) for i in batch_inputs])
    nary = isinstance(batch_outputs, Tensor)
    if nary:
        batch_outputs = split_batch(batch_outputs, batch)
    else:
        batch_outputs = [split_batch(b, batch) for b in batch_outputs]
    outputs = []
    for i, inputs in enumerate(zip(*batch_inputs)):
        module.load_state_dict(states[i])
        if nary:
            outputs.append(module(*inputs))
    for a, b in zip(outputs, batch_outputs):
        assert_all_close(a, b, rtol, atol)
