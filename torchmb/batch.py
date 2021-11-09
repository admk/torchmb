import copy
from typing import Type, Dict, OrderedDict, Callable, Iterator, Tuple

import torch
from torch import nn, fx

from .base import AbstractBatchModule, StateDict
from .layers import BatchLinear, BatchConv2d, BatchBatchNorm2d


BATCH_FUNCS: Dict[Type, Callable[[nn.Module, int], nn.Module]] = {
    nn.Linear: (
        lambda module, batch: BatchLinear(
            module.in_features, module.out_features, module.bias, batch)),
    nn.Conv2d: (
        lambda module, batch: BatchConv2d(
            module.in_channels, module.out_channels,
            module.kernel_size, module.stride, module.padding,
            module.dilation, module.groups, module.bias,
            module.padding_mode, batch)),
    nn.BatchNorm2d: (
        lambda module, batch: BatchBatchNorm2d(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats, batch)),
    # nn.ReLU: BatchElementWise,
    # nn.MaxPool2d: BatchElementWise,
}


class BatchModule(AbstractBatchModule):
    def __init__(self, model: nn.Module, batch: int = 1, inplace: bool = True):
        super().__init__(batch)
        self._module = self._create_batch_module(model, inplace)
        self.load_state_dicts(model.state_dict())

    def _match_replace(
            self, node: fx.Node, modules: Dict[str, nn.Module]) -> None:
        if len(node.args) == 0:
            return
        if not isinstance(node, fx.Node):
            return
        if node.op != 'call_module':
            return
        if not isinstance(node.target, str):
            return
        try:
            module = modules[node.target]
            func = BATCH_FUNCS[type(module)]
        except KeyError:
            return
        *parent, name = node.target.rsplit('.', 1)
        parent_name = parent[0] if parent else ''
        setattr(modules[parent_name], name, func(module, self.batch))

    def _create_batch_module(
        self, model: nn.Module, inplace: bool = True
    ) -> nn.Module:
        if not inplace:
            model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)
        modules = dict(fx_model.named_modules())
        graph = copy.deepcopy(fx_model.graph)
        for node in graph.nodes:
            self._match_replace(node, modules)
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
        self, x: torch.Tensor, merge: bool = True, split: bool = True
    ) -> torch.Tensor:
        if merge:
            x = self.merge_batch(x)
        x = self._module(x)
        if split:
            x = self.split_batch(x)
        return x

    def extra_repr(self) -> str:
        return f'{super().extra_repr()}, '
