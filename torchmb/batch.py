import copy
import collections
from typing import Type, Dict, Callable, Union, Iterable

import torch
from torch import nn, fx

import einops


StateDict = Dict[str, torch.Tensor]


class AbstractBatchModule(nn.Module):
    def __init__(self, batch: int):
        super().__init__()
        self.batch = batch

    def load_state_dicts(
            self, state_dict_or_dicts: Union[Iterable[StateDict], StateDict],
            strict: bool = True):
        if isinstance(state_dict_or_dicts, collections.abc.Mapping):
            state_dict = {
                k: einops.repeat(v, '... -> g ...', g=self.batch)
                for k, v in state_dict_or_dicts.items()}
        else:
            state_dict_or_dicts = list(state_dict_or_dicts)
            state_dict = {}
            for k in state_dict_or_dicts[0]:
                values = [d[k] for d in state_dict_or_dicts]
                state_dict[k] = torch.stack(values)
        return self.load_state_dict(state_dict, strict=strict)

    def state_dicts(self) -> StateDict:
        state = self.state_dict()
        states = [{} for _ in range(self.batch)]
        for k, v in state.items():
            if v.size(0) != self.batch:
                raise ValueError(
                    f'Model batching size mismatch for parameter: {k}, '
                    f'({self.batch} != {v.size(0)}).')
            for s, gv in zip(states, v):
                s[k] = gv
        return states

    def _batch_size(self, x: torch.Tensor):
        if x.size(0) % self.batch:
            raise ValueError(
                'The size of tensor is not divisible by model batch size.')
        return x.size(0) // self.batch



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
            self, model: nn.Module, inplace: bool=True) -> nn.Module:
        if not inplace:
            model = copy.deepcopy(model)
        fx_model = fx.symbolic_trace(model)
        modules = dict(fx_model.named_modules())
        graph = copy.deepcopy(fx_model.graph)
        for node in graph.nodes:
            self._match_replace(node, modules)
        module = fx.GraphModule(fx_model, graph)
        return module

    def load_state_dict(self, state_dict: StateDict, strict: bool = True):
        self._module.load_state_dict(state_dict, strict=strict)

    def state_dict(self) -> StateDict:
        return self._module.state_dict()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, 'g b ... -> (b g) ...')
        x = self._module(x)
        x = einops.rearrange(
            x, '(b g) ... -> g b ...', b=self._batch_size(x), g=self.batch)
        return x


# class BatchElementWise(AbstractBatchModule):
#     def __init__(self, layer: nn.Module, batch: int = 1):
#         super().__init__()
#         self.layer = layer
#         self.batch = batch

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = einops.rearrange(x, 'g b ... -> (g b) ...')
#         x = self.layer(x)
#         return einops.rearrange(x, '(g b) ... -> g b ...', g=self.batch)


class BatchLinear(AbstractBatchModule):
    def __init__(
            self, in_features: int, out_features: int,
            bias: bool = True, batch: int = 1):
        super().__init__(batch)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.Tensor(batch, out_features, in_features))
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(torch.Tensor(batch, out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = self._batch_size(x)
        weight = einops.repeat(self.weight, 'g o i -> (b g) o i', b=batch_size)
        bias = einops.repeat(self.bias, 'g o -> (b g) o', b=batch_size)
        x = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias
        return x

    def extra_repr(self) -> str:
        return torch.nn.Linear.extra_repr(self) + f', batch={self.batch}'


class BatchConv2d(AbstractBatchModule):
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int,
            stride: int = 1, padding: int = 0, dilation: int = 1,
            groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
            batch: int = 1):
        super().__init__(batch)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = (0, 0)
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(
            torch.Tensor(
            batch, out_channels, in_channels, *kernel_size),
            requires_grad=True)
        self.bias = None
        if bias is not None:
            self.bias = nn.Parameter(
                torch.Tensor(batch, out_channels), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = self._batch_size(x)
        x = einops.rearrange(x, '(b g) i h w -> b (g i) h w', b=batch_size, g=self.batch)
        weight = einops.rearrange(self.weight, 'g o i k l -> (g o) i k l')
        bias = einops.rearrange(self.bias, 'g o -> (g o)')
        x = nn.functional.conv2d(
            x, weight, bias, self.stride, self.padding,
            self.dilation, self.groups * self.batch)
        x = einops.rearrange(
            x, 'b (g o) h w -> (b g) o h w', b=batch_size, g=self.batch)
        return x

    def extra_repr(self) -> str:
        return torch.nn.Conv2d.extra_repr(self) + f', batch={self.batch}'


BATCH_FUNCS: Dict[Type, Callable[[nn.Module, int], nn.Module]] = {
    nn.Conv2d: (
        lambda module, batch: BatchConv2d(
            module.in_channels, module.out_channels,
            module.kernel_size, module.stride, module.padding,
            module.dilation, module.groups, module.bias,
            module.padding_mode, batch)),
    nn.Linear: (
        lambda module, batch: BatchLinear(
            module.in_features, module.out_features, module.bias, batch)),
    # nn.ReLU: BatchElementWise,
    # nn.MaxPool2d: BatchElementWise,
}
