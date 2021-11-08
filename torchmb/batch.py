import copy
from typing import (
    Type, Dict, OrderedDict, Callable, Union,
    Sequence, Iterator, Mapping, List, Tuple)

import torch
from torch import nn, fx

import einops


StateDict = Dict[str, torch.Tensor]


class AbstractBatchModule(nn.Module):
    base_class = nn.Module

    def __init__(self, batch: int):
        super().__init__()
        self.batch = batch

    def load_state_dicts(
        self, state_dict_or_dicts: Union[Sequence[StateDict], StateDict],
        strict: bool = True
    ) -> None:
        state_dict: StateDict = OrderedDict()
        if isinstance(state_dict_or_dicts, Mapping):
            for k, v in state_dict_or_dicts.items():
                state_dict[k] = einops.repeat(v, '... -> g ...', g=self.batch)
        else:
            for k in state_dict_or_dicts[0]:
                values = [d[k] for d in state_dict_or_dicts]
                state_dict[k] = torch.stack(values)
        self.load_state_dict(state_dict, strict=strict)

    def state_dicts(self) -> List[StateDict]:
        state = self.state_dict()
        states = [{} for _ in range(self.batch)]
        for k, v in state.items():
            if v.ndim == 0:
                raise ValueError(
                    'Model batching expects batched parameter values. '
                    f'Scalar found for parameter {k!r}.')
            if v.size(0) != self.batch:
                raise ValueError(
                    f'Model batching size mismatch for parameter {k!r}, '
                    f'({self.batch} != {v.size(0)}).')
            for s, gv in zip(states, v):
                s[k] = gv
        return states

    def _batch_size(self, x: torch.Tensor) -> int:
        if x.size(0) % self.batch:
            raise ValueError(
                'The size of tensor is not divisible by model batch size.')
        return x.size(0) // self.batch

    def merge_batch(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(x, 'g b ... -> (b g) ...')

    def split_batch(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(
            x, '(b g) ... -> g b ...', b=self._batch_size(x), g=self.batch)
        return x

    def extra_repr(self) -> str:
        reprs = self.base_class.extra_repr(self).split(', ')
        reprs.append(f'batch={self.batch}')
        return ', '.join(r for r in reprs if r)


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
    base_class = nn.Linear

    def __init__(
        self, in_features: int, out_features: int,
        bias: bool = True, batch: int = 1
    ) -> None:
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


class BatchConv2d(AbstractBatchModule):
    base_class = nn.Conv2d

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int,
        stride: int = 1, padding: int = 0, dilation: int = 1,
        groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        batch: int = 1
    ) -> None:
        super().__init__(batch)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, ) * 2
        else:
            self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = (0, 0)
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(
            torch.Tensor(batch, out_channels, in_channels, *self.kernel_size),
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


class BatchBatchNorm2d(AbstractBatchModule):
    def __init__(
            self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
            affine: bool = True, track_running_stats: bool = True,
            batch: int = 1):
        super().__init__(batch)
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.ones(batch, num_features))
            self.bias = nn.Parameter(torch.zeros(batch, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.running_mean = nn.Parameter(
                torch.zeros(batch, num_features), requires_grad=False)
            self.running_var = nn.Parameter(
                torch.ones(batch, num_features), requires_grad=False)
            self.register_buffer(
                'num_batches_tracked', torch.zeros(batch, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        self.num_batches_tracked += 1
        batch_size = self._batch_size(x)
        x = einops.rearrange(
            x, '(b g) c h w -> b (g c) h w', b=batch_size, g=self.batch)
        mean = self.running_mean.flatten()
        var = self.running_var.flatten()
        weight = self.weight.flatten()
        bias = self.bias.flatten()
        x = nn.functional.batch_norm(
            x, mean, var, weight, bias, self.training, self.momentum, self.eps)
        x = einops.rearrange(
            x, 'b (g c) h w -> (b g) c h w', b=batch_size, g=self.batch)
        return x


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
