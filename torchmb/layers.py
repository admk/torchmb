from typing import Tuple

import einops
import torch
from torch import nn

from .base import AbstractBatchModule
from .types import IntOrInts
from .functional import inner_batch_size


def _to_int_tuple(value: IntOrInts, repeat: int = 2) -> Tuple[int, ...]:
    if isinstance(value, int):
        return (value, ) * repeat
    return value


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

    @classmethod
    def from_module(cls, module, batch):
        return cls(
            module.in_features, module.out_features,
            module.bias is not None, batch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = inner_batch_size(x, self.batch)
        weight = einops.repeat(self.weight, 'g o i -> (b g) o i', b=b)
        bias = einops.repeat(self.bias, 'g o -> (b g) o', b=b)
        x = torch.bmm(weight, x.unsqueeze(-1)).squeeze(-1) + bias
        return x


class BatchConv2d(AbstractBatchModule):
    base_class = nn.Conv2d

    @classmethod
    def from_module(cls, module: nn.Conv2d, batch: int) -> 'BatchConv2d':
        if isinstance(module.padding, str):
            raise NotImplementedError(
                'Padding mode as string is not yet supported.')
        return cls(
            module.in_channels, module.out_channels,
            module.kernel_size, module.stride, module.padding,
            module.dilation, module.groups, module.bias is not None,
            module.padding_mode, batch)

    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: IntOrInts, stride: IntOrInts = 1,
        padding: IntOrInts = 0, dilation: IntOrInts = 1,
        groups: int = 1, bias: bool = True, padding_mode: str = 'zeros',
        batch: int = 1
    ) -> None:
        super().__init__(batch)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _to_int_tuple(kernel_size)
        self.stride = _to_int_tuple(stride)
        self.padding = _to_int_tuple(padding)
        self.dilation = _to_int_tuple(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.output_padding = (0, ) * len(self.padding)  # required for repr
        self.weight = nn.Parameter(
            torch.Tensor(batch, out_channels, in_channels, *self.kernel_size),
            requires_grad=True)
        self.bias = None
        if bias:
            self.bias = nn.Parameter(
                torch.Tensor(batch, out_channels), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = inner_batch_size(x, self.batch)
        x = einops.rearrange(
            x, '(b g) i h w -> b (g i) h w', b=batch_size, g=self.batch)
        weight = einops.rearrange(self.weight, 'g o i k l -> (g o) i k l')
        if self.bias is not None:
            bias = einops.rearrange(self.bias, 'g o -> (g o)')
        else:
            bias = None
        x = nn.functional.conv2d(
            x, weight, bias, self.stride, self.padding,
            self.dilation, self.groups * self.batch)
        x = einops.rearrange(
            x, 'b (g o) h w -> (b g) o h w', b=batch_size, g=self.batch)
        return x


class BatchBatchNorm2d(AbstractBatchModule):
    base_class = nn.BatchNorm2d

    @classmethod
    def from_module(
        cls, module: nn.BatchNorm2d, batch: int
    ) -> 'BatchBatchNorm2d':
        return cls(
            module.num_features, module.eps, module.momentum, module.affine,
            module.track_running_stats, batch)

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
            self.register_buffer(
                'running_mean', torch.zeros(batch, num_features))
            self.register_buffer(
                'running_var', torch.ones(batch, num_features))
            self.register_buffer(
                'num_batches_tracked', torch.zeros(batch, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x):
        self.num_batches_tracked += 1
        b = inner_batch_size(x, self.batch)
        x = einops.rearrange(
            x, '(b g) c h w -> b (g c) h w', b=b, g=self.batch)
        mean = self.running_mean.flatten()
        var = self.running_var.flatten()
        weight = self.weight.flatten()
        bias = self.bias.flatten()
        x = nn.functional.batch_norm(
            x, mean, var, weight, bias, self.training, self.momentum, self.eps)
        x = einops.rearrange(
            x, 'b (g c) h w -> (b g) c h w', b=b, g=self.batch)
        return x
