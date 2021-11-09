import einops
import torch
from torch import nn

from .base import AbstractBatchModule


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
        x = einops.rearrange(
            x, '(b g) i h w -> b (g i) h w', b=batch_size, g=self.batch)
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
