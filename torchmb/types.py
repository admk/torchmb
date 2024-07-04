import operator
from typing import Any, Union, Dict, Literal, Callable, Tuple, Type

import torch
from torch import nn, Tensor
from torch.nn import functional as F


StateDict = Dict[str, Any]
DataOrder = Literal['g b', '(g b)', 'b g', '(b g)']
ForwardFunc = Callable[[Tensor], Tensor]
Tensors = Tuple[Tensor, ...]
TensorOrTensors = Union[Tensor, Tensors]
Size = Union[torch.Size, Tuple[int, ...]]
IntOrInts = Union[int, Size]


ELEMENTWISE_FUNCS: Tuple[Callable, ...] = (
    getattr,
    torch.add, torch.sub, torch.mul, torch.div, torch.pow, torch.relu,
    torch.sigmoid, torch.tanh, F.relu,
    F.sigmoid, F.tanh, F.relu6, F.leaky_relu, F.elu, F.selu, F.gelu,
    F.logsigmoid, F.softplus, F.softshrink, F.softsign, F.tanhshrink,
    F.hardshrink, F.hardsigmoid, F.hardswish, F.hardtanh, F.silu, F.mish,
    F.avg_pool1d, F.avg_pool2d, F.avg_pool3d,
    F.max_pool1d, F.max_pool2d, F.max_pool3d,
    F.max_unpool1d, F.max_unpool2d, F.max_unpool3d,
    F.fractional_max_pool2d, F.fractional_max_pool3d,
    F.lp_pool1d, F.lp_pool2d,
    F.adaptive_max_pool1d, F.adaptive_max_pool2d, F.adaptive_max_pool3d,
    F.adaptive_avg_pool1d, F.adaptive_avg_pool2d, F.adaptive_avg_pool3d,
    operator.add, operator.sub, operator.mul, operator.truediv,
    operator.pow, operator.abs, operator.neg, operator.pos, operator.invert,
    operator.not_, operator.lt, operator.le, operator.eq, operator.ne,
    operator.ge, operator.gt, operator.and_, operator.or_, operator.xor,
    operator.lshift, operator.rshift, operator.floordiv, operator.mod,
)
BATCH_INDEPENDENT_FUNCS: Tuple[Callable, ...] = (
    F.softmax, F.log_softmax,
)
PARAMETER_FREE_ELEMENTWISE_MODULES: Tuple[Type[nn.Module], ...] = (
    nn.ReLU, nn.Sigmoid, nn.Tanh, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.SELU,
    nn.GELU, nn.LogSigmoid, nn.Softplus, nn.Softshrink, nn.Softsign,
    nn.Tanhshrink, nn.Hardshrink, nn.Hardsigmoid, nn.Hardswish, nn.Hardtanh,
    nn.SiLU, nn.Mish,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.MaxUnpool1d, nn.MaxUnpool2d, nn.MaxUnpool3d,
    nn.FractionalMaxPool2d, nn.FractionalMaxPool3d,
    nn.LPPool1d, nn.LPPool2d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
)
