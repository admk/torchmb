from typing import Union, Sequence, Mapping, List, OrderedDict, Type

import einops
import torch
from torch import nn, Tensor

from .types import StateDict


class AbstractBatchModule(nn.Module):
    base_class: Type[nn.Module] = nn.Module

    @classmethod
    def from_module(
        cls, module: nn.Module, batch: int
    ) -> 'AbstractBatchModule':
        return cls(batch)

    def __init__(self, batch: int):
        super().__init__()
        self.batch = batch
        self._shared_buffers = []

    @property
    def shared_buffers(self):
        return tuple(self._shared_buffers)

    def register_shared_buffer(
        self, name: str, tensor: Tensor, persistent: bool = True
    ) -> None:
        self._shared_buffers.append(name)
        return self.register_buffer(name, tensor, persistent)

    def load_state_dicts(
        self, state_dict_or_dicts: Union[Sequence[StateDict], StateDict],
        strict: bool = True
    ) -> None:
        state_dict: StateDict = OrderedDict()
        if isinstance(state_dict_or_dicts, Mapping):
            for k, v in state_dict_or_dicts.items():
                if k not in self.shared_buffers:
                    v = einops.repeat(v, '... -> g ...', g=self.batch)
                state_dict[k] = v
        else:
            for k in state_dict_or_dicts[0]:
                values = [d[k] for d in state_dict_or_dicts]
                if k not in self.shared_buffers:
                    values = torch.stack(values)
                else:
                    if strict:
                        if any((v != values[0]).any() for v in values[1:]):
                            raise ValueError(
                                'Shared buffer should have the same values.')
                    values = values[0]
                state_dict[k] = values
        self.load_state_dict(state_dict, strict=strict)

    def state_dicts(self, device=None) -> List[StateDict]:
        state = self.state_dict()
        states = [{} for _ in range(self.batch)]
        for k, v in state.items():
            v = v.to(device) if device is not None else v
            if k in self.shared_buffers:
                for b in range(self.batch):
                    states[b][k] = v
                continue
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

    def extra_repr(self) -> str:
        reprs = self.base_class.extra_repr(self).split(', ')
        reprs.append(f'batch={self.batch}')
        return ', '.join(r for r in reprs if r)
