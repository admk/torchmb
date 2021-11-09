from typing import Union, Sequence, Mapping, List, Dict, OrderedDict

import einops
import torch
from torch import nn


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
