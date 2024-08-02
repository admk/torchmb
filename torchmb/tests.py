import copy
from typing import Optional, Sequence

import torch
from torch import nn, Tensor

from .types import Size
from .base import AbstractBatchModule
from .batch import BatchModule


def assert_all_close(
    a: Tensor, b: Tensor, rtol: float = 1e-6, atol: float = 1e-6
) -> Optional[str]:
    if a.size() != b.size():
        return f'Size mismatch, "{a.size()} != "{b.size()}".'
    if not torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
        return (
            f'Values mismatch, (|a - b|={(a - b).abs().max()}) > '
            f'({rtol=}) * (|b|={b.abs().max()}) + ({atol=}).')


def test_batched_model(
    batched_model: AbstractBatchModule, model: nn.Module,
    input_shapes: Sequence[Size], rtol: float = 1e-6, atol: float = 1e-6,
    randomize: bool = True, restore_state: bool = True
) -> None:
    batch = batched_model.batch
    if restore_state:
        training = model.training
        trainings = batched_model.training
        old_state = copy.deepcopy(model.state_dict())
        old_states = copy.deepcopy(batched_model.state_dicts())
    else:
        training = trainings = old_state = old_states = None
    model.eval()
    batched_model.eval()
    if randomize:
        state = model.state_dict()
        states = []
        for _ in range(batch):
            rand_state = {}
            for k, v in state.items():
                r = torch.randn_like(v)
                rand_state[k] = r / r.norm() * v.norm()
            states.append(rand_state)
        batched_model.load_state_dicts(states)
    else:
        states = None
    device = next(model.parameters()).device
    batch_inputs = [
        torch.randn((batched_model.batch, ) + tuple(shape), device=device)
        for shape in input_shapes]
    batch_outputs = batched_model(*batch_inputs)
    outputs = []
    for i, inputs in enumerate(zip(*batch_inputs)):
        if states is not None:
            model.load_state_dict(states[i])
        outputs.append(model(*inputs))
    try:
        if isinstance(batch_outputs, Tensor):
            assert_all_close(batch_outputs, torch.stack(outputs), rtol, atol)
        else:
            for bo, o in zip(batch_outputs, zip(*outputs)):
                assert_all_close(bo, torch.stack(o), rtol, atol)
    except AssertionError as e:
        raise AssertionError(
            'Batched model failed independent model test.') from e
    if training is not None:
        model.train(training)
    if trainings is not None:
        batched_model.train(trainings)
    if old_state is not None:
        model.load_state_dict(old_state)
    if old_states is not None:
        batched_model.load_state_dicts(old_states)


def test_model_batching(
    model: nn.Module, input_shapes: Sequence[Size], batch: int = 3,
    rtol: float = 1e-6, atol: float = 1e-6
) -> None:
    batch_model = BatchModule(model, batch)
    test_batched_model(batch_model, model, input_shapes, rtol, atol)
