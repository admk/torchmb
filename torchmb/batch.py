import copy
from typing import (
    Optional, Callable, Union, Dict, OrderedDict, Iterator, Tuple, List)

from torch import nn, fx, Tensor

from .types import (
    StateDict, DataOrder, TensorOrTensors,
    ELEMENTWISE_FUNCS, PARAMETER_FREE_ELEMENTWISE_MODULES,
    BATCH_DEPENDENT_MODULES)
from .base import AbstractBatchModule
from .functional import merge_batch, split_batch
from .utils import to_batch_func, BATCH_MODULES


class BatcherTracer(fx.Tracer):
    leaf_types = tuple(BATCH_MODULES) + BATCH_DEPENDENT_MODULES

    def symbolic_trace(self, module: nn.Module) -> fx.GraphModule:
        graph = super().trace(module)
        return fx.GraphModule(module, graph, module.__class__.__name__)

    def is_leaf_module(
        self, m: nn.Module, module_qualified_name: str
    ) -> bool:
        if any(isinstance(m, module_cls) for module_cls in self.leaf_types):
            return True
        return super().is_leaf_module(m, module_qualified_name)


class BatchModule(AbstractBatchModule):
    def __init__(
        self, model: nn.Module, batch: int = 1, inplace: bool = True,
        replace_func: Optional[Callable] = None,
    ):
        super().__init__(batch)
        self._replace_func = replace_func
        self._module = self._create_batch_module(model, inplace)
        self.load_state_dicts(model.state_dict())

    def to_batch_func(self, node: fx.Node, batch: int) -> Callable:
        return to_batch_func(node, batch)

    def to_batch_module(
        self, node: fx.Node, module: nn.Module, batch: int
    ) -> Union[nn.Module, AbstractBatchModule]:
        if isinstance(module, PARAMETER_FREE_ELEMENTWISE_MODULES):
            return module
        if not isinstance(module, BATCH_DEPENDENT_MODULES):
            if all(not p.requires_grad for p in module.parameters()):
                # Skip non-learnable batch-independent modules
                return module
        try:
            batch_func = BATCH_MODULES[type(module)]
        except KeyError as e:
            raise NotImplementedError(
                f'Batch module for {type(module)} is not registered.') from e
        return batch_func.from_module(module, batch)

    def _match_replace(
        self, node: fx.Node, modules: Dict[str, fx.GraphModule],
        shared_buffers: List[str]
    ) -> None:
        if len(node.args) == 0:
            return
        if node.op == 'call_function':
            if node.target in ELEMENTWISE_FUNCS:
                return
            try:
                setattr(node, 'target', self.to_batch_func(node, self.batch))
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
            batch_module = self.to_batch_module(node, module, self.batch)
        except KeyError:
            return
        *parent, name = node.target.rsplit('.', 1)
        parent_name = parent[0] if parent else ''
        setattr(modules[parent_name], name, batch_module)
        if hasattr(batch_module, 'shared_buffers'):
            for n in batch_module.shared_buffers:
                shared_buffers.append(f'{node.target}.{n}')

    def _create_batch_module(
        self, model: nn.Module, inplace: bool = True,
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

    def load_state_dict(  # type: ignore
        self, state_dict: StateDict, strict: bool = True
    ) -> None:
        self._module.load_state_dict(OrderedDict(state_dict), strict=strict)

    def state_dict(self) -> StateDict:  # type: ignore
        return self._module.state_dict()

    def named_parameters(  # type: ignore
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
