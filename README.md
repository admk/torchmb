# PyTorch Model Batcher

## Installation

```bash
pip install "git+https://github.com/admk/torchmb.git#egg=torchmb"
```

## Usage

### Model Batching

Common layers are supported. To use, simply instantiate a PyTorch module
and use `torchmb.BatchModule(module, batch)`
to generate a batch of identical models:
```python
from torchmb import BatchModule

model_batch_size = 100
batch_model = BatchModule(LeNet(), batch=model_batch_size)
```

### Forward Passes

For forward passes,
prepare your batch input data `batch_input`
with shape `(model_batch_size, image_batch_size, ...)`,
and use `batch_model` by calling it:
```python
batch_output = batch_mode(batch_input)
```
This computes a `batch_output`
with shape `(model_batch_size, image_batch_size, ...)`.

### Batch Utility Functions

The `torchmb` package also provides batch utility functions
for common top-K and loss functions.
To compute the cross-entropy loss,
prepare a batch of targets
with shape `(model_batch_size, image_batch_size)`,
and use:
```python
from torchmb import batch_loss

loss_func = nn.functional.cross_entropy
losses = batch_loss(
    batch_inputs, batch_targets, self.model_batch, loss_func, 'mean')
```
This computes a batch of loss values `losses`
with shape `(model_batch_size)`.

Similarly, for top-K accuracy evaluation, use:
```python
from torchmb import batch_topk

accs = batch_topk(batch_inputs, batch_targets, self.model_batch, (1, 5))
```
where `accs` is a batch of top-1 and top-5 accuracies
with shape `(2, model_batch_size)`,
and the rows respectively list top-1 and top-5 values.

### Backward Passes

Batched modules and batch utility functions
are fully compatible with automatic differentiation.
To invoke backpropagation on `batch_loss`,
simply use for instance:
```python
batch_loss.sum().backward()
```
The gradients for all batched models
will be independently updated in batch.

### Extending the Model Batcher

To support custom modules,
implement your `MyBatchModule` class
by inheriting from `AbstractBatchModule`
and register it with:
```python
from torch import Tensor
from torchmb import AbstractBatchModule, register_module


class MyBatchModule(AbstractBatchModule):
    def __init__(self, batch: int, ...):
        super().__init__(batch)
        ...

    def forward(self, batch_inputs: Tensor) -> Tensor:
        ...

register_module(MyModule, lambda module, batch: MyBatchModule(...))
```


### Caveats

To ensure isolated training in batched models,
we performed extensive testing in `tests/test_(functional|layers).py`.
However, it is important to note that
to prevent information leakage,
the user is expected to be aware
of how their algorithms can affect model isolation
in forward and backward passes.
For example,
the SGD optimizer (even with momentum or Nesterov)
does not leak information,
but the `AdamW` violates the constraint.

Platform-dependent behaviour, floating-point rounding errors,
and the choice of algorithms used by CuDNN
can all affect the accuracy of the outputs.
Sometimes there may be a non-negligble difference
between the batch outputs and non-batch results.
This is generally not an issue
because in either case it is very difficult to predict
how errors are introduced in the implementation,
and the user has very little control over this.
