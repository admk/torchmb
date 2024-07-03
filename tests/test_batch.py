import torch
from torch import nn

from torchmb.base import AbstractBatchModule
from torchmb.batch import register_batch_module, test_batch_module, BatchModule

from tests.base import TestBase


class TestBatch(TestBase):
    def setUp(self) -> None:
        self.batch = 5
        self.input_shape = (5, 3, 16, 16)

    def test_register_module(self):
        class Dummy(nn.Module):
            def forward(self, x):
                return torch.zeros_like(x)

        class BatchDummy(AbstractBatchModule):
            base_class = Dummy

            @classmethod
            def from_module(cls, module, batch):
                return cls(batch)

            def forward(self, x):
                return torch.zeros_like(x)

        register_batch_module(BatchDummy)
        module = Dummy()
        batch_module1 = BatchDummy.from_module(module, self.batch)
        batch_module2 = BatchModule(module, self.batch)
        x = torch.ones(self.input_shape)
        self.assertAllClose(batch_module1(x), batch_module2(x))

    def test_test_mimo_module(self):
        class MIMO(nn.Module):
            def forward(self, x, y):
                return x + y, x - y

        class BatchMIMO(AbstractBatchModule):
            base_class = MIMO

            @classmethod
            def from_module(cls, module, batch):
                return cls(batch)

            def forward(self, x, y):
                return x + y, x - y

        register_batch_module(BatchMIMO)
        module = MIMO()
        batch_module = BatchMIMO.from_module(module, self.batch)
        x = torch.randn(self.input_shape)
        y = torch.randn(self.input_shape)
        u, v = batch_module(x, y)
        self.assertAllClose(u, x + y)
        self.assertAllClose(v, x - y)

    def test_test_batch_module(self):
        module = nn.Conv2d(3, 32, 3)
        test_batch_module(module, (self.input_shape, ), self.batch, atol=1e-5)
