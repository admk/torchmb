import unittest
from typing import Any

import torch
from torch import Tensor


class TestBase(unittest.TestCase):
    rtoi = 1e-7
    atoi = 1e-7

    def assertAllClose(self, a: Tensor, b: Tensor, msg: Any = None) -> None:
        self.assertTrue(torch.allclose(a, b, self.rtoi, self.atoi), msg)

    def grad(self, t: Tensor) -> Tensor:
        self.assertIsNotNone(t.grad)
        return t.grad
