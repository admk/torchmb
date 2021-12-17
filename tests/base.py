import unittest

from torch import Tensor


class TestBase(unittest.TestCase):
    rtoi = 1e-7
    atoi = 1e-7

    def grad(self, t: Tensor) -> Tensor:
        self.assertIsNotNone(t.grad)
        return t.grad
