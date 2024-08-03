from torchmb.base import AbstractBatchModule


class AbstractBatchAdapter(AbstractBatchModule):
    def __init__(self, batch: int):
        super().__init__(batch)
        self.batch = batch

    def forward(self, x):
        raise NotImplementedError
