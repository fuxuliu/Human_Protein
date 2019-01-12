from ..core import *
from ..callback import *


class SavePeriodically(Callback):
    def __init__(self, period: int, name: str='saved'):
        self.period = period
        self.path = name

    def on_epoch_end(self, epoch, learner, **kwargs):
        if epoch % self.period == 0:
            torch.save(learner, self.path, dill)