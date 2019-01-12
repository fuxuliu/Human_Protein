from ..callback import *

class ReduceOnPlateau(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, metrics: float, **kwargs):
        self.scheduler.step(metrics)


class LRSchedWrapper(Callback):
    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_begin(self, **kwargs):
        self.scheduler.step()