from ..core import *
from ..callback import *
from .save_best_model import *


class Cyclic(Callback):
    def __init__(self, scheduler, sv_best):
        self.scheduler = scheduler
        self.epoch = 0
        self.cycle = 0
        self.sv_best = sv_best

    def on_epoch_begin(self, **kwargs):
        self.scheduler.step(self.epoch)
        if self.epoch == 0:
            self.sv_best.best_metrics = None
            self.sv_best.path = self.sv_best.model_dir/f'cycle{self.cycle}'
        self.epoch += 1
        if self.epoch == self.scheduler.T_max:
            self.epoch = 0
            self.cycle += 1