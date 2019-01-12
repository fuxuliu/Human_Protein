from ..core import *
from ..callback import *


class LrScheduler(Callback):
    def __init__(self, optimizer, lrs: Sequence[float], lr_ratio: Sequence[float]=None):
        self.optimizer = optimizer
        self.lrs = lrs
        self.iter = 0
        self.lr_ratio = [1]*len(self.optimizer.param_groups) if lr_ratio is None else lr_ratio

    def on_batch_begin(self, **kwargs):
        param_groups_lrs = self.lr_span_param_groups(self.lrs[self.iter])
        for param_group, lr in zip(self.optimizer.param_groups, param_groups_lrs):
            param_group['lr'] = lr
        self.iter += 1

    def lr_span_param_groups(self, lr: float)->List[float]:
        lrs = []
        for ratio in self.lr_ratio:
            lrs.append(ratio * lr)
        return lrs
