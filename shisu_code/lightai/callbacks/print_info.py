from ..core import *
from ..callback import *


class Printer(Callback):
    def __init__(self, metrics):
        self.layout = '{:^11}' + '{:11.6f}' * (2 + len(metrics))
        self.metrics = metrics

    def on_train_begin(self, mb, **kwargs):
        names = [metric.__class__.__name__ for metric in self.metrics]
        names = ['epoch', 'train_loss', 'val_loss'] + names
        names_layout = '{:^11}' * len(names)
        names = [name[:10] for name in names]
        mb.write(names_layout.format(*names))

    def on_epoch_end(self, trn_loss: float, eval_res: List[float], epoch: int,
                     mb, **kwargs):
        mb.write(self.layout.format(epoch, trn_loss, *eval_res))