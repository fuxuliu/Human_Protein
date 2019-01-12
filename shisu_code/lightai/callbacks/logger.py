from ..core import *
from ..callback import *


class Logger(Callback):
    def __init__(self, writer: SummaryWriter, metrics: List):
        self.writer = writer
        self.metrics_names = [metric.__class__.__name__ for metric in metrics]

    def on_epoch_end(self, trn_loss: float, eval_res: List[float], epoch: int, **kwargs):
        if self.writer:
            self.writer.add_scalars('loss', {
                'train': trn_loss,
                'val': eval_res[0]
            }, epoch)
            for i, name in enumerate(self.metrics_names):
                self.writer.add_scalar(name, eval_res[i+1], epoch)
