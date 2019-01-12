from ..core import *
from ..callback import *


class SaveBestModel(Callback):
    def __init__(self, learner, small_better: bool, model_dir: str = 'saved',
                 name: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True, parents=True)
        if name:
            self.path = self.model_dir/name
        self.small_better = small_better
        self.best_metrics = None
        self.learner = learner

    def on_epoch_end(self, eval_res: List[float], **kwargs):
        metrics = eval_res[-1]
        if self.small_better:
            metrics = -metrics
        if not self.best_metrics or metrics >= self.best_metrics:
            self.best_metrics = metrics
            torch.save({
                'model': self.learner.model.state_dict(),
                'optimizer': self.learner.optimizer.state_dict()
            }, self.path)

    def on_train_end(self, **kwargs):
        if self.best_metrics is None:
            return
        best = -self.best_metrics if self.small_better else self.best_metrics
        print(f'best metric: {best:.6f}')
