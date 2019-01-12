from .callbacks import *
from .core import *


class Learner:
    def __init__(self, model: nn.Module, trn_dl: DataLoader, val_dl: DataLoader,
                 optim_fn: optim.Optimizer, loss_fn: Callable, metrics: List,
                 callbacks: List[Callback] = [], writer: Optional[SummaryWriter] = None,
                 layer_groups: List[List[nn.Module]] = None):
        self.model = model
        self.trn_dl = trn_dl
        self.val_dl = val_dl
        self.optim_fn = optim_fn
        self.layer_groups = [model] if layer_groups is None else layer_groups
        if layer_groups is None:
            param_groups = [{'params': model.parameters()}]
        else:
            param_groups = []
            for layer in layer_groups:
                param_group = []
                for m in layer:
                    param_group.extend(m.parameters())
                param_groups.append({'params': param_group})
        self.param_groups = param_groups
        self.optimizer = optim_fn(self.param_groups)
        self.loss_fn = loss_fn
        self.callbacks = callbacks
        self.writer = writer
        self.epoch = 0
        self.metrics = metrics
        self.callbacks.append(Printer(metrics))
        self.callbacks.append(Logger(writer=self.writer, metrics=metrics))

    def fit(self, n_epoch: Optional[int] = None, sched: Optional[Callback] = None):
        callbacks = self.callbacks + [sched]
        mb = master_bar(range(n_epoch))
        for cb in callbacks:
            cb.on_train_begin(mb=mb)
        try:
            for epoch in mb:
                self.train()
                for cb in callbacks:
                    cb.on_epoch_begin()
                losses = []
                for x, target in progress_bar(self.trn_dl, parent=mb):
                    x, target = x.cuda(non_blocking=True), target.cuda(
                        non_blocking=True)
                    for cb in callbacks:
                        cb.on_batch_begin(x=x, target=target)
                    trn_loss = self.step(x, target)
                    losses.append(trn_loss)
                    for cb in callbacks:
                        stop = cb.on_batch_end(trn_loss=trn_loss)
                        if stop:
                            return
                trn_loss = np.mean(losses)
                eval_res = self.evaluate()
                self.epoch += 1
                for cb in callbacks:
                    cb.on_epoch_end(trn_loss=trn_loss, eval_res=eval_res, epoch=self.epoch,
                                    learner=self, mb=mb)
        finally:
            for cb in callbacks:
                cb.on_train_end()

    def step(self, x: np.ndarray, target: np.ndarray) -> float:
        predict = self.model(x)
        predict = predict.float()
        loss = self.loss_fn(predict, target)
        self.optimizer.zero_grad()
        for cb in self.callbacks:
            cb.on_backward_begin(loss=loss)
        loss.backward()
        for cb in self.callbacks:
            cb.on_backward_end(loss=loss)
        for cb in self.callbacks:
            cb.on_step_begin()
        self.optimizer.step()
        for cb in self.callbacks:
            cb.on_step_end()
        return loss.item()

    def evaluate(self):
        self.model.eval()
        losses = []
        bses = []
        with torch.no_grad():
            for x, target in self.val_dl:
                x, target = x.cuda(non_blocking=True), target.cuda(
                    non_blocking=True)
                predict = self.model(x)
                predict = predict.float()
                for metric in self.metrics:
                    metric(predict, target)
                losses.append(self.loss_fn(predict, target))
                bses.append(target.shape[0])
            loss = np.average(torch.stack(losses).cpu().numpy(), weights=bses)
            res = [loss] + [metric.res() for metric in self.metrics]
            return res

    def freeze_to(self, n):
        for layer in self.layer_groups[:n]:
            for m in layer:
                for p in m.parameters():
                    p.requires_grad = False
                apply_leaf(m, freeze_bn, freeze=True)
        for layer in self.layer_groups[n:]:
            for m in layer:
                for p in m.parameters():
                    p.requires_grad = True
                apply_leaf(m, freeze_bn, freeze=False)

    def unfreeze(self):
        self.freeze_to(0)

    def train(self):
        self.model.train()
        apply_leaf(self.model, set_freezed_bn_eval)


def freeze_bn(module, freeze):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.freeze_bn = freeze


def set_freezed_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and getattr(module, 'freeze_bn', False):
        module.eval()
