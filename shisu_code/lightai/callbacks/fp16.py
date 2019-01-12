from ..core import *
from ..callback import *


def get_params(learner):
    model_param_groups = learner.param_groups
    master_param_groups = []
    for pg in model_param_groups:
        master_param_group = [param.detach().clone().float()
                              for param in pg['params']]
        master_param_groups.append({'params': master_param_group})
    return model_param_groups, master_param_groups


def bn_to_float(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        bn_to_float(child)


class FP16(Callback):
    def __init__(self, model_param_groups, master_param_groups, loss_scale):
        self.model_param_groups = model_param_groups
        self.master_param_groups = master_param_groups
        self.loss_scale = loss_scale

    def on_train_begin(self, **kwargs):
        for model_group, master_group in zip(self.model_param_groups, self.master_param_groups):
            for model, master in zip(model_group['params'], master_group['params']):
                master.data.copy_(model.data)

    def on_backward_begin(self, loss, **kwargs):
        loss *= self.loss_scale

    def on_backward_end(self, loss, **kwargs):
        loss /= self.loss_scale

    def on_step_begin(self, **kwargs):
        for model_group, master_group in zip(self.model_param_groups, self.master_param_groups):
            for model, master in zip(model_group['params'], master_group['params']):
                if not model.requires_grad:
                    continue
                if master.grad is None:
                    master.grad = master.detach().clone()
                master.grad.data.copy_(model.grad.data)
                master.grad.data /= self.loss_scale

    def on_step_end(self, **kwargs):
        for model_group, master_group in zip(self.model_param_groups, self.master_param_groups):
            for model, master in zip(model_group['params'], master_group['params']):
                if not model.requires_grad:
                    continue
                model.data.copy_(master.data)
                model.grad.data.zero_()


def to_fp16(learner, loss_scale):
    model = learner.model
    model.half()
    bn_to_float(model)
    model_param_groups, master_param_groups = get_params(learner)
    learner.optimizer = learner.optim_fn(master_param_groups)
    learner.callbacks.append(
        FP16(model_param_groups, master_param_groups, loss_scale))
