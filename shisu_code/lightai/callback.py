from .core import *


class Callback:
    def on_train_begin(self, **kwargs):
        pass

    def on_batch_begin(self, **kwargs):
        pass

    def on_epoch_begin(self, **kwargs):
        pass

    def on_batch_end(self, **kwargs):
        pass

    def on_epoch_end(self, **kwargs):
        pass

    def on_train_end(self, **kwargs):
        pass

    def on_backward_begin(self, **kwargs):
        pass

    def on_backward_end(self, **kwargs):
        pass

    def on_step_begin(self, **kwargs):
        pass

    def on_step_end(self, **kwargs):
        pass
