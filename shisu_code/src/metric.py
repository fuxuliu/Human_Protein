from lightai.core import *


class F1:
    def __init__(self, threshold=0.5):
        self.predicts = []
        self.targets = []
        self.threshold = threshold

    def __call__(self, predict, target):
        """
        predict and target are in batch
        """
        self.predicts.append(predict)
        self.targets.append(target)

    def res(self):
        origin_predict = torch.cat(self.predicts).sigmoid()
        target = torch.cat(self.targets)
        scores = []
        if self.threshold is not None:
            predict = (origin_predict > self.threshold).float()
            tp = (predict*target).sum(dim=0)  # shape (28,)
            precision = tp/(predict.sum(dim=0) + 1e-8)
            recall = tp/(target.sum(dim=0) + 1e-8)
            f1 = 2*(precision*recall/(precision+recall+1e-8))
            self.predicts = []
            self.targets = []
            return f1.mean().item()
        for threshold in np.linspace(0, 1, num=100, endpoint=False):
            predict = (origin_predict > threshold).float()
            tp = (predict*target).sum(dim=0)  # shape (28,)
            precision = tp/(predict.sum(dim=0) + 1e-8)
            recall = tp/(target.sum(dim=0) + 1e-8)
            f1 = 2*(precision*recall/(precision+recall+1e-8))
            scores.append(f1)
        scores = torch.stack(scores)
        self.predicts = []
        self.targets = []
        return scores.max(dim=0)[0].mean().item()
