from lightai.core import *


def f1_loss(predict, target):
    loss = 0
    # fp = predict[target == 0]
    # loss += ((fp.exp()+1).log()*fp.sigmoid()**2).mean()
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += F.binary_cross_entropy_with_logits(
            predict[:, lack_cls], target[:, lack_cls])
    # loss += predict[:, target.sum(dim=0) == 0].mean()
    predict = torch.sigmoid(predict)
    predict = torch.clamp(predict * (1-target), min=0.01) + predict * target
    tp = predict * target
    tp = tp.sum(dim=0)
    precision = tp / (predict.sum(dim=0) + 1e-8)
    recall = tp / (target.sum(dim=0) + 1e-8)
    f1 = 2 * (precision * recall / (precision + recall + 1e-8))
    return 1 - f1.mean() + loss
