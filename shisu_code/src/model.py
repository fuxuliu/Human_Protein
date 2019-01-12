"""
The code and idea behind it in this file mainly stole from fastai. For better code, greater ideas and amazing free courses, definitely
goto https://www.fast.ai/
"""


from lightai.core import *


# mean = T(np.array(
#     [25.8022, 14.9129, 15.5262, 20.3930]).reshape((-1, 1, 1))).half()
# std = T(np.array([41.7929, 29.0815, 42.3371, 31.1343]).reshape(
#     (-1, 1, 1))).half()

mean = T(np.array(
    [24.8897, 14.2778, 15.0474, 19.7499]).reshape((-1, 1, 1))).half()
std = T(np.array([41.5745, 28.6833, 41.9189, 31.0968]).reshape(
    (-1, 1, 1))).half()


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        return x


def bn_drop_lin(n_in: int, n_out: int, bn: bool = True, p: float = 0., actn: Optional[nn.Module] = None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0:
        layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None:
        layers.append(actn)
    return layers


def create_head(nf: int, nc: int, lin_ftrs: Optional[Collection[int]] = None, ps=[0.5],
                bn_final: bool = False):
    "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
    lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
    if len(ps) == 1:
        ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
    actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
    layers = [AdaptiveConcatPool2d(), Flatten()]
    for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
        layers += bn_drop_lin(ni, no, True, p, actn)
    if bn_final:
        layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
    return nn.Sequential(*layers)


class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."

    def __init__(self, sz: Optional[int] = None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Model(nn.Module):
    def __init__(self, base=torchvision.models.resnet18, pretrained=True):
        super().__init__()
        # self.inp_bn = nn.BatchNorm2d(4)
        self.base = self.get_base(base, pretrained)
        self.head = create_head(1024, 28, ps=[0])

    def forward(self, x):
        x = x.half()
        x = x.permute(0, 3, 1, 2)
        x = (x-mean)/std
        # x = self.inp_bn(x)
        x = self.base(x)
        x = self.head(x)
        # x = torch.sigmoid(x)
        return x

    def get_base(self, base, pretrained):
        resnet = base(pretrained=pretrained)
        conv1 = nn.Conv2d(4, 64, kernel_size=(
            7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        conv1.weight.data[:, :-1] = resnet.conv1.weight.data
        conv1.weight.data[:, -1] = resnet.conv1.weight.data.mean(dim=1)
        resnet.conv1 = conv1
        return nn.Sequential(*list(resnet.children())[:-2])
