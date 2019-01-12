from torchvision.models import *
from torch import nn
from config import *
from collections import OrderedDict
import torch.nn.functional as F
import torch
import pretrainedmodels
from Nadam import Nadam
from utils import *
from sync_models import sysc_model,DataParallelWithCallback,patch_replication_callback

def resnet18_model(cfg):
    model = resnet18(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
                nn.Linear(512,28),
                nn.Sigmoid())
    main_params  = list(map(id, model.fc.parameters()))
    main_params += list(map(id, model.conv1.parameters()))
    main_params += list(map(id, model.bn1.parameters()))
    base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.fc.parameters()},
        {'params': model.conv1.parameters()},
        {'params': model.bn1.parameters()},
    ], lr=cfg['lr'])
    return model,opt


def resnet34_model(cfg):
    model = resnet34(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
                nn.Linear(512,28),
                nn.Sigmoid())
    main_params  = list(map(id, model.fc.parameters()))
    main_params += list(map(id, model.conv1.parameters()))
    main_params += list(map(id, model.bn1.parameters()))
    base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.fc.parameters()},
        {'params': model.conv1.parameters()},
        {'params': model.bn1.parameters()},
    ], lr=cfg['lr'])
    return model,opt

def resnet50_model(cfg):
    model = resnet50(pretrained=True)
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3, 3), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Sequential(
                nn.Linear(2048,28),
                nn.Sigmoid())
    if len(cfg['device']) > 1:
        model = sysc_model(model)
        model = DataParallelWithCallback(model, device_ids=cfg['device'])

        main_params = list(map(id, model.module.fc.parameters()))
        main_params += list(map(id, model.module.conv1.parameters()))
        main_params += list(map(id, model.module.bn1.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.module.fc.parameters()},
            {'params': model.module.conv1.parameters()},
            {'params': model.module.bn1.parameters()},
        ], lr=cfg['lr'])
    else:
        main_params  = list(map(id, model.fc.parameters()))
        main_params += list(map(id, model.conv1.parameters()))
        main_params += list(map(id, model.bn1.parameters()))
        base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.fc.parameters()},
            {'params': model.conv1.parameters()},
            {'params': model.bn1.parameters()},
        ], lr=cfg['lr'])
    return model,opt

def se_resnext50(cfg):
    model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet', num_classes=1000)
    w = model.layer0.conv1.weight
    model.layer0.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.layer0.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
        nn.Linear(2048, 28),
        nn.Sigmoid()
    )
    if len(cfg['device']) > 1:
        model = sysc_model(model)
        model = DataParallelWithCallback(model, device_ids=cfg['device'])
        # model = nn.DataParallel(model, device_ids=cfg['device'])
        main_params = list(map(id, model.module.last_linear.parameters()))
        main_params += list(map(id, model.module.layer0.conv1.parameters()))
        main_params += list(map(id, model.module.layer0.bn1.parameters()))
        base_params = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.module.last_linear.parameters()},
            {'params': model.module.layer0.conv1.parameters()},
            {'params': model.module.layer0.bn1.parameters()},
        ], lr=cfg['lr'])
    else:
        main_params =  list(map(id, model.last_linear.parameters()))
        main_params += list(map(id, model.layer0.conv1.parameters()))
        main_params += list(map(id, model.layer0.bn1.parameters()))
        base_params  = filter(lambda p: id(p) not in main_params, model.parameters())
        opt = Nadam([
            {'params': base_params, 'lr': cfg['backbone_lr']},
            {'params': model.last_linear.parameters()},
            {'params': model.layer0.conv1.parameters()},
            {'params': model.layer0.bn1.parameters()},
        ], lr=cfg['lr'])
    return model, opt

def xception_model(cfg):
    model = pretrainedmodels.models.xception(pretrained='imagenet')
    w = model.conv1.weight
    model.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    model.conv1.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.last_linear = nn.Sequential(
                nn.Linear(2048, 28),
                nn.Sigmoid()
            )
    
    return model

def bninception_model(cfg):
    model = pretrainedmodels.models.bninception(pretrained='imagenet')
    w = model.conv1_7x7_s2.weight
    model.conv1_7x7_s2 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.conv1_7x7_s2.weight = torch.nn.Parameter(torch.cat((w, w[:, :1, :, :]), dim=1))
    model.global_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.Dropout(0.5),
                nn.Linear(1024, 28),
                nn.Sigmoid()
            )
    main_params = list(map(id, model.last_linear.parameters()))
    main_params += list(map(id, model.conv1_7x7_s2.parameters()))
    main_params += list(map(id, model.conv1_7x7_s2_bn.parameters()))
    base_params = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.last_linear.parameters()},
        {'params': model.conv1_7x7_s2.parameters()},
        {'params': model.conv1_7x7_s2_bn.parameters()},
    ], lr=cfg['lr'])
    return model, opt

class weightedBCELoss(nn.Module):

    def __init__(self,alpha):
        super(weightedBCELoss, self).__init__()
        self.alpha = alpha
    def forward(self, input, target, dim=None):
        loss = F.binary_cross_entropy(input, target, weight=None, reduction='none')
        loss = self.alpha * target *loss + (2-self.alpha) * (1-target) *loss
        if dim:
            return torch.mean(loss,dim)
        else:
            return torch.mean(loss)


if __name__ == '__main__':
    # print(resnet_model(None))
    cfg = {}
    m = bninception_model(cfg)

    print(m)























