import argparse
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import f1_score
from glob import glob
from models import *
from utils import *
from tqdm import tqdm
import pandas as pd
from Nadam import Nadam
from torch.utils.data import DataLoader
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch.optim import lr_scheduler
from sync_models import *
from torch import nn
torch.backends.cudnn.benchmark = True

def train(fold,tr_data,val_data,cfg):

    val_gen = HumanDataset(val_data, 'test', cfg, None, None)
    val_loader = DataLoader(val_gen, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=4)

    tr_gen = HumanDataset(tr_data, 'train', cfg, None, get_augumentor('train'))
    tr_loader = DataLoader(tr_gen, batch_size=cfg['bs'], shuffle=True, pin_memory=True, num_workers=4)

    get_loss = nn.BCELoss().cuda()
    # get_loss = weightedBCELoss(cfg['alpha']).cuda()

    model = cfg['model'](cfg)
    model = nn.DataParallel(model)

    main_params = list(map(id, model.module.last_linear.parameters()))
    main_params += list(map(id, model.module.bn1.parameters()))
    main_params += list(map(id, model.module.conv1.parameters()))
    base_params = filter(lambda p: id(p) not in main_params, model.parameters())
    opt = Nadam([
        {'params': base_params, 'lr': cfg['backbone_lr']},
        {'params': model.module.last_linear.parameters()},
        {'params': model.module.conv1.parameters()},
        {'params': model.module.bn1.parameters()},
    ], lr=cfg['lr'])

    model.cuda()

    scheduler = lr_scheduler.MultiStepLR(opt,cfg['milestones'],gamma=cfg['gamma'])

    f1_best = np.float('-inf')
    best_i = 0
    for i in range(1000):
        if i - best_i > 6  and i > 20:
            break
        print(f'epoch {i}')
        scheduler.step(i)
        model.train()
        opt.zero_grad()
        for curr_iter, (images, target) in enumerate(tr_loader):

            images = images.cuda(non_blocking=True)
            target = torch.from_numpy(np.array(target)).float().cuda(non_blocking=True)

            output = model(images)

            loss = get_loss(output, target)/cfg['M']
            # if loss.size(0) == cfg['bs']:
            #     loss = loss.mean(1)
            #     loss, _ = loss.topk(k=int(loss.size(0) * cfg['ohem_keep']))
            # loss = loss.mean()
            loss.backward()

            if (curr_iter+1) % cfg['M'] == 0:
                opt.step()
                opt.zero_grad()

            if curr_iter % 200 == 0:
                print(loss)


        if True: # evaulate
            y, pred = predict(model,val_loader)

            y = np.array(y)

            # for i in np.arange()
            label = (np.array(pred) > cfg['thres']).astype(int)


            f1 = f1_score(y,label, average='macro')
            if f1_best < f1:
                f1_best = f1
                best_i = i
                print(f'f1_score{f1}, improved save model.......')
                torch.save(model.state_dict(), f"./weights/{cfg['name']}_fold{fold}.pkl")
            else:
                print(f'f1_score{f1}, best f1_score{f1_best}')


def main(cfg):
    data_dir = './input/'
    dataset = pd.read_csv(data_dir+'train.csv')
    dataset['Id'] = data_dir + 'train/' + dataset['Id']
    dataset['suffix'] = '.png'
    if cfg['use_external_data']:
        ex_data = pd.read_csv(data_dir+'HPAv18RBGY_wodpl.csv')
        ex_data['Id'] = data_dir + 'HPAv18_512/' + ex_data['Id']
        ex_data['suffix'] = '.jpg'
        dataset = dataset.append(ex_data).reset_index(drop=True)

    print(cfg)

    target = np.zeros((len(dataset), 28))
    for i, labels in enumerate(dataset['Target']):
        labels = [int(t) for t in labels.split() if t != '']
        for l in labels:
            target[i, l] = 1
    # folds = KFold(len(dataset),cfg['nfold'],shuffle=True,random_state=66666)
    folds = MultilabelStratifiedKFold(cfg['nfold'],shuffle=True,random_state=66666)

    for n_fold, (tr_idx, val_idx) in enumerate(folds.split(dataset['Id'],target)):
        print(val_idx)

        if n_fold not in cfg['fold']:
            continue

        print(n_fold,'-----------------')
        tr_data = dataset.iloc[tr_idx]
        val_data = dataset.iloc[val_idx]



        train(n_fold,tr_data,val_data,cfg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('--fold', type=str, required=True)
    # parser.add_argument('--device', type=str, required=True)


    from config import *

    args = parser.parse_args()
    cfg = {}
    # cfg['device'] = [i for i in range(len(args.device))]
    cfg['nfold'] = 5
    cfg['fold'] = [int(fold) for fold in range(5)]

    cfg['channel'] = 4
    cfg['aug'] = True
    cfg['thres'] = 0.3
    cfg['use_external_data'] = True
    cfg['alpha'] = 1.0

    cfg['milestones'] = [7,13,18,23,30]
    # cfg['milestones'] = [7, 14, 20, 25]
    # cfg['milestones'] = [7,13,18]
    cfg['gamma'] = 0.5

    cfg['lr'] = 0.0004
    cfg['backbone_lr'] = cfg['lr']/2
    cfg['bs'] = 24
    cfg['M'] = 1
    cfg['name'] = 'xception'
    cfg['model'] = xception_model
    cfg['ohem_keep'] = 0.8
    main(cfg)



