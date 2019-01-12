from torch.utils.data import Dataset
from torchvision import transforms as T
from imgaug import augmenters as iaa
import numpy as np
import random
from tqdm import tqdm
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from models import *
import torch
from PIL import Image
# set random seed
random.seed(66666)
np.random.seed(66666)
torch.manual_seed(66666)
torch.cuda.manual_seed_all(66666)



def get_augumentor(mode):
    if mode =='train':
        return iaa.SomeOf(n=(1,6),children=[
                iaa.Noop(),
                iaa.Sequential([
                    iaa.Add((-5,5),per_channel=True),
                    iaa.Multiply((0.8,1.2),per_channel=True)
                ]),
                iaa.Crop(percent=(0,0.15)),
                iaa.Affine(shear=(-16, 16)),
                iaa.OneOf([
                    iaa.Affine(rotate=90),
                    iaa.Affine(rotate=180),
                    iaa.Affine(rotate=270),
                    iaa.Fliplr(1),
                    iaa.Flipud(1),
                ])
            ])
    elif mode == 'TTA1':
        return iaa.Flipud(1)
    elif mode == 'TTA2':
        return iaa.Fliplr(1)
    elif mode == 'TTA3':
        return iaa.Affine(rotate=90)
    elif mode == 'TTA4':
        return iaa.Affine(rotate=180)
    elif mode == 'TTA5':
        return iaa.Affine(rotate=270)
    else:
        raise ValueError("aug error")


class HumanDataset(Dataset):
    def __init__(self, df, mode, cfg, info, augmentor):

        self.images_df = df.copy()
        self.cfg = cfg
        self.info = info
        self.images_id = df['Id'].values
        self.suffix = df['suffix'].values
        self.mode = mode

        assert mode in ['train','test']
        target = np.zeros((len(df),28))
        for i,labels in enumerate(df['Target']):
            labels = [int(t) for t in labels.split() if t!='']
            for l in labels:
                target[i,l] = 1
        self.target = target

        self.augumentor = augmentor

        self.mean = np.array([12.74,11.28,5.96,22.42]).reshape((1,1,4))
        self.std = np.array([19.58,17.93,15.21,33.29]).reshape((1,1,4))

        print(target.shape)

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, index):

        X = self.read_images(index)

        if self.augumentor:
            X = self.augumentor.augment_image(X)

        X = np.transpose(X, axes=(2, 0, 1))
        X = X.astype(np.float32)

        return torch.FloatTensor(X),self.target[index]

    def read_images(self, index):

        img_path = self.images_id[index]
        suffix = self.suffix[index]

        r = np.array(Image.open(img_path + "_red" + suffix))
        g = np.array(Image.open(img_path + "_green" + suffix))
        b = np.array(Image.open(img_path + "_blue" + suffix))

        if self.cfg['channel'] == 4:
            y = np.array(Image.open(img_path + "_yellow" + suffix))
            images = np.zeros(shape=(512, 512, 4),dtype=np.uint8)
            images[:, :, 3] = y.astype(np.uint8)
        else:
            images = np.zeros(shape=(512, 512, 3))

        images[:, :, 0] = r.astype(np.uint8)
        images[:, :, 1] = g.astype(np.uint8)
        images[:, :, 2] = b.astype(np.uint8)

        return images


def predict(model,data_loader):
    with torch.no_grad():  # evaulate
        model.eval()
        pred = []
        y = []
        for _, (images, target) in enumerate(data_loader):
            images_var = images.cuda(non_blocking=True)

            output = model(images_var)
            pred += output.cpu().data.tolist()
            y += target.tolist()
    return y,pred


def get_best_thres(y_true_all,y_pred_all,num_split=100):

    num_cate = y_true_all.shape[1]
    cate2th = {}
    for c in range(num_cate):
        y_true = y_true_all[:,c]
        y_pred = y_pred_all[:,c]

        best_th = 0
        best_f1 = -1

        for th in np.linspace(0,1,num_split,endpoint=False):
            f1 = f1_score(y_true,(y_pred > th).astype(int))
            if best_f1 <= f1:
                best_f1 = f1
                best_th = th

        cate2th[c] = best_th

    return cate2th


def get_oof_data(cfg):
    dataset = pd.read_csv(data_dir + 'train.csv')
    dataset['Id'] = data_dir + 'img_data/' + dataset['Id']
    dataset['suffix'] = '.png'
    if cfg['use_external_data']:
        ex_data = pd.read_csv(data_dir + 'HPAv18RBGY_wodpl.csv')
        ex_data['Id'] = data_dir + 'HPAv18/' + ex_data['Id']
        ex_data['suffix'] = '.jpg'
        dataset = dataset.append(ex_data).reset_index(drop=True)

    print(cfg)



    model,opt = cfg['model'](cfg)
    model.cuda()

    target = np.zeros((len(dataset), 28))
    oof_pred = np.zeros((len(dataset), 28))
    for i, labels in enumerate(dataset['Target']):
        labels = [int(t) for t in labels.split() if t != '']
        for l in labels:
            target[i, l] = 1

    folds = MultilabelStratifiedKFold(cfg['nfold'], shuffle=True, random_state=66666)

    for n_fold, (tr_idx, val_idx) in enumerate(folds.split(dataset['Id'], target)):
        print(val_idx)
        if n_fold not in cfg['fold']:
            continue

        print(n_fold, '-----------------')

        val_data = dataset.iloc[val_idx]
        val_y = target[val_idx]
        model.load_state_dict(torch.load(f"../weights/{cfg['name']}_fold{n_fold}.pkl"))
        gen1 = HumanDataset(val_data, 'test', cfg, None, None)
        gen2 = HumanDataset(val_data, 'test', cfg, None, get_augumentor('TTA1'))
        gen3 = HumanDataset(val_data, 'test', cfg, None, get_augumentor('TTA2'))
        gen4 = HumanDataset(val_data, 'test', cfg, None, get_augumentor('TTA3'))
        gen5 = HumanDataset(val_data, 'test', cfg, None, get_augumentor('TTA4'))
        gen6 = HumanDataset(val_data, 'test', cfg, None, get_augumentor('TTA5'))

        dataloader1 = DataLoader(gen1, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
        dataloader2 = DataLoader(gen2, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
        dataloader3 = DataLoader(gen3, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
        dataloader4 = DataLoader(gen4, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
        dataloader5 = DataLoader(gen5, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
        dataloader6 = DataLoader(gen6, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)

        _, pred1 = predict(model, dataloader1)
        _, pred2 = predict(model, dataloader2)
        _, pred3 = predict(model, dataloader3)
        _, pred4 = predict(model, dataloader4)
        _, pred5 = predict(model, dataloader5)
        _, pred6 = predict(model, dataloader6)
        pred = np.average([pred1, pred2, pred3, pred4, pred5, pred6], axis=0)

        print(get_best_thres(val_y,pred))

        oof_pred[val_idx] = pred

    print(get_best_thres(target,oof_pred))
    if cfg['save_oof']:
        np.save('../data/oof_pred',oof_pred)


def get_class_weight(mode,mu=0.5):
    import math
    labels_dict = [
        40958.0,
        3072.0,
        10871.0,
        3329.0,
        5130.0,
        5938.0,
        3725.0,
        9405.0,
        217.0,
        197.0,
        182.0,
        2194.0,
        2233.0,
        1458.0,
        2692.0,
        63.0,
        1290.0,
        446.0,
        1893.0,
        3672.0,
        438.0,
        13809.0,
        2729.0,
        10345.0,
        428.0,
        37366.0,
        706.0,
        127.0
    ]

    total = np.sum(labels_dict)

    class_weight_log = np.zeros(28)

    for key in range(28):
        score_log = math.log(mu * total / float(labels_dict[key]))
        class_weight_log[key] = round(score_log, 2) if score_log > 1.0 else 1.0
    class_weight_log = 28 * class_weight_log / np.sum(class_weight_log)
    return np.expand_dims(class_weight_log,0)

if __name__ == '__main__':
    # print(get_class_weight('log'))
    # print(get_class_weight('linear'))
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    #
    from config import *

    args = parser.parse_args()
    cfg = {}
    cfg['fold'] = [int(fold) for fold in args.fold]
    cfg['nfold'] = 5
    cfg['model'] = se_resnext
    cfg['lr'] = 0.0001
    cfg['backbone_lr'] = cfg['lr']
    cfg['bs'] = 64
    cfg['channel'] = 4
    cfg['name'] = args.name
    cfg['aug'] = True
    cfg['save_oof'] = False
    cfg['use_external_data'] = True
    #
    get_oof_data(cfg)





























