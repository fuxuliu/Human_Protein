from lightai.core import *
from lightai.train import *
import cv2
from torch.utils.data import DataLoader
from src.dataset import *
from src.file_op import *
from src.metric import *
from src.tsfm import *
from src.model import *
from src.loss import *
from src.utils import *
from albumentations import *
import shutil
import torch.multiprocessing as mp
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from sync_models import sysc_model
torch.backends.cudnn.benchmark = True



def main(cfg):
    print(cfg)
    sz = 512


    df = pd.read_csv('../data/full.csv')
    for fold in cfg['folds']:
        wd = 4e-4
        sgd = partial(optim.SGD, lr=0, momentum=0.9, weight_decay=wd)

        fair_img_tsfm = Compose([
            Flip(p=0.75),
            Transpose(),
            RandomBrightnessContrast(brightness_limit=(-0.25, 0.1), contrast_limit=(-0, 0)),
        ])
        weighted_img_tsfm = Compose([
            ShiftScaleRotate(rotate_limit=45, shift_limit=0.1, scale_limit=0.1, p=1,
                             border_mode=cv2.BORDER_CONSTANT)
        ])
        trn_tsfm = Tsfm(sz, fair_img_tsfm, weighted_img_tsfm)
        val_tsfm = Tsfm(sz)

        trn_ds = Dataset(df, fold=fold, train=True, tsfm=trn_tsfm)
        trn_sampler = BatchSampler(WeightedRandomSampler(trn_ds.df['weight'].values, len(trn_ds)),
                                   batch_size=cfg['bs'], drop_last=True)
        trn_dl = DataLoader(trn_ds, batch_sampler=trn_sampler, num_workers=num_workers, pin_memory=True)
        val_ds = Dataset(df, fold=fold, train=False, tsfm=val_tsfm)
        val_sampler = BatchSampler(SequentialSampler(val_ds), batch_size=cfg['bs'], drop_last=False)
        val_dl = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=num_workers, pin_memory=True)

        name = f'{cfg["model_name"]}_fold{fold}'
        writer = SummaryWriter(f'./log/{name}')
        model = Model(base=torchvision.models.resnet34).cuda()
        metric = F1(threshold=0.5)
        learner = Learner(model=model, trn_dl=trn_dl, val_dl=val_dl, optim_fn=sgd,
                          metrics=[metric], loss_fn=f1_loss,
                          callbacks=[], writer=writer)
        to_fp16(learner, 512)
        learner.callbacks.append(SaveBestModel(learner, small_better=False, name=name+'.pkl',
                                               model_dir=f'../weights/'))

        epoches = 10
        warmup_batches = 2 * len(trn_dl)
        lr1 = np.linspace( cfg['base_lr'] / 25,  cfg['base_lr'], num=warmup_batches, endpoint=False)
        lr2 = np.linspace( cfg['base_lr'],  cfg['base_lr'] / 25, num=epoches * len(trn_dl) - warmup_batches)
        lrs = np.concatenate((lr1, lr2))

        # epoches = 10
        # max_lr = 5e-2
        # warmup_batches = 2 * len(trn_dl)
        # lr1 = np.linspace(max_lr / 25, max_lr, num=warmup_batches, endpoint=False)
        # lr2 = np.linspace(max_lr, max_lr / cfg['rate'], num=epoches * len(trn_dl) - warmup_batches)
        # lrs = np.concatenate((lr1, lr2))
        lr_sched = LrScheduler(learner.optimizer, lrs)
        learner.fit(epoches, lr_sched)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)

    args = parser.parse_args()
    torch.cuda.set_device(int(args.device))

    cfg = {}
    cfg['folds'] = [int(fold) for fold in args.fold]
    cfg['bs'] = 96
    cfg['model_name'] = 'res18_shisu'
    cfg['base_lr'] = 5e-2
    num_workers = 6
    main(cfg)














































































