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
from progressbar import *
from tqdm import tqdm
torch.backends.cudnn.benchmark = True


def get_predict_npy(cfg):
    submission = pd.read_csv('../data/sample_submission.csv')
    sz = 512
    test_ds = Dataset(submission, fold=None, train=None, tsfm=TestTsfm(sz, tta=True))
    test_sampler = BatchSampler(SequentialSampler(test_ds), batch_size=cfg['bs'], drop_last=False)
    test_dl = DataLoader(test_ds, batch_sampler=test_sampler, num_workers=6, pin_memory=True)

    model = Model(base=torchvision.models.resnet18)
    model.half()
    bn_to_float(model)


    res = []
    with torch.no_grad():
        for fold in cfg['fold']:
            model.load_state_dict(torch.load(f"../weights/{cfg['name']}_fold{fold}.pkl")['model'])
            model.cuda()
            model.eval()
            predicts = []
            for idx, imgs in tqdm(zip(test_sampler, test_dl), total=len(test_dl)):
                num_sample = imgs.size(0)
                imgs = imgs.view((-1,sz,sz,4))
                pred = model(imgs.cuda()).sigmoid().view((num_sample,-1,28)).mean(dim=1).cpu().numpy()
                predicts.append(pred)
            res.append(np.concatenate(predicts,axis=0))
    res = np.average(res,axis=0)
    np.save(f"{cfg['name']}submit",res)

def main(cfg):
    dataset = pd.read_csv('../data/sample_submission.csv', dtype={'Predicted': str})
    result = np.load(f"{cfg['name']}submit.npy")

    target = []
    for sample_pred in result:
        pred = []
        for i,score in enumerate(sample_pred):
            if score > cfg['thres'][i]:
                pred.append(str(i))
        if len(pred) == 0:
            pred.append(str(sample_pred.argmax()))
        target.append(' '.join(pred))

    dataset['Predicted'] = target
    dataset.to_csv('submit.csv',index=False)





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)


    args = parser.parse_args()
    cfg = {}
    cfg['fold'] = [int(fold) for fold in args.fold]
    cfg['bs'] = 32
    cfg['name'] = args.name
    cfg['thres'] = [0.3]*28

    get_predict_npy(cfg)
    # get_oof_data(cfg)
    main(cfg)

    # cfg['thres'] = [0.5]*28
    # main(cfg)



