import argparse
import numpy as np
from sklearn.cross_validation import StratifiedKFold,KFold
from sklearn.metrics import f1_score
from glob import glob
from models import *
from utils import *
import pandas as pd
from torch.utils.data import DataLoader
import argparse


def get_predict_npy(cfg):
    dataset = pd.read_csv(data_dir+'sample_submission.csv',dtype={'Predicted':str})

    print(cfg)

    test = dataset.copy()
    test.columns = ['Id','Target']
    test['Id'] = data_dir + 'img_data/' + test['Id']
    test['suffix'] = '.png'
    gen1 = HumanDataset(test, 'test', cfg, None, None)
    gen2 = HumanDataset(test, 'test', cfg, None, get_augumentor('TTA1'))
    gen3 = HumanDataset(test, 'test', cfg, None, get_augumentor('TTA2'))
    gen4 = HumanDataset(test, 'test', cfg, None, get_augumentor('TTA3'))
    gen5 = HumanDataset(test, 'test', cfg, None, get_augumentor('TTA4'))
    gen6 = HumanDataset(test, 'test', cfg, None, get_augumentor('TTA5'))

    dataloader1 = DataLoader(gen1, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
    dataloader2 = DataLoader(gen2, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
    dataloader3 = DataLoader(gen3, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
    dataloader4 = DataLoader(gen4, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
    dataloader5 = DataLoader(gen5, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)
    dataloader6 = DataLoader(gen6, batch_size=cfg['bs'], shuffle=False, pin_memory=True, num_workers=5)

    model,_ = cfg['model'](cfg)
    model.cuda()

    result = []
    result2 = []
    for fold in cfg['fold']:
        model.load_state_dict(torch.load(f"../weights/{cfg['name']}_fold{fold}.pkl"))
        _, pred1 = predict(model, dataloader1)
        _, pred2 = predict(model, dataloader2)
        _, pred3 = predict(model, dataloader3)
        _, pred4 = predict(model, dataloader4)
        _, pred5 = predict(model, dataloader5)
        _, pred6 = predict(model, dataloader6)
        result.append(np.average([pred1,pred2,pred3,pred4,pred5,pred6],axis=0))
        result2.append(np.average([pred1,pred2,pred3,pred4],axis=0))
    result = np.average(result,axis=0)
    result2 = np.average(result2,axis=0)
    np.save(f"{cfg['name']}submit",result)
    np.save(f"{cfg['name']}submit2", result2)

def main(cfg):
    dataset = pd.read_csv(data_dir + 'sample_submission.csv', dtype={'Predicted': str})
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
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    from config import *

    args = parser.parse_args()
    cfg = {}
    cfg['fold'] = [int(fold) for fold in args.fold]
    cfg['device'] = [i for i in range(len(args.device))]
    cfg['model'] = resnet18_model
    cfg['lr'] = 0.0001
    cfg['backbone_lr'] = cfg['lr']
    cfg['bs'] = 256
    cfg['channel'] = 4
    cfg['name'] = args.name
    cfg['aug'] = True
    cfg['thres'] = [0.3]*28

    get_predict_npy(cfg)
    main(cfg)

    # cfg['thres'] = [0.5]*28
    # main(cfg)



