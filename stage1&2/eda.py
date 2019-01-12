import numpy as np
import pandas as pd
from config import *
from PIL import Image
from tqdm import tqdm

def get_mean_std():
    tr = pd.read_csv(data_dir + 'train.csv')
    tr['Id'] = data_dir + 'img_data/' + tr['Id']
    tr['suffix'] = '.png'

    te = pd.read_csv(data_dir + 'sample_submission.csv', dtype={'Predicted': str})
    te = te.copy()
    te.columns = ['Id', 'Target']
    te['Id'] = data_dir + 'img_data/' + te['Id']
    te['suffix'] = '.png'

    dataset = tr[['Id','suffix']].append(te[['Id','suffix']]).reset_index(drop=True)

    # for color in ['red', 'green', 'blue', 'yellow']:
    #     mean = 0
    #     std = 0
    #     dataset['path'] = dataset['Id'] + '_' + color + dataset['suffix']
    #     for path in tqdm(dataset['path']):
    #         img = np.array(Image.open(path))
    #         mean += img.mean()
    #         std += img.std()
    #     mean /= len(dataset)
    #     std /= len(dataset)
    #     print(f'{color}  mean {mean}   std {std}')

    ex_data = pd.read_csv(data_dir + 'HPAv18RBGY_wodpl.csv')
    ex_data['Id'] = data_dir + 'HPAv18/' + ex_data['Id']
    ex_data['suffix'] = '.jpg'
    dataset = dataset[['Id','suffix']].append(ex_data[['Id','suffix']]).reset_index(drop=True)

    for color in ['red', 'green', 'blue', 'yellow']:
        mean = 0
        std = 0
        dataset['path'] = dataset['Id'] + '_' + color + dataset['suffix']
        for path in tqdm(dataset['path']):
            img = np.array(Image.open(path))
            mean += img.mean()
            std += img.std()
        mean /= len(dataset)
        std /= len(dataset)
        print(f'{color}  mean {mean}   std {std}')

get_mean_std()