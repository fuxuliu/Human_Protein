import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import json
import imagehash
from config import *
from PIL import Image
from sklearn.metrics import f1_score


def cal_phash_worker(dataset,color):

    dataset['path'] = dataset['path'] + '_' + color + dataset['suffix']
    data_hash = [' '.join(imagehash.phash(Image.open(path)).hash.astype(int).reshape(-1).astype(str)) for path in tqdm(dataset['path'])]

    return data_hash

def prepocess():

    def parallel_cal(df,color):

        num_samples = len(df)
        num_worker = mp.cpu_count()
        bs = 1 + num_samples // num_worker
        pool = mp.Pool()
        results = []

        for i in range(num_worker):
            result = pool.apply_async(cal_phash_worker, args=(df.iloc[i*bs:(i+1)*bs],color))
            results.append(result)
        pool.close()
        pool.join()

        data_hash = []
        for result in results:
            data_hash += result.get()

        df[f'hash_{color}'] = data_hash
        return df


    dataset = pd.read_csv(data_dir + 'train.csv')
    dataset['path'] = data_dir + 'img_data/' + dataset['Id']
    dataset['suffix'] = '.png'


    ex_data = pd.read_csv(data_dir + 'HPAv18RBGY_wodpl.csv')
    ex_data['path'] = data_dir + 'HPAv18/' + ex_data['Id']
    ex_data['suffix'] = '.jpg'

    dataset = dataset.append(ex_data).reset_index(drop=True)

    for color in ['red', 'green', 'blue', 'yellow']:
        dataset = parallel_cal(dataset,color)

    dataset.to_csv(data_dir+'trhash.csv',index=False)

    te = pd.read_csv(data_dir + 'sample_submission.csv', dtype={'Predicted': str})
    te.columns = ['Id', 'Target']
    te['path'] = data_dir + 'img_data/' + te['Id']
    te['suffix'] = '.png'

    for color in ['red', 'green', 'blue', 'yellow']:
        te = parallel_cal(te,color)
    te.to_csv(data_dir+'tehash.csv',index=False)



def cal_dis(data1,data2,device_id,bs=2,k=2):

    print(data1.shape, data2.shape)
    data1 = np.expand_dims(data1, axis=1)
    data2 = np.expand_dims(data2, axis=0)
    print(data1.shape)
    print(data2.shape)

    data2 = torch.from_numpy(np.array(data2)).cuda(non_blocking=True, device=device_id)

    distance = []
    indexes = []
    for i in tqdm(range(0, len(data1), bs)):
        img1 = data1[i:i + bs]
        img1 = torch.from_numpy(np.array(img1)).cuda(non_blocking=True, device=device_id)

        dis = (img1 - data2).abs()
        dis = torch.sum(dis,dim=2)
        value, index = torch.topk(dis, k=k+1, largest=False, dim=-1)

        distance += value.tolist()
        indexes += index.tolist()
    assert len(distance) == len(data1)
    return distance,indexes



def main(color='green',device_id=2,is_test=False,k=3):


    dataset2 = pd.read_csv(f'{data_dir}trhash.csv')

    if is_test:
        dataset1 = pd.read_csv(f'{data_dir}tehash.csv')
    else:
        dataset1 = dataset2.copy()

    if color == 'all':
        dataset1[f'hash_{color}'] = dataset1['hash_red'] + ' ' + dataset1['hash_green'] + ' ' + dataset1['hash_blue'] + ' ' + dataset1['hash_yellow']
        dataset2[f'hash_{color}'] = dataset2['hash_red'] + ' ' + dataset2['hash_green'] + ' ' + dataset2['hash_blue'] + ' ' + dataset2['hash_yellow']

    data1 = np.array([np.asarray(hash_v.split(),dtype=int) for hash_v in dataset1[f'hash_{color}']])
    data2 = np.array([np.asarray(hash_v.split(),dtype=int) for hash_v in dataset2[f'hash_{color}']])

    label = np.zeros((len(data2), 28))
    for i, target in enumerate(dataset2['Target']):
        target = [int(t) for t in target.split() if t != '']
        for l in target:
            label[i, l] = 1


    if True:  # 获取近邻
        dis1, idx1 = cal_dis(data1, data2, device_id, k=k)
        df_dis = np.array(dis1)
        df_idx = np.array(idx1)
        print(df_dis.shape,df_idx.shape)


    if True: # 过滤至k个
        result = {}
        result.update({f'top{i}':[] for i in range(k)})
        result.update({f'i{i}':[] for i in range(k)})
        for curr_i,(dis,idx) in enumerate(zip(df_dis,df_idx)):
            neribour = list(sorted(zip(dis,idx),key=lambda x:x[0]))
            i = 0
            for d,index in neribour:
                if is_test==False and index == curr_i:
                    continue
                result[f'top{i}'].append(d)
                result[f'i{i}'].append(index)
                i += 1
                if i == k:
                    break

    df = {}
    for i in range(k):
        df[f'top{i}'] = result[f'top{i}']
        df[f'i{i}'] = result[f'i{i}']
        df[f'score{i}'] = [label[int(j)] for j in result[f'i{i}']]

    pred_label = []
    for j in range(len(df['i0'])):
        pred = 0
        for i in range(k):
            pred += label[int(df[f'i{i}'][j])]
        pred /= k
        pred_label.append(pred.round())

    df['pred_label'] = np.array(pred_label)
    for key in df.keys():
        df[key] = np.array(df[key])

    def get_subdata(df,idx):
        return {key:v[idx] for key,v in df.items()}

    if is_test == False:
        th2score = []

        for th in range(int(df['top0'].min()),20):

            df_sub = get_subdata(df,df['top0']==th)
            label_sub = label[df['top0']==th]
            if len(df_sub['i0']) == 0:
                continue

            f1_pred = [f1_score(label_sub[:,i],df_sub['pred_label'][:,i]) for i in range(28)]

            print(f"thres:{th} num_samples:{len(df_sub['i0'])}")
            print(f'f1 {f1_pred}')

            print(f"marco f1 {f1_score(label_sub, df_sub['score0'],average='macro')}")
            print()
            th2score.append((th, f1_pred))

        open('../data/th2score.json', 'w').write(json.dumps(th2score, indent=4, separators=(',', ': ')))

    pred = pd.DataFrame()
    pred['Id'] = dataset1['Id']

    pred_target = []
    for target in df['pred_label']:
        pred_target.append(' '.join([str(i) for i,score in enumerate(target) if score > 0.5]))

    pred['Target'] = pred_target
    pred['distance'] = df['top0']

    pred.to_csv(f'leak_correct{is_test}.csv',index=False)



# def postpocess(filename,submit_file,th):
#
#     simility = pd.read_csv(filename)
#     pred = pd.read_csv(submit_file,header=None).values
#
#     assert len(pred) == len(simility)
#
#     result = np.zeros((len(pred),17))
#     num_change = 0
#     for label,(i,row) in zip(pred,simility.iterrows()):
#         if row['top0'] < th and label.argmax()!=row['pred_label']:
#             result[i,int(row['pred_label'])] = 1
#             num_change += 1
#         else:
#             result[i] = label
#     print(num_change)
#     pd.DataFrame(result).to_csv(f'post{submit_file}',index=False,header=False)

if __name__ == '__main__':


    prepocess()
    main('all', device_id=1, is_test=False, k=1)
    main('all', device_id=1, is_test=True, k=1)
    # for color in ['red', 'green', 'blue', 'yellow']:
    #     main(color,device_id=2,is_test=False,k=1)




