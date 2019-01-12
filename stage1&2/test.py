# import pandas as pd
# from config import *
# import numpy as np
#
# dataset = pd.read_csv(data_dir+'train.csv')
#
#
# ex_data = pd.read_csv(data_dir+'HPAv18RBGY_wodpl.csv')
#
# dataset = dataset.append(ex_data).reset_index(drop=True)
#
# target = np.zeros((len(dataset), 28))
# for i, labels in enumerate(dataset['Target']):
#     labels = [int(t) for t in labels.split() if t != '']
#     for l in labels:
#         target[i, l] = 1
#
# print(len(target),target.shape)
#
# print(target.sum(axis=0).tolist())


import numpy as np
from sklearn.svm import SVC

X = np.array([
    [3,3],
    [4,3],
    [1,1]
])
Y = [1,1,0]
test = np.array([
    [1,0],
    [4,4],
    [2,2]
])
model = SVC()
model.fit(X,Y)
print(model.predict(X))
print(model.predict(test))