from lightai.core import *

def create_k_fold(k):
    trn_df = pd.read_csv('../data/train.csv')
    trn_df = trn_df.sample(frac=1)
    trn_df['fold'] = (list(range(k)) * (len(trn_df) // k + 1))[:len(trn_df)]
    trn_df.to_csv(f'../data/{k}_fold.csv', index=False)

def create_small_imgs(img_path):
    name = img_path.parts[-1]
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (sz, sz))
    cv2.imwrite(str(new_path/name), img)

def to_gray(origin_path):
    img_name = origin_path.parts[-1]
    img = cv2.imread(str(origin_path), 0)
    cv2.imwrite(str(new_path/img_name), img)