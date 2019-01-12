from lightai.core import *
import os

def get_img(row, sz, train):
    channels = []
    for color in ['red', 'green', 'blue', 'yellow']:
        name = f"{row['Id']}_{color}"
        img_path = f'../data/img_data/{name}'
        if os.path.exists(img_path + '.png'):
            img_path += '.png'
        else:
            img_path = f'../data/HPAv18/{name}.jpg'
        channel = cv2.imread(img_path, -1)
        channels.append(channel)
    img = np.stack(channels, axis=-1)
    return img


def get_target(row):
    targets = row['Target'].split()
    targets = [int(t) for t in targets]
    res = np.zeros(28, dtype=np.float32)
    res[targets] = 1
    return res


class Tsfm:
    def __init__(self, sz, fair_img_tsfm=None, weighted_img_tsfm=None):
        self.sz = sz
        self.fair_img_tsfm = fair_img_tsfm
        self.weighted_img_tsfm = weighted_img_tsfm

    def __call__(self, row):
        img = get_img(row, self.sz, True)
        target = get_target(row)
        if self.fair_img_tsfm:
            img = self.fair_img_tsfm(image=img)['image']
        if self.weighted_img_tsfm:
            weight = row['weight']
            p = weight*0.25 + 0.5
            if np.random.rand() < p:
                img = self.weighted_img_tsfm(image=img)['image']
            # img = self.weighted_img_tsfm(image=img)['image']
        return img, target


class TestTsfm:
    def __init__(self, sz, tta=True):
        self.sz = sz
        self.tta = tta

    def __call__(self, row):
        img = get_img(row, self.sz, False)
        if not self.tta:
            return img
        imgs = []
        for transpose in [0, 1]:
            for h_flip in [0, 1]:
                for v_flip in [0, 1]:
                    tta = img
                    if transpose:
                        tta = np.transpose(tta, axes=(1, 0, 2))
                    if h_flip:
                        tta = tta[:, ::-1]
                    if v_flip:
                        tta = tta[::-1]
                    imgs.append(tta.copy())
        return np.array(imgs)
