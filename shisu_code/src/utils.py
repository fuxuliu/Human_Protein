from lightai.core import *
from .metric import F1


def get_mean(dl):
    means = []
    for img, target in dl:
        img = img.cuda()
        img = img.float()
        img = img.permute(0, 3, 1, 2)
        img = img.view(img.shape[0], img.shape[1], -1)
        means.append(img.mean(dim=-1))
    mean = torch.cat(means).mean(dim=0)
    return mean


def get_std(mean, dl):
    items = []
    for img, target in dl:
        img = img.cuda()
        img = img.float()
        img = img.permute(0, 3, 1, 2)
        img = img.view(img.shape[0], img.shape[1], -1)
        mean = mean.view(-1, 1)
        item = ((img-mean)**2).mean(dim=-1)
        items.append(item)
    std = torch.cat(items).mean(dim=0)
    return std**0.5


def get_idx_from_target(df, target):
    res = []
    for idx, targets in zip(df.index, df['Target']):
        targets = targets.split()
        for each in targets:
            if int(each) == target:
                res.append(idx)
                break
    return res


def get_cls_weight(df):
    cls_sz = []
    for i in range(28):
        sz = len(get_idx_from_target(df, i))
        cls_sz.append(sz)
    cls_sz = np.array(cls_sz)
    weight = np.log(cls_sz)/cls_sz
    weight = weight/weight.max()
    return weight


def assign_weight(df, weights=None):
    df['weight'] = 0.0
    if weights is None:
        weights = get_cls_weight(df)
    for idx, row in df.iterrows():
        targets = row['Target'].split()
        weight = 0
        for t in targets:
            weight += weights[int(t)]
        # weight = max([weights[int(t)] for t in targets])
        df.loc[idx, 'weight'] = weight
    df.weight = df.weight / df.weight.max()


def create_k_fold(k, df):
    df['fold'] = 0.0
    df = df.iloc[np.random.permutation(len(df))]
    df['fold'] = (list(range(k))*(len(df)//k+1))[:len(df)]
    return df


def make_rgb(img_id, img_fold):
    fold_path = Path(img_fold)
    colors = ['red', 'green', 'blue']
    channels = []
    for color in colors:
        channel = cv2.imread(str(fold_path/f'{img_id}_{color}.png'), -1)
        channels.append(channel)
    img = np.stack(channels, axis=-1)
    return img


def score_wrt_threshold_per_cls(logits, targets):
    scores = []
    thresholds = np.linspace(0, 1, num=100, endpoint=False)
    for threshold in thresholds:
        predict = (logits.sigmoid() > threshold).float()
        tp = (predict*targets).sum(dim=0)  # shape (28,)
        precision = tp/(predict.sum(dim=0) + 1e-8)
        recall = tp/(targets.sum(dim=0) + 1e-8)
        f1 = 2*(precision*recall/(precision+recall+1e-8))
        scores.append(f1)
    scores = torch.stack(scores).permute(1, 0).numpy()
    return scores


def score_wrt_threshold(logits, targets):
    metrics = [F1(t) for t in np.linspace(0, 1, num=100, endpoint=False)]
    for metric in metrics:
        metric(logits, targets)
    return np.array([metric.res() for metric in metrics])


def resize(sz, src, dst):
    """
    src, dst: fold path
    """
    src = Path(src)
    dst = Path(dst)

    def _resize(inp_img_path):
        img = cv2.imread(str(inp_img_path), 0)
        img = cv2.resize(img, (sz, sz))
        cv2.imwrite(str(dst/inp_img_path.parts[-1].replace('jpg', 'png')), img)
    with ProcessPoolExecutor(6) as e:
        e.map(_resize, src.iterdir())


def p_tp_vs_tn(logits, targets):
    p = logits.sigmoid()
    p_for_tp = p.masked_select(targets == 1).numpy()
    p_for_tn = p.masked_select(targets == 0).numpy()
    return p_for_tp, p_for_tn


def p_wrt_test(model, test_dl):
    ps = []
    with torch.no_grad():
        model.eval()
        for img in test_dl:
            img = img.cuda()
            p = model(img).sigmoid().view(-1).cpu().float()
            ps.append(p)
    return torch.cat(ps).numpy()


def val_vs_test(model, val_dl, test_dl):
    val_p_tp, val_p_tn = p_tp_vs_tn(model, val_dl)
    val_p = np.concatenate((val_p_tp, val_p_tn))
    test_p = p_wrt_test(model, test_dl)
    plt.figure(figsize=(9, 9))
    val_p_num = plt.hist(val_p, log=True, bins=30, alpha=0.5, weights=np.ones_like(val_p) /
                         len(val_p), label='val')[0]
    test_p_num = plt.hist(test_p, log=True, bins=30, alpha=0.5, weights=np.ones_like(test_p) /
                          len(test_p), label='test')
    plt.legend()


def tp_vs_tn(logits, targets):
    p_tp, p_tn = p_tp_vs_tn(logits, targets)
    plt.figure(figsize=(9, 9))
    tn_num = plt.hist(p_tn, log=True, bins=30, alpha=0.5)[0]
    tp_num = plt.hist(p_tp, log=True, bins=30, alpha=0.5)[0]
    return tp_num, tn_num


def c_p_tp_vs_tn(logits, targets):
    tp_cls = []
    tn_cls = []
    for c in range(28):
        tp = logits[:, c][targets[:, c] == 1]
        tn = logits[:, c][targets[:, c] == 0]
        tp_cls.append(tp.numpy())
        tn_cls.append(tn.numpy())
    return tp_cls, tn_cls


def c_tp_vs_tn(logits, targets):
    tp_cls, tn_cls = c_p_tp_vs_tn(logits, targets)
    _, axes = plt.subplots(28, 1, figsize=(6, 6*28))
    for c, (ax, tp, tn) in enumerate(zip(axes, tp_cls, tn_cls)):
        tptn = np.concatenate([tp, tn])
        bins = np.linspace(tptn.min(), tptn.max(), 50)
        tn_num = ax.hist(tn, bins, log=True, label='tn', alpha=0.5)[0]
        tp_num = ax.hist(tp, bins, log=True, label='tp', alpha=0.5)[0]
        ax.legend()
        ax.set_title(c)


def tsfm_contrast(ds, aug):
    row = 2
    column = 2
    img_sz = 8
    _, axes = plt.subplots(row, column, figsize=(img_sz*column, img_sz*row))
    for row in axes:
        i = np.random.randint(0, len(ds))
        img = ds[i][0]
        row[0].imshow(img[:, :, :3])
        auged_img = aug(image=img)['image']
        row[1].imshow(auged_img[:, :, :3])


def mis_classify(logits, targets):
    logits = logits.sigmoid()
    fn = logits * targets
    fp = logits * (1 - targets)
    return fn, fp


def get_logits(model, val_dl):
    logits = []
    targets = []
    with torch.no_grad():
        model.eval()
        for img, target in val_dl:
            img = img.cuda()
            logit = model(img)
            logits.append(logit)
            targets.append(target)
    logits = torch.cat(logits).cpu().float()
    targets = torch.cat(targets)
    return logits, targets


def most_wrong(logits, targets, val_df):
    p = logits.sigmoid()
    wrong = (1-p) * targets / targets.sum(dim=1).view(-1, 1)
    wrong = wrong.sum(dim=1)
    wrong_sorted, perm = torch.sort(wrong, descending=True)
    wrong_sorted_df = val_df.iloc[perm.numpy()]
    wrong_sorted_df['wrong'] = wrong_sorted.numpy()
    return wrong_sorted_df
