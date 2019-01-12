from lightai.core import *


class Dataset:
    def __init__(self, df, fold, train, tsfm):
        if fold is not None:
            if train:
                self.df = df[df['fold'] != fold]
            else:
                self.df = df[df['fold'] == fold]
        else:
            # test
            self.df = df
        self.tsfm = tsfm

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = self.tsfm(row)
        return sample

    def __len__(self):
        return len(self.df)
