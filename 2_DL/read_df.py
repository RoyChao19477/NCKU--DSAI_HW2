import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd

class df(Dataset):
    def __init__(self, train=True):
        df_train = pd.read_csv("data/1_training_set.csv", index_col=0)
        df_test = pd.read_csv("data/2_testing_set.csv", index_col=0)

        if train:
            self.x = df_train.drop(columns='PRED')
            self.y = df_train['PRED']
        else:
            self.x = df_test.drop(columns='PRED')
            self.y = df_test['PRED']


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor( self.x.iloc[idx].values )
        y = torch.tensor( self.y.iloc[idx] )
        #y = torch.tensor( self.y.iloc[idx].values )

        return x, y

class df_30(Dataset):
    def __init__(self, train=True):
        df_train = pd.read_csv("data/1_training_set.csv", index_col=0)
        df_test = pd.read_csv("data/2_testing_set.csv", index_col=0)

        if train:
            self.x = df_train.drop(columns='PRED')
            self.y = df_train['PRED']
        else:
            self.x = df_test.drop(columns='PRED')
            self.y = df_test['PRED']


    def __len__(self):
        return len(self.x) - 30

    def __getitem__(self, idx):
        x = torch.tensor( self.x.iloc[idx : idx + 30].values )
        y = torch.tensor( self.y.iloc[idx + 29] )
        #y = torch.tensor( self.y.iloc[idx].values )

        return x, y