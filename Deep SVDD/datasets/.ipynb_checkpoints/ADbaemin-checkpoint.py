# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(1, '/workspace/Deep SVDD/')
sys.path.insert(1, '/workspace/')

from base.torch_dataset import TorchDataset
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class ADbaemin_Dataset(TorchDataset):
    def __init__(self, root: str, normal_class=0): #normal_class는 어차피 main 돌릴 때 지정해버림.
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)

        #우리는 모든 data가 normal class 이므로 그냥 넘기면 된다.
        self.train_set = MyADBaemin(root_dir=self.root, train=True)
        self.test_set = MyADBaemin(root_dir=self.root, train=False)

class MyADBaemin(Dataset):
    def __init__(self, root_dir, train, transform = None):
        train_dir = os.path.join(root_dir, 'train_data.csv')
        test_dir = os.path.join(root_dir, 'test_data.csv')
        
        temp = pd.read_csv(test_dir)
        temp = temp.iloc[:, 2:]
        temp = temp.fillna(0)
        normal_temp = self.normalize(temp)
        self.train_data = normal_temp.drop(['abuse_yn'], axis=1)
        self.train_labels = pd.DataFrame(np.zeros(self.train_data.shape))
        
        temp = pd.read_csv(test_dir)
        temp = temp.iloc[:, 2:]
        temp = temp.fillna(0)
        normal_temp = self.normalize(temp)
        self.test_data = normal_temp.drop(['abuse_yn'], axis=1)
        self.test_labels = normal_temp['abuse_yn']
        
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        
    def __len__(self):
        if self.train:
            length = len(self.train_data)
        else:
            length = len(self.test_data)
        
        return length

    def normalize(self, df):
        result = df.copy()
        for feature_name in df.columns:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            if max_value == min_value:
                print("error")
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
        return result

    def __getitem__(self, idx):
        if self.train:
            vec, target = np.array(self.train_data.iloc[idx]), np.array(self.train_labels.iloc[idx])
        else:
            vec, target = np.array(self.test_data.iloc[idx]), np.array(self.test_labels.iloc[idx])

        if self.transform is not None:
            vec = self.transform(vec)

        return vec, target, idx
