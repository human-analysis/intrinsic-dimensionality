# featarray.py

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
import scipy.io
import pickle
from sklearn.utils import shuffle

def get_labels_from_txt(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        class_sorted = [x.split('/')[-2] for x in lines]
        classname = []
        classname[:] = class_sorted[:]
        class_sorted.sort()
    labels = [int(class_sorted.index(x)) for x in classname]
    return labels

class Featarray(data.Dataset):
    def __init__(self, feat_filename, label_filename, if_norm, nimgs, ndim):
        self.nimgs = nimgs
        self.ndim = ndim
        self.if_norm = if_norm

        if feat_filename.endswith('npm'):
            self.feats = np.memmap(feat_filename, dtype='float32', mode='r', shape=(self.nimgs,ndim))
        elif feat_filename.endswith('npy'):
            self.feats = np.load(feat_filename)
        elif feat_filename.endswith('mat'):
            self.feats = scipy.io.loadmat(feat_filename)
            self.feats = self.feats['feat']
        elif feat_filename.endswith('npz'):
            self.feats = np.load(feat_filename)
            self.feats = self.feats['feat']
        elif feat_filename.endswith('pth'):
            self.feats = torch.load(feat_filename)
        else:
            raise(RuntimeError('Format does not support!'))

        if label_filename.endswith('npy'):
            self.labels = np.load(label_filename)
        elif label_filename.endswith('txt'):
            self.labels = get_labels_from_txt(label_filename)
        elif label_filename.endswith('pkl'):
            with open(label_filename, 'rb') as f:
                self.labels = pickle.load(f)
                self.labels = self.labels['labels']
        elif label_filename.endswith('npz'):
            self.labels = np.load(label_filename)
            self.labels = self.labels['feat']
        else:
            raise(RuntimeError('Format does not support!'))

    def get_feat(self, f):
        f = torch.Tensor(f)
        return f

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature = self.get_feat(self.feats[index,:])
        # normalization
        if self.if_norm:
            norm = float(np.linalg.norm(feature))
            feature = feature/norm

        label = self.labels[index]

        return feature, label