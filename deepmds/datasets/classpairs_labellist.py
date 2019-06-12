# 1. load a certain number of images from each class.
# 2. The genuine and imposter pairs are generated from those images
# 3. The way to decide which images go to genuine or imposter group is by setting two
# independent iterators.
# 4. The class label information is from the input label file.

import os
import torch
import torch.utils.data as data
import datasets.loaders as loaders
import numpy as np
import scipy.io
import pickle

IMG_EXTENTIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENTIONS)

def get_labels_from_txt(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        class_sorted = [x.split('/')[-2] for x in lines]
        classname = []
        classname[:] = class_sorted[:]
        class_sorted.sort()
    labels = [int(class_sorted.index(x)) for x in classname]
    return labels

def make_dataset(label_filename):
    datadict = {}
    if label_filename.endswith('pkl'):
        with open(label_filename, 'rb') as fp:
            labels = pickle.load(fp)
            labels = np.array(labels['labels'])
    elif label_filename.endswith('txt'):
        labels = get_labels_from_txt(label_filename)
        labels = np.array(labels)
    else:
        raise(RuntimeError('Input data file does not support!'))

    classes = list(set(labels))
    classes.sort()
    for i in classes:
        datadict[i] = list(np.where(labels==i)[0])
        if len(datadict[i]) == 0:
            print('length is 0!')
    return datadict, classes

class Iterator(object):
    def __init__(self, imagelist):
        self.length = len(imagelist)
        self.temp = torch.randperm(self.length)
        self.current = 0

    def __iter__(self):
        return self

    def next(self):
        value = self.temp[self.current]
        self.current += 1
        if self.current == self.length:
            self.current = 0
            self.temp = torch.randperm(self.length)
        return value

class ClassPairDataLoader_LabelList(data.Dataset):
    def __init__(self, feat_filename, label_filename, if_norm, 
        batch_imposter_image, batch_geunine_image):
        self.if_norm = if_norm
        self.batch_imposter_image = batch_imposter_image
        self.batch_genuine_image = batch_geunine_image
        if feat_filename.endswith('pth'):
            self.feats = torch.load(feat_filename)
        elif feat_filename.endswith('npy'):
            self.feats = np.load(feat_filename)
        elif feat_filename.endswith('npz'):
            self.feats = np.load(feat_filename)
            self.feats = self.feats['feat']
        elif feat_filename.endswith('mat'):
            self.feats = scipy.io.loadmat(feat_filename)
            self.feats = self.feats['feat']
        else:
            raise(RuntimeError('Feature file format does not support!'))

        datadict, classes = make_dataset(label_filename)
        if len(datadict) == 0:
            raise(RuntimeError('No images found'))
        else:
            self.classes = classes
            self.datadict = datadict

        self.num_classes = len(self.classes)
        self.iterdict_pos = {}
        self.iterdict_neg = {}
        for i in range(self.num_classes):
            self.iterdict_pos[i] = Iterator(datadict[self.classes[i]])
            self.iterdict_neg[i] = Iterator(datadict[self.classes[i]])

    def __len__(self):
        return self.num_classes

    def __getitem__(self, index):
        images = []
        num_images = self.batch_imposter_image + self.batch_genuine_image;
        for i in range(num_images):
            if i < self.batch_imposter_image:
                ind = self.iterdict_neg[index].next()
            else:
                ind = self.iterdict_pos[index].next()
            imgidx = self.datadict[self.classes[index]][ind]
            image = self.feats[imgidx,:]
            # normalization
            if self.if_norm:
                norm = float(np.linalg.norm(image))
                image = image/norm
                
            image = torch.Tensor(image)
            images.append(image)
        return images