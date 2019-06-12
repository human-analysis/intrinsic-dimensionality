import numpy as np
import numpy.linalg as LA
from scipy.spatial.distance import cdist
from sklearn.neighbors import kneighbors_graph
import multiprocessing
import math
import pickle
import scipy.io
import os
import pdb

def main():
    data_file = '/research/prip-gongsixu/results/feats/evaluation/imagenet-100-resnet34-train.npy'
    dist_table_file = '/research/prip-gongsixu/results/idest/ResNet34/ImageNet/Euclidean/dist_table.npy'
    K = 128
    dist_type = 'Euclidean'
    knn = KNN(K, dist_table_file, data_file, dist_type)
    knn.get_dist_multip()

def normalize(x, ord=None, axis=None, epsilon=10e-12):
    ''' Devide the vectors in x by their norms.'''
    if axis is None:
        axis = len(x.shape) - 1
    norm = np.linalg.norm(x, ord=None, axis=axis, keepdims=True)
    x = x / (norm + epsilon)
    return x

class KNN():
    def __init__(self, K, dist_table_file, data_file=None, dist_type='Euclidean', if_norm=True):
        if data_file.endswith('npy'):
            self.data = np.load(data_file)
        elif data_file.endswith('mat'):
            self.data = scipy.io.loadmat(data_file)
            self.data = self.data['feat']
        elif data_file.endswith('npz'):
            self.data = np.load(data_file)
            self.data = self.data['feat']
        if if_norm:
            self.data = normalize(self.data)
        self.dist_table_file = dist_table_file
        self.K = K
        self.dist_type = dist_type
        # self.num_cores = multiprocessing.cpu_count()
        self.num_cores = 4
        self.dist_table = np.zeros((self.data.shape[0], 2, K))

    # *_multip: just use multiprocessing to traverse all pairs (slow, but need less memories);
    # for large datasets
    def get_dist_multip(self, saveflag=True):
        pool = multiprocessing.Pool(self.num_cores)
        # dist_table is a n * 2 * K matrix
        # [n samples, (index, distance), K neighbors]
        dist_table = pool.map(self.get_neighbors_multip, range(self.data.shape[0]))
        pool.close()
        pool.join()
        self.dist_table[:] = np.array(dist_table)[:]
        if saveflag:
            subdir = os.path.dirname(self.dist_table_file)
            if os.path.isdir(subdir) is False:
                os.makedirs(subdir)
            np.save(self.dist_table_file, self.dist_table)
        return np.array(dist_table)

    def get_neighbors_multip(self, i):
        sample = self.data[i,:]
        if self.dist_type == 'Euclidean':
            # eclidean distance
            sample = np.tile(sample, (self.data.shape[0],1))
            dists = LA.norm(sample-self.data, axis=1)
        elif self.dist_type == 'Arclength':
            # arc length
            epsilon = 1e-6
            num = np.dot(self.data, sample)
            den = epsilon + np.linalg.norm(self.data, axis=1) * np.linalg.norm(sample)
            dists = np.arccos(num/den)
        elif self.dist_type == 'Cosine':
            # cosine distance
            epsilon = 1e-6
            num = np.dot(self.data, sample)
            den = epsilon + np.linalg.norm(self.data, axis=1) * np.linalg.norm(sample)
            dists = 1 - num/den
        else:
            raise(RuntimeError('Wrong distance metric!'))

        sorted_dists = np.sort(dists)
        sorted_idx = np.argsort(dists)

        return (sorted_idx[1:(self.K+1)], sorted_dists[1:(self.K+1)])

# if __name__ == "__main__":
#     main()