import numpy as np
import numpy.linalg as LA
from scipy.spatial.distance import cdist
import multiprocessing
import math
import pickle
import knn
import os
import scipy.io
import config

def main():
    epsilon = 0.01
    max_iter = 200
    # mu = [0, 0, 0]
    # cov = [[1, 0, 0], [0, 100, 0], [0, 0, 100]]
    # data = np.random.multivariate_normal(mu, cov, 1000)

    args = config.parse_args()

    if args.data_filename.endswith('npy'):
        data = np.load(args.data_filename)
    if args.data_filename.endswith('mat'):
        data = scipy.io.loadmat(args.data_filename)
        data = data['feat']
    if args.data_filename.endswith('npz'):
        data = np.load(args.data_filename)
        data = data['feat']
    nrof_image = data.shape[0]
    dim = data.shape[1]

    # compute and sort distance matrix
    if args.if_dist_table:
        obj = knn.KNN(128, args.dist_table_filename, args.data_filename, 
            args.dist_type, args.if_norm)
        obj.get_dist_multip()
    # compute and sort distance matrix

    # load distance matrix if you already have one (comment the lines to compute matrix)
    else:
        dist_table = np.load(args.dist_table_filename)
    # load distance matrix if you already have one
    
    # get dimension
    K_array = [4, 7, 9, 15, 21, 30, 70, 90, 128]
    for K in K_array:
        print('compute dimension')
        dim_est = Dimest(data, K, epsilon, max_iter, dist_table[:,1,0:K])
        i,d0,d2 = dim_est.get_dim()
        print('iteration {}: dim0 = {}; dim = {}'.format(i,d0,d2))
        respath = os.path.join(args.resfolder, 'knn_dim.txt')
        with open(respath, 'a') as f:
            f.write('K=%d: iteration=%d; dim0=%.2f, dim=%.2f;\n' % (K, i, d0, d2))
    # get dimension

class Dimest():
    def __init__(self, data, K, epsilon, max_iter, dist_table=None):
        self.data = data
        self.K = K
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.num_cores = multiprocessing.cpu_count()

        if dist_table is not None:
            self.dist_table = dist_table
        else:
            self.dist_table = self.get_dist_multip()
        self.dist_table = self.remove_outlier(dist_table)

    
    # *_mat: use matrix coperation (fast, but need a lot of memories); 
    # for small datasets
    def get_dist_mat(self):
        dist_mat = cdist(self.data, self.data)
        pool = multiprocessing.Pool(self.num_cores)
        # dist_table is a n * K matrix
        dist_table = pool.map(self.get_neighbors_mat, dist_mat)
        # dist_table = np.array(dist_table).T # dist_table is a K * n matrix
        pool.close()
        pool.join()

        return dist_table
    
    def get_neighbors_mat(self, dists):
        dists.sort()
        return tuple(dists[1:(self.K+1)])

    
    # *_multip: just use multiprocessing to traverse all pairs (slow, but need less memories);
    # for large datasets
    def get_dist_multip(self):
        pool = multiprocessing.Pool(self.num_cores)
        # dist_table is a n * K matrix
        dist_table = pool.map(self.get_neighbors_multip, self.data)
        # dist_table = np.array(dist_table).T # dist_table is a K * n matrix
        pool.close()
        pool.join()

        return dist_table

    def get_neighbors_multip(self, sample):
        sample = np.tile(sample, (self.data.shape[0],1))
        dists = LA.norm(sample-self.data, axis=1)
        dists.sort()
        return tuple(dists[1:(self.K+1)])

    def remove_outlier(self, dist_table):
        n_samples = self.data.shape[0]
        max_dist = dist_table[:, self.K-1]
        max_mean = np.mean(max_dist)
        max_std = np.sqrt(np.sum((max_dist - max_mean)**2)/(n_samples -1))
        thred = max_mean + max_std

        idx = [x[0] for x in enumerate(max_dist) if x[1] <= thred]
        new_dist_table = dist_table[idx, :]
        return new_dist_table

    def get_dim(self):
        log_rk = np.log(np.mean(self.dist_table, axis=0))
        log_k = np.log(np.array(range(self.K)) + 1)
        log_rkGkd = log_rk + 0 # intially, log(Gk,d) is 0
        i = 0
        d2 = (self.K * np.sum(log_k**2) - (np.sum(log_k))**2) / \
            (self.K * np.sum(log_rkGkd*log_k) - (np.sum(log_k))*(np.sum(log_rkGkd)))
        d1 = d2+1

        k_array = np.arange(self.K) + 1
        d_array = np.tile(round(d2), (self.K))

        d0 = d2
        print('First dimension: {}'.format(d0))

        while i < self.max_iter and abs(d2-d1) >= self.epsilon:
            d1 = d2
            log_Gkd = np.true_divide(d_array-1, 2*k_array*(d_array**2)) + \
                np.true_divide((d_array-1)*(d_array-2), 12*(k_array**2)*(d_array**3)) - \
                np.true_divide((d_array-1)**2, 12*(k_array**3)*(d_array**4)) - \
                np.true_divide((d_array-1)*(d_array-2)*((d_array**2)+3*d_array-3), 120*(k_array**4)*(d_array**5))

            log_rkGkd = log_rk + log_Gkd
            d2 = (self.K * np.sum(log_k**2) - (np.sum(log_k))**2) / \
                (self.K * np.sum(log_rkGkd*log_k) - (np.sum(log_k))*(np.sum(log_rkGkd)))
            d_array = np.tile(round(d2), (self.K))
            i += 1
        
        return i,d0,d2

if __name__ == "__main__":
    main()