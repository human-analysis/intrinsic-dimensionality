import numpy as np
import os
import sys
import math
import scipy.io

import knn

import config

def main():
    args = config.parse_args()

    # load data
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
    # load data

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

    # compute dimension
    # K_array = [4]
    K_array = [4, 7, 9, 15, 21, 30, 70, 90, 128]
    for K in K_array:
        d = idea(dist_table, K)
        # d = mind_kl(data, dist_table, K, dim, 'Arclength')
        print('K={}: dim = {}'.format(K,d))
        respath = os.path.join(args.resfolder, 'idea_dim.txt')
        with open(respath, 'a') as f:
            f.write('K=%d: dim=%.2f;\n' % (K, d))
    # compute dimension

def idea(dist_table, K):
    m = 0
    N = dist_table.shape[0]
    for i in range(N):
        m += (1.0/(N*K)) * (np.sum(dist_table[i,1,0:K]) / dist_table[i,1,K-1])

    print('m={}'.format(m))
    d = float(m) / (1-m)
    return d

def mind_mli(dist_table, K, dim):
    N = dist_table.shape[0]
    px = np.divide(dist_table[:,1,0], dist_table[:,1,K-1])
    px = px[np.nonzero(px)]

    d = 1
    mle = 0
    for i in range(dim):
        cur_mle = - i * np.sum(np.log(px)) - (K-1) * np.sum(np.log(1-np.power(px,i+1)))
        print(cur_mle)
        if cur_mle > mle:
            mle = cur_mle
            d = i+1
    return d

def get_pr(dist_table, K):
    N = dist_table.shape[0]
    pr = np.zeros(N)
    r = np.divide(dist_table[:,1,0], dist_table[:,1,K-1])
    r = np.sort(r)
    for i in range(N):
        if i == 0:
            pr[i] = r[i+1] - r[i]
        elif i == N-1:
            pr[i] = r[i] - r[i-1]
        else:
            pr[i] = min(r[i]-r[i-1], r[i+1]-r[i])
    return pr

def mind_kl(data, dist_table, K, dim, dist_type):
    N = dist_table.shape[0]
    
    # compute pr
    print('compute pr')
    pr = get_pr(dist_table, K)
    print('pr=%.2f' %np.min(pr))
    del dist_table
    # compute pr

    # compute pdr
    print('compute pdr')
    kld = sys.maxsize
    d = 1
    for i in range(dim):
        sample_id = np.random.choice(dim, i+1, replace=False)
        samples = data[:, sample_id]
        nb = knn.KNN(K, dist_type=dist_type, data=samples)
        dtable = nb.get_dist_multip(False)

        pdr = get_pr(dtable, K)
        cur_kld = math.log(N/(N-1)) + np.mean(np.log(np.divide(pr, pdr)))
        if cur_kld < kld:
            kld = cur_kld
            d = i+1 
        print('[%d\%d]: kld=%.2f' % (i,dim,kld))
    # compute pdr
    return d

if __name__ == '__main__':
    main()