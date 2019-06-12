import os
import argparse
import numpy as np

def main(args):
    filename = args.labelfile
    labels = get_labels_from_txt(filename)
    genid,impid = get_genpairs_imppairs(labels)
    np.save(args.filename_genpair, genid)
    np.save(args.filename_imppair, impid)

def get_labels_from_txt(filename):
    with open(filename,'r') as f:
        lines = f.readlines()
        class_sorted = [x.split('/')[-2] for x in lines]
        classname = []
        classname[:] = class_sorted[:]
        class_sorted.sort()
    labels = [int(class_sorted.index(x)) for x in classname]
    labels = np.array(labels)
    return labels

def get_genpairs_imppairs(label):
    # the returned genid and impid are tuple with lengh of 2
    # if convert to the numpy, shape = (2*N)
    # N is the number of pairs
    n = label.size
    triu_indices = np.triu_indices(n,1)
    if len(label.shape) == 1:
        label = label[:,None]

    label_mat = label==label.T
    temp = np.zeros(label_mat.shape, dtype=bool)
    temp[triu_indices] = True
    genlab = label_mat&temp
    genid = np.where(genlab==True)

    temp = np.ones(label_mat.shape, dtype=bool)
    temp[triu_indices] = False
    implab = label_mat|temp
    impid = np.where(implab==False)
    return genid,impid

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelfile', 
        type=str, 
        default='/research/prip-gongsixu/results/feats/evaluation/list_lfwblufr.txt',
        help=" The path to the text file of image list, \
        each row is an address of one image, \
        e.g., './subject_id/filename.jpg' ")
    parser.add_argument('--filename_genpair',
        type=str,
        default='./genpair.npy',
        help="The path to the numpy file of genuine pairs")
    parser.add_argument('--filename_imppair',
        type=str,
        default='./imppair.npy',
        help="The path to the numpy file of imposter pairs")
    args = parser.parse_args()

    main(args)