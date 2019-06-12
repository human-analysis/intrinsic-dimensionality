# arguments for ID
import os

n_neighbors = 4
# dist_type = 'Arclength'
# dist_type = 'Euclidean'
dist_type = 'Cosine'
if_norm = True

# resfolder = '/research/prip-gongsixu/results/idest/ResNet34/ImageNet/Euclidean/k{:d}'.format(n_neighbors)
resfolder = '/research/prip-gongsixu/results/idest/sphereface/lfw/cosine/k{:d}'.format(n_neighbors)

data_filename = '/research/prip-gongsixu/results/feats/evaluation/feat_lfwblufr_sphere.mat'
dist_table_filename = '/research/prip-gongsixu/results/idest/sphereface/lfw/cosine/dist_table.npy'
# data_filename = '/research/prip-gongsixu/results/idest/swiss_roll/data_swiss.npy'
# dist_table_filename = '/research/prip-gongsixu/results/idest/swiss_roll/arclength/dist_table.npy'

if_dist_table = True
if_knn_matrix = True
if_shortest_path = True
if_histogram = True

n_bins = 100
radius = -1
r_max = 0.0
r_min = -10

hist_filename = os.path.join(resfolder, 'histogram.txt')
param_filename = os.path.join(resfolder, 'params.txt')
radius_filename = os.path.join(resfolder, 'max_distance.txt')