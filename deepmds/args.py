import os

# ===================== Visualization Settings =============================
port = 8095
env = 'main'
same_env = True
# ===================== Visualization Settings =============================

# ======================== Main Setings ====================================
log_type = 'traditional'
save_results = False
result_path = '/research/prip-gongsixu/results/models/projection/lfw/facenet128/cos_norm2/lfw_64'
extract_feat = False
just_test = False
feat_savepath = '/research/prip-gongsixu/results/feats/projection/lfw/facenet128/cos_norm/feat_64.mat'
resume = None
# resume = '/research/prip-gongsixu/results/models/projection/lfw/sphereface/cos_norm/lfw_64/Save/0.99/model_99_64D_0.99.pth'
finetune = False
# ======================== Main Setings ====================================

# ======================= Data Setings =====================================
dataset_root_test = ''
dataset_root_train = ''
dataset_options = {}

dataset_train = 'Featpair'
# dataset_train = 'ClassPairDataLoader_LabelList'
if dataset_train == 'ClassPairDataLoader_LabelList':
    train_by_class = True
else:
    train_by_class = False
input_filename_train = '/research/prip-gongsixu/results/feats/evaluation/feat_lfwblufr_facenet.npy'
label_filename_train = '/research/prip-gongsixu/results/feats/evaluation/list_lfwblufr.txt'
pair_index_filename = ['/research/prip-gongsixu/results/evaluation/lfw/blufr_imp_pair.npy', \
    '/research/prip-gongsixu/results/evaluation/lfw/blufr_gen_pair.npy']
template_filename = None
# num_images = 131341
num_images = 13233
in_dims = 128
if_norm = True

dataset_test = 'Featarray'
input_filename_test = '/research/prip-gongsixu/results/feats/evaluation/feat_lfwblufr_facenet.npy'
label_filename_test = '/research/prip-gongsixu/results/feats/evaluation/list_lfwblufr.txt'

split = 1
loader_input = 'loader_image'
loader_label = 'loader_numpy'
test_dev_percent = None
train_dev_percent = None

save_dir = os.path.join(result_path,'Save')
logs_dir = os.path.join(result_path,'Logs')

# cpu/gpu settings
cuda = True
ngpu = 1
nthreads = 10
# ======================= Data Setings =====================================

# ======================= Network Model Setings ============================
# model_type = 'resnet18'
model_type = 'CompNet_later'
# model_type = 'incep_resnetV1'
model_options = [ \
    # {"in_dims": 512, "out_dims": 256}, \
    # {"in_dims": 256, "out_dims": 128}, \
    {"in_dims": 128, "out_dims": 64}, \
    # {"in_dims": 64, "out_dims": 32}, \
    # {"in_dims": 32, "out_dims": 16}, \
    # {"in_dims": 16, "out_dims": 10}, \
    # {"in_dims": 10, "out_dims": 3}, \
    # {"in_dims": 3, "out_dims": 2}, \
    ]
# loss_type = 'Classification'
loss_type = 'BatchHardPairL2NormLoss'
loss_options = {"dist_metric": 'Euclidean', "threshold": 0.0}

# input data size
input_high = 182
input_wide = 182
resolution_high = 160
resolution_wide = 160
# ======================= Network Model Setings ============================

# ======================= Training Settings ================================
# initialization
manual_seed = 0
nepochs = 20
epoch_number = 0

# batch
batch_size = 6000
batch_imposter_image = 5 # (5 for imposter; 10 for genuine)
batch_genuine_image = 10

# optimization
optim_method = 'Adam'
optim_options = {"betas": (0.9, 0.999)}
# optim_method = "SGD"
# optim_options = {"momentum": 0.9, "weight_decay": 0.0}

# learning rate
learning_rate = 3e-4
scheduler_method = 'CosineAnnealingLR'
scheduler_options = {"T_max": nepochs, "eta_min": 1e-6}

# weight of loss
weight_loss = [1 for x in model_options]

# preprocessing
preprocess_train = {"Resize": True, 
    "RandomRotation": 10,
    # "CenterCrop": True,
    "RandomCrop": "True",
    "RandomHorizontalFlip": True, 
    "RandomVerticalFlip": True,  
    "Normalize": ((0.5,0.5,0.5), (0.5,0.5,0.5)), 
    "ToTensor": True}

preprocess_test = {"Resize": True, 
    "CenterCrop": True, 
    # "RandomCrop": True, 
    # "RandomHorizontalFlip": True, 
    # "RandomVerticalFlip": True, 
    # "RandomRotation": 10, 
    "Normalize": ((0.5,0.5,0.5), (0.5,0.5,0.5)), 
    "ToTensor": True}
# ======================= Training Settings ================================

# ======================= Evaluation Settings ==============================
# protocol and metric
protocol = 'LFW'
metric = 'Euclidean'

# files related to protocols
# IJB
eval_dir = '/research/prip-gongsixu/results/evaluation/ijbc'
# eval_dir = '/research/prip-gongsixu/results/evaluation/ijba'
imppair_filename = os.path.join(eval_dir, 'imp_pairs.csv')
genpair_filename = os.path.join(eval_dir, 'gen_pairs.csv')
pair_dict_filename={'imposter':imppair_filename,'genuine':genpair_filename}
# pair_dict_filename = eval_dir

# LFW
pairs_filename = '/research/prip-gongsixu/results/evaluation/lfw/lfw_pairs.txt'
nfolds=10

evaluation_type = 'FaceVerification'
evaluation_options = {'label_filename': label_filename_test,\
    'protocol': protocol, 'metric': metric,\
    'nthreads': nthreads, 'multiprocess':True,\
    'pair_index_filename': pair_dict_filename,'template_filename': template_filename,\
    'pairs_filename': pairs_filename, 'nfolds': nfolds,\
    'nimgs': num_images, 'ndim': in_dims}
# evaluation_type = 'ImageRetrieval'
# evaluation_options = {"topk": 10}
# ======================= Evaluation Settings ==============================