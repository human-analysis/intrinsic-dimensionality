# *ID&DeepMDS*: On the Intrinsic Dimensionality of Image Representation

By Sixue Gong, Vishnu Naresh Boddeti, and Anil K. Jain

## Contents
0. [Introduction](#introduction)
0. [Citation](#citation)
0. [Requirements](#requirements)
0. [Usage](#usage)

### Introduction

This code archive includes the Python implementation of intrinsic dimensionality estimation for image representation, and the proposed dimensionality reduction method -- DeepMDS. Our work, *ID&DeepMDS*, addressed two basic but fundamental questions in representation learning, i.e., its intrinsic dimensionality and if we can find a mapping between the ambient and intrinsic space while maintaining the discriminative capability of the representation. The proposed *ID&DeepMDS* is able to estimate intrinsic dimensionality for a given image representation, and then transform the ambient space to the intrinsic space based on an unsupervised DNN dimensionality reduction method under the framework of multidimensional scaling.

### Citation

If you think **ID&DeepMDS** is useful to your research, please cite:

    @inproceedings{gong2019intrinsic,
      title={On the Intrinsic Dimensionality of Image Representations},
      author={Gong, Sixue and Boddeti, Vishnu Naresh and Jain, Anil K},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      pages={3987--3996},
      year={2019}
    }
    
**Link** to the paper: https://arxiv.org/abs/1803.09672

### Requirements

1. Require `Python3`
2. Require `PyTorch1.0`
3. Require `Visdom0.1.7`
4. Check `Requirements.txt` for detailed dependencies.

### Usage

#### Part A: Estimating Intrinsic Dimensionality (ID)
**Note:** All scripts for ID estimation is in the `ID_estimate` folder. We provide implementations of three ID estimation methods, one graph-based estimator that we used in our main paper; two other baselines, including the k-nearest neighbor based estimator and the one called "Intrinsic Dimensionality Estimation Algorithm" (IDEA). All parameters for ID estimation are automatically loaded from `./ID_estimate/args.py`. Here, we show an example on how to use it to estimate ID for your own data.

1. Define how to build the estimator model:

        1) the number of points that are considered to be neighbors -- "n_neighbors"
        2) the metric for measuring distance between data points -- "dist_type"
        3) if you want to normalize your feature vectors or not -- "if_norm"

    Example in `args.py`:
    ```
    n_neighbors = 4
    # dist_type = 'Arclength'
    # dist_type = 'Euclidean'
    dist_type = 'Cosine'
    if_norm = True
    ```
2. Set the path to your input data and the path where you want to save results
**Note**: We support three formats of the input data, i.e., `.npy`, `.npz`, and `.mat` with the keyword `feat` for representation vectors.

        1) the folder to save all the results, including intermediate ones -- "resfolder"
        2) the path to your input data -- "data_filename"
        3) the path to save the sorted distance matrix of your input data -- "dist_table_filename"

    Example in `args.py`:
    ```
    resfolder = '/research/prip-gongsixu/results/idest/sphereface/lfw/cosine/k{:d}'.format(n_neighbors)

    data_filename = '/research/prip-gongsixu/results/feats/evaluation/feat_lfwblufr_sphere.mat'
    dist_table_filename = '/research/prip-gongsixu/results/idest/sphereface/lfw/cosine/dist_table.npy'
    ```

3. Set steps you want to skip
We save intermediate results for multiple computations. Some intermediate results can share across different model settings and different estimators. For example, the distance matrix can be used for different `n_neighbors` and different ID methods. Once you determine the distance metric, you only need to compute distance matrix one time.

    Example in `args.py`:
    ```
    if_dist_table = True
    if_knn_matrix = True
    if_shortest_path = True
    if_histogram = True
    ```
    which means all steps need to be computed.

4. Run the scripts
Once you set all the necessary parameters, you can start running the scripts for each method.
        
    1) Graph based estimator: `python ID_graph_largedata.py`
    2) KNN based estimator: `python ID_knn.py`
    3) IDEA estimator: `python ID_idea.py`

#### Part B: Reducing Dimensionality (DeepMDS)
**Note:** All scripts for dimensionality reduction is in the `deepmds` folder. Similar to ID estimation, all parameters are automatically loaded from `./deepmds/args.py`. Here, we show an example on how to use it to reduce dimensionality of your own data in multiple stages. You leave the default settings for the parameters that are not mentioned in the following example.

1. Visualization Settings
The parameters for visdom to plot training and testing curves.

        1) the port number for visdom -- "port"
        2) the name for current environment -- "env"
        3) if you want to create a new environment every time you run the program or not -- "same_env".  If you do, set it "False"; otherwise, it's "True".

    Example in `args.py`:
    ```
    port = 8095
    env = 'main'
    same_env = True
    ```

2. Main Settings

        1) if you want to save the results or not -- "save_results". If you do, set it "True". It saves all the trained models, the loss tracking log, the test accuracy log, and the copy of the argument file ("args.py")
        2) if you just need to extract features or not -- "extract_feat". If you do, set it "True".
        3) if you just need to test the model or not -- "just_test". If you do, set it "True".
        4) the path to saving extracted feature vectors -- "feat_savepath".
        5) the path to a half-trained model or a well-trained model -- "resume". If you want to resume a previous training, or if you want to test a well trained model, or if you want to extract features by some model, you need to put the model path to it. Otherwise, set it "None".
        6) if you want to finetune the entire deepmds model throughout all stages or not -- "finetune". If you do, set it "True".

    Example in `args.py`:
    ```
    log_type = 'traditional'
    save_results = False
    result_path = '/research/prip-gongsixu/results/models/projection/lfw/facenet128/cos_norm2/lfw_64'
    extract_feat = False
    just_test = False
    feat_savepath = '/research/prip-gongsixu/results/feats/projection/lfw/facenet128/cos_norm/feat_64.mat'
    resume = None
    # resume = '/research/prip-gongsixu/results/models/projection/lfw/sphereface/cos_norm/lfw_64/Save/0.99/model_99_64D_0.99.pth'
    finetune = False
    ```

3. Data Settings
        
        1) the way to load your training data -- "dataset_train". If you want to load feature vectors pairwise, use "Featpair"; if the dataset is big, and it's not possible to generate all possible pairs in advance, use "ClassPairDataLoader_LabelList".
        2) the path to your training data -- "input_filename_train".
        3) the path to the text file of the file list for your training data -- "label_filename_train". Each row in the text file is an address of one image, e.g., "./class_id/filename.jpg".
        4) the path to the files of image pair indices -- "pair_index_filename". It's a list of strings. The first one is filename of imposter pair, and the second one is filename of genuine pair.
        5) the total number of images in your training dataset -- "num_images". If you want to use "numpy.memmap", you need to give the correct number of images.
        6) the dimensionality of input data -- "in_dims". Again, if your data format is "numpy.memmap", you need to provide the right dimensions.
        7) if you want to normalize the input feature vectors -- "if_norm". If you do, set it "True".
        8) the path to your testing data -- "input_filename_test"
        9) the path to the text file of the file list for your testing data -- "label_filename_test". Same format as the training labels.
        10) if you want to use GPU -- "cuda". If you do, set it "True".
        11) the number of GPUs you want to use -- "ngpu".
        12) the number of threads to load your data -- "nthreads"

    Example in `args.py`:
    ```
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
    ```

4. Network Model Settings
        
        the stages where you want to reduce the dimensionality -- "model_options". In the training for each stage, set the input dimension ("in_dims") and output dimension ("out_dims"). If you finish training all stages separately, you may also want to finetune all stages. Then, you need to set dimensions for all stages. For example, uncomment several lines in the example "args.py".

    Example in `args.py`:
    ```
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
    ```

5. Training Settings
        
        1) the total number of epochs for training -- "nepochs". If you use "ClassPairDataLoader_LabelList" as the training dataloader, you may need to train more epochs than the pairwise loader.
        2) the initial index of epoch -- "nepochs".
        3) your batch size -- "batch_size". If you use "Featpair", "batch_size" means the number of image pairs; if you use "ClassPairDataLoader_LabelList", "batch_size" means the number of classes to be loaded in one mini-batch. You may need to set a large batch size for "Featpair" considering the large number of image pairs.
        4) the number of images loaded as imposter in one batch -- "batch_imposter_image". It's used by "ClassPairDataLoader_LabelList", for generating imposter pairs.
        5) the number of images loaded as genuine in one batch -- "batch_genuine_image". Let x denote "batch_imposter_image", y denote "batch_genuine_image", and z denote "batch_size". Then, for "ClassPairDataLoader_LabelList", the total number of imposter pairs == $z \times combination(x,2)$; the total number of genuine pairs == $combination(y,2)$.

    Example in `args.py`:
    ```
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
    ```

6. Evaluation Settings
**Note:** The *DeepMDS* code only provides face verification as the evaluation metric, including protocol of LFW, IJB-A, IJB-B, and IJB-C datasets. Here, we only give an example of face verification on LFW dataset.

        the path to the pair file of LFW provided by the dataset protocol -- "pairs_filename"

    Example in `args.py`:
    ```
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
    ```

7. Run *DeepMDS*
Once you set the arguments as mentioned above, you can start running *DeepMDS* to reduce the dimensionality stagewisely. You can either train a model, or test a certain model, or extract features from the pre-trained model. 
**Note:** The model training mode also includes validation step.
Start running the command below in a terminal:
    
        bash run.sh
