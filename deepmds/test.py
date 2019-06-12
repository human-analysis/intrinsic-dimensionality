# test.py

import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import plugins
import os
import numpy as np
from scipy.spatial.distance import cdist
import scipy.io
import pdb

class Tester:
    def __init__(self, args, model, criterion, evaluation):
        self.args = args
        self.model = model
        self.criterion = criterion
        self.evaluation = evaluation
        self.save_results = args.save_results

        self.env = args.env
        self.port = args.port
        self.dir_save = args.save_dir
        self.log_type = args.log_type

        self.cuda = args.cuda
        self.nepochs = args.nepochs
        self.batch_size = args.batch_size

        self.resolution_high = args.resolution_high
        self.resolution_wide = args.resolution_wide

        # for classification
        self.labels = torch.zeros(self.batch_size).long()
        self.inputs = torch.zeros(
            self.batch_size,
            self.resolution_high,
            self.resolution_wide
        )

        if args.cuda:
            self.labels = self.labels.cuda()
            self.inputs = self.inputs.cuda()

        self.inputs = Variable(self.inputs)
        self.labels = Variable(self.labels)

        # logging testing
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TestLogger.txt',
            self.save_results
        )
        self.params_loss = ['Loss', 'ACC']
        self.log_loss.register(self.params_loss)

        # monitor testing
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'Loss': {'dtype': 'running_mean'},
            'ACC': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # visualize testing
        self.visualizer = plugins.Visualizer(self.port, self.env, 'Test')
        self.params_visualizer = {
            'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss',
                    'layout': {'windows': ['train', 'test'], 'id': 1}},
            'ACC': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'ACC',
                    'layout': {'windows': ['train', 'test'], 'id': 1}},
        }
        self.visualizer.register(self.params_visualizer)

        # display training progress
        self.print_formatter = 'Test [%d/%d]] '
        for item in self.params_loss:
            self.print_formatter += item + " %.4f "

        self.evalmodules = []
        self.losses = {}

    def model_eval(self):
        for key in list(self.model):
            self.model[key].eval()

    def expanded_pairwise_distances(self, x, y=None):
        '''
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        '''
        if y is not None:
             differences = x.unsqueeze(1) - y.unsqueeze(0)
        else:
            differences = x.unsqueeze(1) - x.unsqueeze(0)
        distances = torch.sum(differences * differences, -1)
        return distances

    def test(self, epoch, dataloader):
        dataloader = dataloader['test']
        self.monitor.reset()
        torch.cuda.empty_cache()

        # switch to eval mode
        self.model_eval()

        end = time.time()

        target_dim = self.args.model_options[-1]['out_dims']

        # extract features
        for i, (inputs,testlabels) in enumerate(dataloader):
            # keeps track of data loading time
            data_time = time.time() - end

            ############################
            # Evaluate Network
            ############################

            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(testlabels.size()).copy_(testlabels)

            # output features
            outputs = {}
            keys = list(self.model)
            for j,key in enumerate(keys):
                if j == 0:
                    outputs[key] = self.model[key](self.inputs)
                else:
                    outputs[key] = self.model[key](outputs[keys[j-1]])
            
            if i == 0:
                features = outputs[str(target_dim)].data.cpu().numpy()
                input_features = self.inputs.data.cpu().numpy()
                labels = self.labels.data
            else:
                features = np.vstack((features, outputs[str(target_dim)].data.cpu().numpy()))
                input_features = np.vstack((input_features, self.inputs.data.cpu().numpy()))
                labels = torch.cat((labels, self.labels.data), 0) 

        labels = labels.squeeze()
        
        acc,_,_ = self.evaluation(features)
        loss = 1 - acc
        self.losses['Loss'] = loss
        self.losses['ACC'] = acc
        batch_size = 1
        self.monitor.update(self.losses, batch_size)

        # print batch progress
        print(self.print_formatter % tuple(
            [epoch + 1, self.nepochs] +
            [self.losses[key] for key in self.params_monitor]))
            
        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the visualization
        self.visualizer.update(loss)

        # return self.monitor.getvalues('Loss')
        return acc

    def extract_features(self, dataloader):
        dataloader = dataloader['test']
        torch.cuda.empty_cache()
        self.model_eval()

        target_dim = self.args.model_options[-1]['out_dims']

        # extract features
        for i, (inputs,testlabels) in enumerate(dataloader):

            self.inputs.data.resize_(inputs.size()).copy_(inputs)
            self.labels.data.resize_(testlabels.size()).copy_(testlabels)

            # output features
            outputs = {}
            keys = list(self.model)
            for j,key in enumerate(keys):
                if j == 0:
                    outputs[key] = self.model[key](self.inputs)
                else:
                    outputs[key] = self.model[key](outputs[keys[j-1]])
            
            if i == 0:
                features = outputs[str(target_dim)].data
                labels = self.labels.data
            else:
                features = torch.cat((features, outputs[str(target_dim)].data), 0)
                labels = torch.cat((labels, self.labels.data), 0)

        labels = labels.squeeze()
        feat = features.data.cpu().numpy()
        label = labels.data.cpu().numpy()

        # save the features
        subdir = os.path.dirname(self.args.feat_savepath)
        if os.path.isdir(subdir) is False:
            os.makedirs(subdir)

        if self.args.feat_savepath.endswith('npz'):
            np.savez(self.args.feat_savepath, feat=feat, label=label)
        elif self.args.feat_savepath.endswith('mat'):
            scipy.io.savemat(self.args.feat_savepath, mdict={'feat': feat})
        elif self.args.feat_savepath.endswith('npy'):
            np.save(self.args.feat_savepath, feat)
        else:
            raise(RuntimeError('The saving format is not supoorted!'))
