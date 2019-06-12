# train.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import plugins
import itertools
import numpy as np
import os
from random import shuffle
import math
import pdb

class Trainer:
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

        self.lr = args.learning_rate
        self.optim_method = args.optim_method
        self.optim_options = args.optim_options
        self.scheduler_method = args.scheduler_method
        self.scheduler_options = args.scheduler_options
        self.weight_loss = args.weight_loss

        # setting optimizer for multiple modules
        self.module_list = nn.ModuleList()
        for key in list(self.model):
            self.module_list.append(self.model[key])
        self.optimizer = getattr(optim, self.optim_method)(
            self.module_list.parameters(), lr=self.lr, **self.optim_options)
        if self.scheduler_method is not None:
            self.scheduler = getattr(optim.lr_scheduler, self.scheduler_method)(
                self.optimizer, **self.scheduler_options
            )

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

        # logging training
        self.log_loss = plugins.Logger(
            args.logs_dir,
            'TrainLogger.txt',
            self.save_results
        )
        self.params_loss = ['LearningRate','Loss']#, 'TAR']
        self.log_loss.register(self.params_loss)

        # monitor training
        self.monitor = plugins.Monitor()
        self.params_monitor = {
            'LearningRate': {'dtype': 'running_mean'},
            'Loss': {'dtype': 'running_mean'},
            # 'TAR': {'dtype': 'running_mean'},
        }
        self.monitor.register(self.params_monitor)

        # visualize training
        self.visualizer = plugins.Visualizer(self.port, self.env, 'Train')
        self.params_visualizer = {
            'LearningRate': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'learning_rate',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            'Loss': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'loss',
                    'layout': {'windows': ['train', 'test'], 'id': 0}},
            # 'TAR': {'dtype': 'scalar', 'vtype': 'plot', 'win': 'mAP',
            #         'layout': {'windows': ['train', 'test'], 'id': 0}},
        }
        self.visualizer.register(self.params_visualizer)

        # display training progress
        self.print_formatter = 'Train [%d/%d][%d/%d] '
        for item in self.params_loss:
            self.print_formatter += item + " %.4f "

        self.evalmodules = []
        self.losses = {}

    def get_imp_gen_pairs(self, batch_size):
        batch_imposter_image = self.args.batch_imposter_image
        batch_genuine_image = self.args.batch_genuine_image
        step_length = batch_imposter_image + batch_genuine_image
        # get imposter index
        base_idx = np.array(list(itertools.combinations(list(range(0,batch_size)), 2))) # ?x2
        inner_idx = np.reshape(np.indices((batch_imposter_image, batch_imposter_image)), (2,-1)) # 2x?
        imp_idx = np.zeros((base_idx.shape[0]*inner_idx.shape[1], 2))
        idx = 0
        for i in range(base_idx.shape[0]):
            for j in range(inner_idx.shape[1]):
                imp_idx[idx,0] = base_idx[i,0]*step_length + inner_idx[0,j]
                imp_idx[idx,1] = base_idx[i,1]*step_length + inner_idx[1,j]
                idx += 1
        # get genuine index
        base_idx = np.array(list(itertools.combinations(list(range(0, batch_genuine_image)), 2)))
        gen_idx = np.zeros((base_idx.shape[0]*batch_size, base_idx.shape[1]))
        for i in range(batch_size):
            temp = np.zeros((base_idx.shape))
            temp[:,0] = i*step_length + base_idx[:,0] + batch_imposter_image
            temp[:,1] = i*step_length + base_idx[:,1] + batch_imposter_image
            gen_idx[i*base_idx.shape[0]:(i+1)*base_idx.shape[0],:] = temp[:]
        gen_idx = gen_idx.astype('int')

        imp_idx = torch.LongTensor(imp_idx)
        gen_idx = torch.LongTensor(gen_idx)
        if self.args.cuda:
            imp_idx = imp_idx.cuda()
            gen_idx = gen_idx.cuda()
        return imp_idx, gen_idx

    def model_train(self):
        for key in list(self.model):
            self.model[key].train()

    def train(self, epoch, dataloader):
        dataloader = dataloader['train']
        self.monitor.reset()

        # switch to train mode
        self.model_train()

        end = time.time()

        index_list = list(range(len(dataloader)))
        shuffle(index_list)
        num_batch = math.ceil(float(len(index_list))/float(self.args.batch_size))
        for i in range(num_batch):
            # keeps track of data loading time
            data_time = time.time() - end

            start_index = i*self.args.batch_size
            end_index = min((i+1)*self.args.batch_size, len(index_list))
            batch_size = end_index - start_index
            data = []

            # iter_info = []
            for ind in index_list[start_index:end_index]:
                rawdata = dataloader[ind]
                data.append(torch.stack(rawdata))

            for j in range(len(data)):
                if j == 0:
                    images = data[j]
                else:
                    images = torch.cat((images, data[j]), 0)
            in_dims = images.size(1)

            # concatenate imposter and genuine
            inputs = images.view(-1, in_dims)

            ############################
            # Update network
            ############################

            self.inputs.data.resize_(inputs.size()).copy_(inputs)

            # get imposter and genuine indices
            imp_idx, gen_idx = self.get_imp_gen_pairs(batch_size)
            idx = torch.cat((imp_idx, gen_idx), 0)

            # input pairs
            input_pair1 = self.inputs[idx[:,0],:]
            input_pair2 = self.inputs[idx[:,1],:]

            # output pairs
            outputs = {}
            keys = list(self.model)
            for j,key in enumerate(keys):
                if j == 0:
                    outputs[key] = self.model[key](self.inputs)
                else:
                    outputs[key] = self.model[key](outputs[keys[j-1]])

            # loss
            if self.args.finetune:
                output_pair1 = outputs[keys[-1]][idx[:,0],:]
                output_pair2 = outputs[keys[-1]][idx[:,1],:]
                loss = self.criterion(input_pair1,input_pair2,output_pair1, output_pair2)
            else:
                for j,key in enumerate(keys):
                    output_pair1 = outputs[key][idx[:,0],:]
                    output_pair2 = outputs[key][idx[:,1],:]
                    if j == 0:
                        loss = self.weight_loss[j]*self.criterion(
                            input_pair1,input_pair2,output_pair1,output_pair2)
                    else:
                        loss += self.weight_loss[j]*self.criterion(
                            pre_output_pair1,pre_output_pair2,output_pair1,output_pair2)
                    pre_output_pair1 = output_pair1
                    pre_output_pair2 = output_pair2

            self.optimizer.zero_grad()
            if loss is not None and loss != 0:
                loss.backward()
                self.optimizer.step()
                self.losses['Loss'] = loss.item()
            else:
                self.losses['Loss'] = loss
            for param_group in self.optimizer.param_groups:
                self.cur_lr = param_group['lr']
            self.losses['LearningRate'] = self.cur_lr
            self.monitor.update(self.losses, batch_size)

            # print batch progress
            print(self.print_formatter % tuple(
                [epoch + 1, self.nepochs, i, num_batch] +
                [self.losses[key] for key in self.params_monitor]))

        # update the log file
        loss = self.monitor.getvalues()
        self.log_loss.update(loss)

        # update the visualization
        self.visualizer.update(loss)

        # update the learning rate
        if self.scheduler_method is not None:
            if self.scheduler_method == 'ReduceLROnPlateau':
                self.scheduler.step(loss['Loss'])
            else:
                self.scheduler.step()

        return self.monitor.getvalues('Loss')
