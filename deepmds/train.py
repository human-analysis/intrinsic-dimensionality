# train.py

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import plugins
import itertools

import os

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

    def model_train(self):
        for key in list(self.model):
            self.model[key].train()

    def train(self, epoch, dataloader_imp, dataloader_gen):
        dataloader_imp = dataloader_imp['train']
        dataloader_gen = dataloader_gen['train']
        self.monitor.reset()

        # switch to train mode
        self.model_train()

        end = time.time()

        gen_extended = itertools.cycle(dataloader_gen)
        zipped_loader = itertools.zip_longest(gen_extended, dataloader_imp, fillvalue=None)
        combined_loader = itertools.takewhile((lambda t: t != None), zipped_loader)

        for i, (data) in enumerate(combined_loader):
            # keeps track of data loading time
            data_time = time.time() - end

            images_imp = data[1]
            images_gen = data[0]
            if images_imp is None:
                break
            in_dims = images_imp.size(2)

            # concatenate imposter and genuine
            inputs = torch.cat((images_imp, images_gen),0)
            inputs = inputs.view(-1, in_dims)

            ############################
            # Update network
            ############################

            batch_size = images_imp.size(0)
            self.inputs.data.resize_(inputs.size()).copy_(inputs)

            # input pairs
            input_pair1 = self.inputs[0::2,:]
            input_pair2 = self.inputs[1::2,:]

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
                output_pair1 = outputs[keys[-1]][0::2,:]
                output_pair2 = outputs[keys[-1]][1::2,:]
                loss = self.criterion(input_pair1,input_pair2,output_pair1, output_pair2)
            else:
                for j,key in enumerate(keys):
                    output_pair1 = outputs[key][0::2,:]
                    output_pair2 = outputs[key][1::2,:]
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
                [epoch + 1, self.nepochs, i, len(dataloader_imp)] +
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
