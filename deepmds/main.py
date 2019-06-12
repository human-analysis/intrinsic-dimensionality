# main.py

import os
import sys
import traceback
import torch
import random
import config
import utils
from model import Model
from dataloader import Dataloader
from checkpoints import Checkpoints

args = config.parse_args()

from test import Tester

if args.train_by_class == True:
    from train_classpair import Trainer
else:
    from train import Trainer
import pdb

def main():
    # parse the arguments
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.save_results:
        utils.saveargs(args)

    # initialize the checkpoint class
    checkpoints = Checkpoints(args)

    # Create Model
    models = Model(args)
    model, criterion, evaluation = models.setup(checkpoints)

    for key in list(model):
        print('Model-{dim}:\n\t{model}\nTotal params:\n\t{npar:.2f}M'.format(
              dim=key,
              model=args.model_type,
              npar=sum(p.numel() for p in model[key].parameters()) / 1000000.0))

    # load imposter pairs
    if args.train_by_class == True:
        dataloader = Dataloader(args)
        loaders = dataloader.create('Test')
        loaders['train'] = dataloader.dataset_train
    else:
        pair_index_filename = args.pair_index_filename
        args.pair_index_filename = pair_index_filename[0]
        dataloader = Dataloader(args)
        loaders_imp = dataloader.create('Train')
        # load genuine pairs
        args.pair_index_filename = pair_index_filename[1]
        dataloader = Dataloader(args)
        loaders_gen = dataloader.create('Train')
        # load testing features
        loaders = dataloader.create('Test')

    # The trainer handles the training loop
    trainer = Trainer(args, model, criterion, evaluation)
    # The trainer handles the evaluation on validation set
    tester = Tester(args, model, criterion, evaluation)

    if args.extract_feat:
        tester.extract_features(loaders)
    elif args.just_test:
        acc_test = tester.test(args.epoch_number, loaders)
    else:
        # start training !!!
        acc_best = 0
        loss_best = 999
        acc_test = 0.1
        for epoch in range(args.nepochs):
            epoch  = epoch + args.epoch_number
            print('\nEpoch %d/%d\n' % (epoch + 1, args.nepochs))

            # train for a single epoch
            if args.train_by_class == False:
                loss_train = trainer.train(epoch, loaders_imp, loaders_gen)
                acc_test = tester.test(epoch, loaders)
            else:
                loss_train = trainer.train(epoch, loaders)
                if epoch % 300 == 0:
                    acc_test = tester.test(epoch, loaders)

            if loss_best > loss_train:
                best_model = True
                loss_best = loss_train
                if args.save_results:
                    checkpoints.save(epoch, acc_test, model, best_model)
                    best_model = False


if __name__ == "__main__":
    utils.setup_graceful_exit()
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        # do not print stack trace when ctrl-c is pressed
        pass
    except Exception:
        traceback.print_exc(file=sys.stdout)
    finally:
        traceback.print_exc(file=sys.stdout)
        utils.cleanup()
