# -*- coding: UTF-8 -*-
import pickle

import torch
import logging
from time import time
from tqdm import tqdm
import gc
import numpy as np
import pandas as pd
import copy
import os

from utils import utils


class KTRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=50,
                            help='Number of epochs.')
        parser.add_argument('--lr', type=float, default=5e-3,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=128,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2', type=float, default=0.,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--optimizer', type=str, default='Adamw',
                            help='optimizer: GD, Adam, Adagrad, Adadelta')
        parser.add_argument('--metric', type=str, default='auc, accuracy, rmse, f1, mae, recall, precision',
                            help='metrics: AUC, F1, Accuracy, Recall, Presicion;'
                                 'The first one will be used to determine whether to early stop')
        return parser

    def __init__(self, args):
        self.optimizer_name = args.optimizer
        self.learning_rate = args.lr
        self.name = args.model_name
        self.max_step = str(args.max_step)
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.metrics = args.metric.strip().lower().split(',')
        self.time = None
        self.embedding_size = 256
        self.dataset = args.dataset
        self.max_length = args.max_step
        self.name = args.model_name

        self.test_results = {}
        self.stat = {}
        for i in range(len(self.metrics)):
            self.metrics[i] = self.metrics[i].strip()
            # self.valid_results[self.metrics[i]] = list()
            self.test_results[self.metrics[i]] = list()
        self.test_results['loss'] = list()
        self.stat['predict'] = list()
        self.stat['label'] = list()

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adadelta':
            logging.info("Optimizer: Adadelta")
            optimizer = torch.optim.Adadelta(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        elif optimizer_name == 'adamw':
            logging.info("Optimizer: Adamw")
            optimizer = torch.optim.AdamW(model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        else:
            raise ValueError("Unknown Optimizer: " + self.optimizer_name)
        return optimizer

    def predict(self, model, corpus, set_name):
        model.eval()
        predictions, labels = [], []
        batches = model.prepare_batches(corpus, corpus.data_df[set_name], self.eval_batch_size, phase=set_name)
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = model.batch_to_gpu(batch)
            outdict = model(batch)
            prediction, label = outdict['prediction'], outdict['label']
            predictions.extend(prediction.detach().cpu().data.numpy())
            labels.extend(label.detach().cpu().data.numpy())
        return np.array(predictions), np.array(labels)

    def fit(self, model, corpus, epoch_train_data, epoch=-1):  # fit the results for an input set
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        model.train()
        loss_lst = list()
        batches = model.prepare_batches(corpus, epoch_train_data, self.batch_size, phase='train')
        for batch in tqdm(batches, leave=False, ncols=100, mininterval=1, desc='Epoch %5d' % epoch):
            batch = model.batch_to_gpu(batch)
            model.optimizer.zero_grad()
            output_dict = model(batch)
            loss = model.loss(batch, output_dict)
            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())
        model.eval()
        return np.mean(loss_lst)

    # def eva_termination(self, model):
    #     # valid = self.valid_results[self.metrics[0]]
    #     if len(valid) > 20 and utils.non_increasing(valid[-10:]):
    #         return True
    #     elif len(valid) - valid.index(max(valid)) > 20:
    #         return True
    #     return False

    def train(self, model, corpus):
        assert(corpus.data_df['train'] is not None)
        self._check_time(start=True)

        try:
            for epoch in range(self.epoch):
                gc.collect()
                self._check_time()
                epoch_train_data = copy.deepcopy(corpus.data_df['train'])
                epoch_train_data = epoch_train_data.sample(frac=1).reset_index(drop=True)
                loss = self.fit(model, corpus, epoch_train_data, epoch=epoch + 1)
                del epoch_train_data
                training_time = self._check_time()

                # output validation
                # valid_result = self.evaluate(model, corpus, 'dev')
                test_result, predictions, labels = self.evaluate(model, corpus, 'test')
                testing_time = self._check_time()

                for metric in self.metrics:
                    # self.valid_results[metric].append(valid_result[metric])
                    self.test_results[metric].append(test_result[metric])
                self.test_results['loss'].append(loss)

                if epoch == 100:
                    self.stat['predict'].append(predictions)
                    self.stat['label'].append(labels)
                    with open("../data/stat_result_{}.pkl".format(self.name + '_' + self.max_step), "wb") as tf:
                        pickle.dump(self.stat, tf)
                    with open("../data/result_{}.pkl".format(self.name + '_' + self.max_step), "wb") as tf:
                        pickle.dump(self.test_results, tf)
                    # with open("../data/result_{}.pkl".format(self.name + '_' + self.max_step), "rb") as tf:
                    #     new_dict = pickle.load(tf)
                logging.info("Epoch {:<3} loss={:<.8f} [{:<.1f} s]\t  test=({}) [{:<.1f} s] ".format(
                             epoch + 1, loss, training_time,
                             utils.format_metric(test_result), testing_time))

                # if max(self.valid_results[self.metrics[0]]) == self.valid_results[self.metrics[0]][-1]:
                #     model.save_model()
                # if self.eva_termination(model) and self.early_stop:
                #     logging.info("Early stop at %d based on validation result." % (epoch + 1))
                #     break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best validation result across iterations
        best_test_score = max(self.test_results[self.metrics[0]])
        best_epoch = self.test_results[self.metrics[0]].index(best_test_score)
        # best_valid_score = max(self.valid_results[self.metrics[0]])
        # best_epoch = self.valid_results[self.metrics[0]].index(best_valid_score)
        test_res_dict = dict()
        # for metric in self.metrics:
        #     # valid_res_dict[metric] = self.valid_results[metric][best_epoch]
        #     test_res_dict[metric] = self.test_results[metric][best_epoch]
        # logging.info("\nBest Iter(dev)=  %5d\t  test=(%s) [%.1f s] "
        #              % (best_epoch + 1,
        #                 utils.format_metric(test_res_dict),
        #                 self.time[1] - self.time[0]))

        best_test_score = max(self.test_results[self.metrics[0]])
        best_epoch = self.test_results[self.metrics[0]].index(best_test_score)
        for metric in self.metrics:
            # valid_res_dict[metric] = self.valid_results[metric][best_epoch]
            test_res_dict[metric] = self.test_results[metric][best_epoch]
        logging.info("Best Iter(test)= %5d\t test=(%s) [%.1f s] \n"
                     % (best_epoch + 1,
                        utils.format_metric(test_res_dict),
                        self.time[1] - self.time[0]))
        model.load_model()

    def evaluate(self, model, corpus, set_name):  # evaluate the results for an input set
        predictions, labels = self.predict(model, corpus, set_name)
        return model.pred_evaluate_method(predictions, labels, self.metrics), predictions, labels


    def print_res(self, model, corpus):
        set_name = 'test'
        result = self.evaluate(model, corpus, set_name)
        res_str = utils.format_metric(result)
        return res_str
