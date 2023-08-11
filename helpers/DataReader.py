# -*- coding: UTF-8 -*-

import os
import sys
import math
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
import copy


class DataReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='../data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='assist09',
                            help='Choose a dataset.')  # assist09, ednet, ednet2
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        parser.add_argument('--max_step', type=int, default=200,
                            help='Max time steps per sequence.')
        parser.add_argument('--kfold', type=int, default=5,
                            help='K-fold number.')
        return parser

    def __init__(self, args):
        self.prefix = args.path
        self.sep = args.sep
        self.k_fold = args.kfold
        self.max_step = int(args.max_step)
        self.dataset = args.dataset
        self.data_df = {
            'train': pd.DataFrame(), 'test': pd.DataFrame()
        }

        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.inter_df = pd.read_csv(os.path.join(self.prefix, self.dataset, 'interactions_our.csv'), sep=self.sep)

        user_wise_dict = dict()
        cnt, n_inters = 0, 0
        for user, user_df in self.inter_df.groupby('user_id'):
            array = np.random.randint(3, 200, size=1)
            df = user_df[:array[0]]  # consider the first 50 interactions
            if len(user_df) > 2:
                df = user_df[:self.max_step]
                user_wise_dict[cnt] = {
                    'user_id': user,
                    'skill_seq': df['skill_id'].values.tolist(),
                    'correct_seq': np.array([round(x) for x in df['correct']]),
                    'time_seq': df['timestamp'].values.tolist(), # timestamp ms_first_response
                    'problem_seq': df['problem_id'].values.tolist()
                }
                cnt += 1
                n_inters += len(df)
        self.user_seq_df = pd.DataFrame.from_dict(user_wise_dict, orient='index')

        self.n_users = max(self.inter_df['user_id'].values) + 1
        self.n_skills = len(self.inter_df['skill_id'].unique())
        self.n_problems = len(self.inter_df['problem_id'].unique())


        logging.info('"n_users": {}, "n_skills": {}, "n_problems": {}, "n_interactions": {}'.format(
            self.n_users, self.n_skills, self.n_problems, n_inters
        ))

    def gen_fold_data(self, split=0.8):
        split = 0.6
        data_length = 0
        n_examples = len(self.user_seq_df)
        split_point = int(n_examples * split)
        self.data_df['test'] = self.user_seq_df.iloc[split_point:]
        self.data_df['train'] = self.user_seq_df.iloc[:split_point]
        for i in range(n_examples):
            data_length += len(self.user_seq_df.iloc[i]['skill_seq'])

        logging.info('# Train: {}, # Test: {}, # data_length: {}'.format(
            len(self.data_df['train']), len(self.data_df['test']), data_length
        ))

    def show_columns(self):
        logging.info('Data columns:')
        logging.info(self.user_seq_df.iloc[np.random.randint(0, len(self.user_seq_df))])


if __name__ == '__main__':
    logging.basicConfig(filename='../../log/test.txt', level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    parser = argparse.ArgumentParser(description='')
    parser = DataReader.parse_data_args(parser)

    args, extras = parser.parse_known_args()
    args.path = '../../data/'
    np.random.seed(2019)
    data = DataReader(args)
    data.gen_fold_data(k=0)
    data.show_columns()

    corpus_path = os.path.join(args.path, args.dataset, 'Corpus_nokfold_{}.pkl'.format(args.max_step))
    logging.info('Save corpus to {}'.format(corpus_path))
    pickle.dump(data, open(corpus_path, 'wb'))
