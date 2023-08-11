import os

import networkx as nx
import numpy as np
import scipy
import pickle
from tqdm import tqdm

def load_assist_data(prefix='data/preprocessed/assist_processed-out2-feat2-trainless-path3/'):
    G00 = nx.read_adjlist(prefix + '/0/0-1-0.adjlist', create_using=nx.MultiDiGraph)
    G10 = nx.read_adjlist(prefix + '/1/1-0-1.adjlist', create_using=nx.MultiDiGraph)
    idx00 = np.load(prefix + '/0/0-1-0_idx.npy')
    idx10 = np.load(prefix + '/1/1-0-1_idx.npy')
    features_0 = scipy.sparse.load_npz(prefix + '/features_0.npz')
    features_1 = scipy.sparse.load_npz(prefix + '/features_1.npz')
    adjP = scipy.sparse.load_npz(prefix + '/adjP.npz')
    type_mask = np.load(prefix + '/node_types.npy')
    labels = np.load(prefix + '/labels.npy')
    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    return [[G00], [G10]], \
           [[idx00], [idx10]], \
           [features_0, features_1],\
           adjP, \
           type_mask,\
           labels,\
           train_val_test_idx

