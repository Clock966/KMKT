
import torch
import time
from tqdm import tqdm
import logging
from sklearn.metrics import *
import numpy as np
import torch.nn.functional as F
import os

from utils.utils import *
class RNNModelScratch:
    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward_fn):
        """Defined in :numref:`sec_rnn_scratch`"""
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size*2, num_hiddens)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state, memory_attention):
        return self.forward_fn(X, state, memory_attention, self.params)
    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)