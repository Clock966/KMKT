# -*- coding: UTF-8 -*-
import math
import os
import time
import numpy as np
from torch import nn
from tqdm import tqdm
import torch
import torch.nn.functional as F

from models.BaseModel import BaseModel
from utils import utils
from models.relstm import RNNModelScratch
import tqdm


class KMKT_09(BaseModel):
    extra_log_args = ['hidden_size', 'num_layer']

    @staticmethod
    def parse_model_args(parser, model_name='KMKT_09'):
        parser.add_argument('--emb_size', type=int, default=256,
                            help='Size of embedding vectors.')
        parser.add_argument('--skill_emb_size', type=int, default=256,
                            help='Size of skill embedding vectors.')
        parser.add_argument('--hidden_size', type=int, default=256,
                            help='Size of hidden vectors in LSTM.')
        parser.add_argument('--num_layer', type=int, default=1,
                            help='Number of GRU layers.')
        return BaseModel.parse_model_args(parser, model_name)

    def __init__(self, args, corpus):
        self.problem_num = int(corpus.n_problems)
        self.skill_num = int(corpus.n_skills)
        self.emb_size = args.emb_size
        self.skill_emb_size = args.skill_emb_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.num_layer
        self.dropout = args.dropout
        self.device = args.gpu

        BaseModel.__init__(self, model_path=args.model_path)

    def get_lstm_params(self, vocab_size, num_hiddens):
        num_inputs = vocab_size

        def normal(shape):
            return torch.randn(size=shape).cuda() * 0.01

        def three():
            return (normal((num_inputs, num_hiddens)),
                    normal((num_hiddens, num_hiddens)),
                    torch.zeros(num_hiddens).cuda())

        W_xi, W_hi, b_i = three()  
        W_xf, W_hf, b_f = three()  
        W_xo, W_ho, b_o = three()  
        W_xc, W_hc, b_c = three()  
        W_xm, W_hm, b_m = three()
        W_xl, W_hl, b_l = three()
        params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
                  b_c, W_xm, W_hm, b_m, W_xl, W_hl, b_l]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_lstm_state(self, batch_size, num_hiddens):
        return (torch.zeros(batch_size, num_hiddens).cuda(),
                torch.zeros(batch_size, num_hiddens).cuda())

    def lstm(self, inputs, state, memory_attention, params):
        [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_xm, W_hm, b_m,
         W_xl, W_hl, b_l] = params
        X = inputs
        memory_attention = memory_attention.cuda().type(torch.float32)
        (H, C) = state
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)
        M = torch.sigmoid((X @ W_xm) + (H @ W_hm) + b_m)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)
        C = F * C + I * C_tilda + M * memory_attention 
        L = torch.tanh((X @ W_xl) + (H @ W_hl) + b_l)
        H = L * H + O * torch.tanh(C)  # 0.73867804 0.74726138

        return H, (H, C)

    def begin_state(self, batch_size, device):
        return self.init_lstm_state(batch_size, self.hidden_size)

    def _init_weights(self):
        self.skill_embeddings = torch.nn.Embedding(self.skill_num * 2, self.skill_emb_size, dtype=torch.float32)
        self.relstm = RNNModelScratch(self.emb_size, self.hidden_size, self.get_lstm_params, self.init_lstm_state,
                                      self.lstm)
        self.fin = torch.nn.Linear(1, self.skill_emb_size, bias=False)
        self.fout = torch.nn.Linear(3, self.hidden_size)
        self.attout = torch.nn.Linear(1, self.hidden_size)
        self.out = torch.nn.Linear(self.hidden_size + self.emb_size, 1)
        self.loss_function = torch.nn.BCELoss()
        self.attention = DotProductAttention(dropout=0.5)

    def forward(self, feed_dict):
        pro_sorted = feed_dict['pro_seq']
        skill_sorted = feed_dict['skill_seq']  # [batch_size, max_step]
        labels_sorted = feed_dict['label_seq']  # [batch_size, max_step]
        lengths = feed_dict['length']  # [batch_size]
        repeated_time_gap_seq = feed_dict['repeated_time_gap_seq']  # [batch_size, max_step]

        zero_dict = torch.zeros((1, self.hidden_size)).cuda()
        memory_dict = dict.fromkeys(range(self.skill_num * 2), 0)
        for key in memory_dict.keys():
            memory_dict[key] = zero_dict

        embed_data = np.load(
            os.path.join('./models/data/preprocessed/graph_pro_assist09_path3-sigmoid.npz')) 
        pro_embed = embed_data['pro']
        pro_embed = torch.from_numpy(pro_embed)
        zero = torch.zeros(1, self.emb_size)
        pro_embed = torch.cat((pro_embed, zero), 0).cuda()
        self.pro_embeddings = torch.nn.Embedding.from_pretrained(pro_embed)
        embed_history_i = self.pro_embeddings(pro_sorted)
        cat = torch.unsqueeze(labels_sorted, 2)
        cat_label = cat
        for i in range(embed_history_i.shape[2] - 1):
            cat_label = torch.cat((cat_label, cat), dim=2)
        embed_history_i = torch.cat((embed_history_i, cat_label), dim=2)

        state = None
        if state is None:
            state = self.begin_state(1, device=self.device)
        else:
            for s in state:
                s.detach_()
        output_series = []
        repeated_time = self.fin(repeated_time_gap_seq * -1)
        repeated_time = torch.exp(repeated_time)
        for i in range(embed_history_i.size()[0]):
            out_rnn = torch.zeros(1, self.hidden_size).cuda()
            for j in range(lengths[i] - 1):
                skill_memory = int(skill_sorted[i][j].cpu().numpy())
                queries = self.skill_embeddings(skill_memory + labels_sorted[i][j] * self.skill_num).cuda().type(
                    torch.float32)
                keys = self.skill_embeddings.weight.cuda().type(torch.float32)
                values = torch.reshape(torch.stack(list(memory_dict.values()), dim=2), [-1, self.hidden_size]).cuda()

                delta = repeated_time[i][j][:]
                memory_attention = self.attention(delta.cuda(), queries, keys, values)
                del values
                hx, state = self.relstm(embed_history_i[i][j:j + 1], state, memory_attention)
                memory_dict[skill_memory] = state[1].detach()
                out_rnn = torch.cat((out_rnn, hx), 0)
            if lengths[i] < max(lengths):
                zero_cat = torch.zeros((max(lengths) - lengths[i]), self.hidden_size).cuda()
                current_output = torch.cat((out_rnn[1:], zero_cat), 0)
            if lengths[i] == max(lengths):
                current_output = out_rnn[1:]
            output_series.append(current_output)
        output = torch.cat(output_series, dim=0).reshape(pro_sorted.size()[0], pro_sorted.size()[1] - 1,
                                                         self.hidden_size)

        output = torch.cat((output, self.pro_embeddings(pro_sorted)[:, 1:, :]), 2)
        pred_vector = self.out(output)
        pred_vector = torch.reshape(pred_vector, [-1, 1])

        target_item = torch.reshape(labels_sorted[:, 1:], [-1])
        index = torch.where(torch.ne(target_item, 2))
        filter_target = torch.squeeze(torch.gather(target_item, 0, index[0]), dim=0).double()

        pred_vector = torch.reshape(pred_vector, [-1])
        filter_pred = torch.squeeze(torch.gather(pred_vector, 0, index[0]), dim=0)
        filter_pred = torch.sigmoid(filter_pred)
        out_dict = {'prediction': filter_pred.double(), 'label': filter_target}
        return out_dict

    def loss(self, feed_dict, outdict):
        predictions, labels = outdict['prediction'], outdict['label']
        loss = self.loss_function(predictions, labels)
        return loss

    def get_feed_dict(self, corpus, data, batch_start, batch_size, phase):
        batch_end = min(len(data), batch_start + batch_size)
        real_batch_size = batch_end - batch_start
        user_ids = data['user_id'][batch_start: batch_start + real_batch_size].values
        user_pro = data['problem_seq'][batch_start: batch_start + real_batch_size].values
        user_seqs = data['skill_seq'][batch_start: batch_start + real_batch_size].values
        label_seqs = data['correct_seq'][batch_start: batch_start + real_batch_size].values
        time_seqs = data['time_seq'][batch_start: batch_start + real_batch_size].values

        sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq = \
            self.get_time_features(user_seqs, time_seqs)

        lengths = np.array(list(map(lambda lst: len(lst), user_seqs)))
        indice = np.array(np.argsort(lengths, axis=-1)[::-1])
        inverse_indice = np.zeros_like(indice)
        for i, idx in enumerate(indice):
            inverse_indice[idx] = i
        pad_pro = corpus.n_problems
        pad_skill = corpus.n_skills
        pad_label1 = 0
        pad_label = 2
        feed_dict = {
            'user_id': torch.from_numpy(user_ids[indice]),
            'pro_seq': torch.from_numpy(utils.pad_lst(user_pro[indice], pad_pro)),
            'skill_seq': torch.from_numpy(utils.pad_lst(user_seqs[indice], pad_skill)),  # [batch_size, max_step]
            'label_seq': torch.from_numpy(utils.pad_lst(label_seqs[indice], pad_label)),  # [batch_size, max_step]
            'repeated_time_gap_seq': torch.from_numpy(repeated_time_gap_seq[indice]),  # [batch_size, max_step]
            'sequence_time_gap_seq': torch.from_numpy(sequence_time_gap_seq[indice]),  # [batch_size, max_step]
            'past_trial_counts_seq': torch.from_numpy(past_trial_counts_seq[indice]),  # [batch_size, max_step]
            'length': torch.from_numpy(lengths[indice]),  # [batch_size]
            'inverse_indice': torch.from_numpy(inverse_indice),
            'indice': torch.from_numpy(indice),
        }
        return feed_dict

    @staticmethod
    def get_time_features(user_seqs, time_seqs):
        skill_max = max([max(i) for i in user_seqs])
        inner_max_len = max(map(len, user_seqs))
        repeated_time_gap_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.float32)
        sequence_time_gap_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.float32)
        past_trial_counts_seq = np.zeros([len(user_seqs), inner_max_len, 1], np.float32)
        for i in range(len(user_seqs)):
            last_time = None
            skill_last_time = [None for _ in range(skill_max)]
            skill_cnt = [0 for _ in range(skill_max)]
            for j in range(len(user_seqs[i])):
                sk = user_seqs[i][j] - 1
                ti = time_seqs[i][j]

                if skill_last_time[sk] is None:
                    repeated_time_gap_seq[i][j][0] = 0
                else:
                    repeated_time_gap_seq[i][j][0] = ti - skill_last_time[sk]
                skill_last_time[sk] = ti

                if last_time is None:
                    sequence_time_gap_seq[i][j][0] = 0
                else:
                    sequence_time_gap_seq[i][j][0] = (ti - last_time)
                last_time = ti

                past_trial_counts_seq[i][j][0] = (skill_cnt[sk])
                skill_cnt[sk] += 1

        repeated_time_gap_seq[repeated_time_gap_seq < 0] = 1
        sequence_time_gap_seq[sequence_time_gap_seq < 0] = 1
        repeated_time_gap_seq[repeated_time_gap_seq == 0] = 1e4
        sequence_time_gap_seq[sequence_time_gap_seq == 0] = 1e4
        past_trial_counts_seq += 1
        sequence_time_gap_seq *= 1.0 / 60
        repeated_time_gap_seq *= 1.0 / 60

        sequence_time_gap_seq = np.log(sequence_time_gap_seq)
        repeated_time_gap_seq = np.log(repeated_time_gap_seq)
        past_trial_counts_seq = np.log(past_trial_counts_seq)
        return sequence_time_gap_seq, repeated_time_gap_seq, past_trial_counts_seq


def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                          value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# @save
class DotProductAttention(nn.Module):

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
    def forward(self, delta, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        delta_query = delta * queries
        scores = torch.matmul(delta_query, keys.transpose(0, 1)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.matmul(self.dropout(self.attention_weights), values)
