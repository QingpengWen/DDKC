# -*- coding: utf-8 -*-
"""
@CreateTime :       2024/05/28 18:38
@File       :       model_ddkc.py
@Software   :       PyCharm
@Framework  :       Pytorch
@description:       main structure of DDKC
@LastModify :       2024/07/30 23:35
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from models.modular.ISA import IA, SA, BiLSTM, FFN
from models.modular.kan import KAN
from utils.config import *

def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

class DDKC(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob):
        """Self-attentive knowledge tracing.

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
        """
        super(DDKC, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos

        # TODO: Embbeding
        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)
        self.label_embeds = nn.Embedding(num_items + 1, 128, padding_idx=0)
        # TODO: Modules       
        self.KAN = KAN(width=[3, 5, 1], grid=5, k=3, device='cuda', seed=0)
        self.dropout = nn.Dropout(p=drop_prob)  
        self.SA = SA(INPUT_SIZE=100, HIDDEN_SIZE=128, DROPOUT_R=drop_prob, FF_SIZE=100, n_heads=num_heads)  # n_heads=8
        self.IA = IA(INPUT_SIZE=128, HIDDEN_SIZE=128, DROPOUT_R=drop_prob, FF_SIZE=100, n_heads=num_heads)  # n_heads=8
        self.IA_alpha = IA(INPUT_SIZE=1, HIDDEN_SIZE=128, DROPOUT_R=drop_prob, FF_SIZE=100, n_heads=num_heads)  # n_heads=8
        self.SA_KT = SA(INPUT_SIZE=128, HIDDEN_SIZE=128, DROPOUT_R=drop_prob, FF_SIZE=100, n_heads=num_heads)  # n_heads=8
        self.IA_KT = IA(INPUT_SIZE=128, HIDDEN_SIZE=128, DROPOUT_R=drop_prob, FF_SIZE=100, n_heads=num_heads)  # n_heads=8
        self.lstm = BiLSTM(input_size=128, hidden_size=256, output_size=128)
        self.ffn = FFN(HIDDEN_SIZE=128, FF_SIZE=100, DROPOUT_R=drop_prob)
        # TODO: Linear out
        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.lin_sl = nn.Linear(in_features=128, out_features=128, bias=False)
        self.lin_out = nn.Linear(in_features=128, out_features=1, bias=False)
        # TODO: register stable learning
        self.register_buffer('pre_weight1', torch.ones(args.batch_size, 1))

    def get_dyn_diff(self, d_count, d_correct, d_skill_correct):
        compre_difficulty = torch.cat([d_count.unsqueeze(-1), d_correct.unsqueeze(-1), d_skill_correct.unsqueeze(-1)], dim=-1)  # [batch, 200, 3]
        compre_difficulties = compre_difficulty.view(-1, compre_difficulty.shape[-1])
        dyn_compre_difficulty = self.KAN(compre_difficulties)
        max_value, max_indices = torch.max(dyn_compre_difficulty, dim=0)
        min_value, min_indices = torch.min(dyn_compre_difficulty, dim=0)
        dyn_compre_difficulty = (dyn_compre_difficulty - min_value) / (max_value - min_value)
        return dyn_compre_difficulty, compre_difficulty,  compre_difficulties
    
    def get_embs(self, item_inputs, skill_inputs):
        item_input = self.item_embeds(item_inputs.long())
        skill_input = self.skill_embeds(skill_inputs.long())
        return item_input, skill_input

    def get_kcs(self,  item_inputs, skill_inputs, level_inputs):
        q_t = self.SA(item_inputs)[0]  # [batch, 200, 128]
        s_t = self.SA(skill_inputs)[0]  # [batch, 200, 128]
        qs_t = self.IA(q_t, s_t)[0]
        c_t, c_t_weight = self.IA_alpha(level_inputs, qs_t)  # [batch, 200, 128]
        c_t = self.lstm(c_t)
        return c_t, c_t_weight
    
    def kt_perception(self, c_t):
        c_1 = c_t.unsqueeze(2)  # [batch, 200, 1, 128]
        c_2 = c_t[0:, 1, :].unsqueeze(1)
        for i in range(2, c_t.shape[1]):
            c_i = c_t[0:, i, :].unsqueeze(1)
            c_2 = torch.cat((c_2, c_i), 1)  # [batch, 200, 128]
        c_2 = torch.cat((c_2, c_t[0:, 0, :].unsqueeze(1)), 1)  # [batch, 200, 128]
        c_2 = c_2.unsqueeze(2)  # [batch, 200, 1, 128]
        c_t1 = c_1[0, :, :]  # [200, 1, 128]
        c_t2 = c_2[0, :, :]  # [200, 1, 128]
        h_t1 = self.SA_KT(c_t1)[0]  # [200, 1, 128]
        h_t2 = self.SA_KT(c_t2)[0]  # [200, 1, 128]
        h_t12 = self.IA_KT(h_t1, h_t2)[0]
        y_1 = self.SA_KT(h_t12)[0]
        y_1 = y_1.squeeze(1)
        y_1 = y_1.unsqueeze(0)
        y = y_1
        for j in range(1, c_1.shape[0]):
            c_i = c_1[j, :, :]  # [200, 1, 128]
            c_j = c_2[j, :, :]  # [200, 1, 128]
            h_i = self.SA_KT(c_i)[0]
            h_j = self.SA_KT(c_j)[0]
            h_ij = self.IA_KT(h_i, h_j)[0]
            y_i = self.SA_KT(h_ij)[0]
            y_i = y_i.squeeze(1)
            y_i = y_i.unsqueeze(0)
            y = torch.cat((y, y_i), 0)
        return y
  
    def forward(self, item_input, skill_input, label_input, d_count=None, d_correct=None, d_skill_correct=None, pre_weight=None):
        """
        <<<Model Main Structure>>>
        Inputs:
            item_inputs: [batch, 200]
            skill_inputs: [batch, 200] 
            label_inputs: [batch, 200] 
            level_inputs: [batch, 200] 
            item_ids: [batch, 200] 
            skill_ids: [batch, 200] 
            levels: [batch, 200] 
        Returns:
            output: [batch, 200, 1]
        """
        # TODO: comculate dynamic comprehensive difficulty
        dyn_compre_difficulty, compre_difficulty,  compre_difficulties = self.get_dyn_diff(d_count, d_correct, d_skill_correct)
        mean_dyn_compre_difficulty = dyn_compre_difficulty.mean()
        mean_dyn_compre_difficulty = mean_dyn_compre_difficulty.item()
                      
        # TODO: Embedding
        item_inputs, skill_inputs = self.get_embs(item_input, skill_input)
        pre_labelout = self.label_embeds(label_input.long())  # [batch,200,128]          
        level_inputs = dyn_compre_difficulty.view(compre_difficulty.shape[0],
                                                  compre_difficulties.shape[0] // compre_difficulty.shape[0],
                                                  dyn_compre_difficulty.shape[-1])  # [batch,200,1]

        # TODO: Knowledgeable Cell
        c_t, c_t_weight = self.get_kcs(item_inputs, skill_inputs, level_inputs)
        
        # TODO: Knowledge Tracing perception
        y = self.kt_perception(c_t)
        
        # TODO: Prediction Layer
        if pre_weight != None:
            # with Stable Learning
            residual = torch.mul(c_t, pre_weight.unsqueeze(1))
            outputs = y + self.dropout(F.relu(self.lin_sl(residual)))
            outputs = self.dropout(F.relu(self.ffn(outputs)))
        else:
            # Without Stable Learning
            outputs = y + self.dropout(F.relu(self.lin_sl(c_t)))
            outputs = self.dropout(F.relu(self.ffn(outputs)))
        
        return self.lin_out(outputs), c_t, pre_labelout, c_t_weight, mean_dyn_compre_difficulty # output->[batch, 200, 1], outputs->[batch, 200, 128]

    def plot(self, folder="./figures", beta=3, mask=False, mode="unsupervised", scale=0.5, tick=False, sample=False, in_vars=None, out_vars=None, title=None):
        self.KAN.plot()

    def prune(self, threshold=1e-2, mode="auto", active_neurons_id=None):
        self.KAN.prune()

    def symbolic_formula(self, floating_digit=2, var=None, normalizer=None, simplify=False, output_normalizer=None):
        self.KAN.auto_symbolic(lib=['exp', 'sin', 'x^2'])
        formula = self.KAN.symbolic_formula(floating_digit=2)
        print("formula: ", formula)
        return formula
        