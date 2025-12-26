# -*- coding: utf-8 -*-
"""
@CreateTime :       2024/05/28 21:25
@File       :       ISA.py
@Software   :       PyCharm
@Framework  :       Pytorch
@description:       module for DDKC
@LastModify :       2024/07/30 23:35
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# full-connected layer
class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=None, use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_features=in_size, out_features=out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x

class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0., use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(
            in_size=in_size,
            out_size=mid_size,
            dropout_r=dropout_r,
            use_relu=use_relu
        )
        self.linear = nn.Linear(in_features=mid_size, out_features=out_size)

    def forward(self, x):
        return self.linear(self.fc(x))

class FFN(nn.Module):
    def __init__(self, HIDDEN_SIZE, FF_SIZE, DROPOUT_R=None):
        super(FFN, self).__init__()
        self.mlp = MLP(
            in_size=HIDDEN_SIZE,
            mid_size=FF_SIZE,
            out_size=HIDDEN_SIZE,
            dropout_r=DROPOUT_R,
            use_relu=True
        )
        
    def forward(self, x):
        return self.mlp(x)
    
class MultiheadAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout=None):
        super(MultiheadAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = int(torch.sqrt(torch.FloatTensor([hid_dim // n_heads])))

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim //
                   self.n_heads).permute(0, 2, 1, 3)
        K_T = K.permute(0, 1, 3, 2)
        attention = torch.matmul(Q, K_T) / self.scale
        attention_weight2 = torch.softmax(attention, dim=-1)

        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)

        attention_weights = torch.softmax(attention, dim=-1)
        attention = self.do(attention_weights)
        attention_weights = attention_weight2.detach().cpu()

        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x, attention_weights

# Self Attention
class SA(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, DROPOUT_R, FF_SIZE, n_heads):
        super(SA, self).__init__()

        self.linear_y = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False)
        self.mth_att = MultiheadAttention(HIDDEN_SIZE, n_heads, DROPOUT_R)
        self.ffn = FFN(HIDDEN_SIZE, FF_SIZE, DROPOUT_R)

        self.dropout1 = nn.Dropout(p=DROPOUT_R)
        self.norm1 = LayerNorm(size=HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(p=DROPOUT_R)
        self.norm2 = LayerNorm(size=HIDDEN_SIZE)

    def forward(self, y, y_mask=None):
        """y = FC(y + Multi-head Attention(y,y,y))"""
        y = self.linear_y(y)
        y_att, att_weight = self.mth_att(query=y, key=y, value=y)
        y = self.norm1(
            y + self.dropout1(
                y_att
            )
        )
        y = self.norm2(
            y + self.dropout2(
                self.ffn(y)
            )
        )
        return y, att_weight

# Inter Attention
class IA(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, DROPOUT_R, FF_SIZE, n_heads):
        super(IA, self).__init__()
        self.linear_x = nn.Linear(INPUT_SIZE, HIDDEN_SIZE, bias=False)
        self.mth_att = MultiheadAttention(HIDDEN_SIZE, n_heads, DROPOUT_R)
        self.ffn = FFN(HIDDEN_SIZE, FF_SIZE, DROPOUT_R)

        self.dropout1 = nn.Dropout(p=DROPOUT_R)
        self.norm1 = LayerNorm(size=HIDDEN_SIZE)

        self.dropout2 = nn.Dropout(p=DROPOUT_R)
        self.norm2 = LayerNorm(size=HIDDEN_SIZE)

        self.dropout3 = nn.Dropout(p=DROPOUT_R)
        self.norm3 = LayerNorm(size=HIDDEN_SIZE)

    def forward(self, x, y, z=None, y_mask=None):
        if z == None:
            z = y
        x = self.linear_x(x)
        y_att, att_weight = self.mth_att(query=x, key=y, value=z, mask=y_mask)
        x = self.norm2(
            x + self.dropout2(
                y_att
            )
        )
        x = self.norm3(
            x + self.dropout3(
                self.ffn(x)
            )
        )
        return x, att_weight

class BiLSTM(nn.Module):
    """
        Param:
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)  # utilize the LSTM model in torch.nn
        self.linear2 = nn.Linear(hidden_size*2, output_size) 

    def forward(self, input):
        x, _ = self.lstm(input)  # _x is input, size (seq_len, batch, input_size)
        x = self.linear2(x)
        return x
