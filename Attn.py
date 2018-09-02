#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.6
@author: 'zenRRan'
@license: Apache Licence 
@contact: zenrran@qq.com
@software: PyCharm
@file: Attn.py
@time: 2018/3/12 14:54
"""


import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Attn(nn.Module):
    def __init__(self, method, hidden_size0,hidden_size1):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size0 = hidden_size0 #129
        self.hidden_size1 = hidden_size1 #256

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size0, self.hidden_size1)
        # elif self.method == 'concat':
        #     self.attn = nn.Linear(self.hidden_size*2, self.hidden_size)
        #     self.other = nn.Parameter(torch.FloatTensor(1, self.hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.data.shape[1]
        attn_energies = Variable(torch.zeros(seq_len))
        for i in range(seq_len):
            temp = encoder_outputs[0,i,:]
            attn_energies[i] = self.score(hidden, temp)
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            hidden = hidden.squeeze(0)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy



















