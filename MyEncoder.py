import torch.nn as nn
import torch
import torch.autograd as autograd
import my2module2rnn
import my2linear
from torch.nn import functional as F

class MyEncoder(nn.Module):

    def __init__(self, featDim, hidden_dim,batch_size):
        super(MyEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.linear1 = nn.Linear(featDim, hidden_dim[0])
        self.lstm1 = nn.LSTM(hidden_dim[0],hidden_dim[1],1,batch_first=True)
        self.lstm = my2module2rnn.sparseLSTM(hidden_dim[2],hidden_dim[3],1,batch_first=True)

    def init_hidden_state(self):
        self.h_c_lstm1 = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim[1]).cpu())
        self.h_c_lstm = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim[3]).cpu())
        return [self.h_c_lstm1,self.h_c_lstm]

    def forward(self,input,sequence):
        layer1 = self.linear1(input)
        layer1 = F.elu(layer1, alpha=1.0, inplace=False)
        packed_layer1 = nn.utils.rnn.pack_padded_sequence(layer1, sequence, batch_first=True)
        layer2, _ = self.lstm1(packed_layer1,self.h_c_lstm1)
        layer3, _ = self.lstm(layer2, self.h_c_lstm)
        unpack_layer3 = nn.utils.rnn.pad_packed_sequence(layer3,batch_first=True)
        unpack_layer3 = unpack_layer3[0]
        return unpack_layer3