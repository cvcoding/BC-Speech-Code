import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from Attn import Attn

class MyDecoder(nn.Module): #BahdanauAttnDecoderRNN
    def __init__(self, hidden_dim, batch_size):
        super(MyDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.attn = Attn('general', hidden_dim[0], hidden_dim[1])
        self.lstm = nn.LSTM(hidden_dim[0]*2, hidden_dim[1], 1, batch_first=True)
        self.linear = nn.Linear(hidden_dim[1], hidden_dim[2])

    def init_hidden_state(self):

        [self.last_hidden, self.last_state] = autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim[1]).cpu())
        return [self.last_hidden.view(1,1,-1), self.last_state.view(1,1,-1)]

    def forward(self, last_hidden, last_state, encoder_outputs,current_index,target_length):

        if current_index<=5:
            matrix = [i + current_index for i in range(11)]
        elif target_length-current_index<=5:
            matrix = [i + current_index -10 for i in range(11)]
        else:
            matrix = [i+ current_index - 5 for i in range(11)]
        frac = encoder_outputs[:,matrix,:]
        attn_weights = self.attn(last_hidden.squeeze(0), frac).squeeze(0) #last_hidden[-1]

        encoder_outputs = encoder_outputs.squeeze(0)
        frac = frac.squeeze(0)

        #temp = temp.transpose(0, 1)
        context = attn_weights.matmul(frac)
        rnn_input = torch.cat((encoder_outputs[current_index,:].unsqueeze(0), context), 1)
        temp = (last_hidden, last_state)
        out, (last_hidden,last_state) = self.lstm(rnn_input.view(1,1,-1), temp)
        output = self.linear(out.view(1,self.hidden_dim[1]))

        return [output,last_hidden, last_state]
