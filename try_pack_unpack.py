import scipy.io as sio
import random
import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
from torch.nn import functional as F

a = torch.Tensor([[1,2,3],[2,1,0],[3,0,0],[1,0,0]]).resize_(4,3,1)
a = Variable(a)
sequence = [3,2,1,1]
print(a)
packed_a = nn.utils.rnn.pack_padded_sequence(a, sequence, batch_first=True)
print(packed_a)
#packed_a = Variable(packed_a)
unpack_a = nn.utils.rnn.pad_packed_sequence(packed_a,batch_first=True)
print(unpack_a)

a = torch.Tensor([[1,2,3],[2,1,0],[3,0,0],[1,0,0]]).resize_(1,3,4)
a = Variable(a)
sequence = [3]
print(a)
packed_a = nn.utils.rnn.pack_padded_sequence(a, sequence, batch_first=True)
print(packed_a)
#packed_a = Variable(packed_a)
unpack_a = nn.utils.rnn.pad_packed_sequence(packed_a,batch_first=True)
print(unpack_a)

# a = torch.FloatTensor([[1,2,3],[2,1,0],[3,0,0],[1,0,0]]).resize_(4,3,1)
# b = torch.FloatTensor([[2,2,3],[3,1,0],[4,0,0],[2,0,0]]).resize_(4,3,1)
# a = Variable(a)
# b = Variable(b)
# c = a-b
# print(c)