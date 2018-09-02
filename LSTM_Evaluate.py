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
import my2module2rnn
from torch.nn import functional as F

# ##------------evaluate------
# load data
TRAIN1 = sio.loadmat('data/f001_STFT_TESTSET')
AC,BC = TRAIN1['STFT_ac'],TRAIN1['STFT_bc']
dataInfo = sio.loadmat('data/f001_datainfo.mat')
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
# normalize data
BC = log_and_normalize(BC,BC_mean,BC_std)
# load model
LSTMModel.load_state_dict(torch.load('data/params.pkl'))

# start to evaluate
testnum = BC.shape[0]
result = []
for i in range(testnum):
    LSTMModel.batch_size = 1  # this should be write before hidden_init
    LSTMModel.hidden = LSTMModel.init_hidden_state() #
    DATA = BC[i]
    sequence = DATA.shape[0]
    DATA = DATA[np.newaxis,:,:]
    DATA = torch.from_numpy(DATA).float()
    DATA = Variable(DATA.cpu())
    predict = LSTMModel(DATA,[sequence])
    # tensor.cuda change back to numpy
    predict = predict.data.cpu().numpy()
    predict = predict.squeeze()
    ##combine predict with the dic to get result and then,
    ##de_log_and_norm it.

    # denormalize
    predict = de_log_and_normalize(predict,AC_mean,AC_std)
    result.append(predict)

sio.savemat('pytorch_lstm_one_to_one.mat',{'result':result})