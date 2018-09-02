# try to build a lstm network
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

torch.manual_seed(1)
writer = SummaryWriter('run/LSTM_129-129(L)-256-256-129(L)')


def log_and_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.shape[1]):
        temp = np.log(data[0][i])
        temp = (temp-mean)/std
        #plt.imshow(temp.T,origin='lower')
        log_norm_data.append(temp)
    return np.array(log_norm_data)



# load data and split to train and val dataset
TRAIN = sio.loadmat('data/f001_STFT_TRAINSET')

AC = TRAIN['STFT_ac'] # change stft to log
BC = TRAIN['STFT_bc']
dataInfo = sio.loadmat('data/f001_datainfo.mat')
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']

# normalize data

AC = log_and_normalize(AC,AC_mean,AC_std)

sio.savemat('f001_TRAIN_AClog.mat',{'AC':AC})
