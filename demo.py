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
from MyEncoder import MyEncoder
from MyDecoder import MyDecoder
import torch.nn.utils as utils
import gc
import resource

torch.manual_seed(1)
writer = SummaryWriter('run/LSTM_129-129(L)-256-256-129(L)')

def shuffle_data(AC,BC,AC_or,trainNum):
    index = range(trainNum)
    #random.shuffle(index)
    AC = AC[index]
    BC = BC[index]
    AC_or = AC_or[index]
    return AC,BC,AC_or

def prepare_data(bc,ac,ac_orig,batchsize,featDim):
    DATA = np.zeros((batchsize,2000,featDim))
    LABEL = np.zeros((batchsize,2000,featDim))
    LABEL_de_log_norm = np.zeros((batchsize,2000,featDim))
    Masking = np.zeros((batchsize,2000,featDim))
    TrueSequence = []
    maxSequence = 1
    for i in range(ac.shape[0]):
            LABEL[i,:ac[i].shape[0],:] = ac[i]
            LABEL_de_log_norm[i,:ac[i].shape[0],:] = ac_orig[i]
            DATA[i,:bc[i].shape[0],:] = bc[i]
            Masking[i,:bc[i].shape[0],:] = np.ones(bc[i].shape)
            TrueSequence.append(ac[i].shape[0])
            if ac[i].shape[0]>maxSequence:
                maxSequence = ac[i].shape[0]
    paixu = np.argsort(TrueSequence)
    paixu = paixu[::-1]
    DATA = DATA[paixu,:maxSequence,:]
    LABEL = LABEL[paixu,:maxSequence,:]
    LABEL_de_log_norm = LABEL_de_log_norm[paixu,:maxSequence,:]
    Masking = Masking[paixu,:maxSequence,:]
    Sequence = np.array(TrueSequence)[paixu]
    return DATA,LABEL,LABEL_de_log_norm,Sequence,Masking

def log_and_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.shape[1]):
        temp = np.log(data[0][i])
        temp = (temp-mean)/std
        #plt.imshow(temp.T,origin='lower')
        log_norm_data.append(temp)
    return np.array(log_norm_data)

def normalize_AC(data, mean, std):
    log_norm_data = []
    for i in range(data.shape[1]):
        temp =  data[0][i]
        temp = (temp - mean) / std
        log_norm_data.append(temp)
    return  np.array(log_norm_data)

def normalize_Dic(data):
    #log_norm_data = np.log(data)
    log_norm_data = data
    dic_num = log_norm_data.shape[0]
    sum1 = log_norm_data.sum( axis=0)
    mean1 = sum1/dic_num
    narray2 = log_norm_data * log_norm_data
    sum2 = narray2.sum(axis=0)
    var = sum2 / dic_num - mean1 ** 2
    log_norm_data = (log_norm_data - mean1) / (var** (1./2))
    log_norm_data = np.array(log_norm_data)
    return log_norm_data

def de_log_and_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.data.shape[0]):
            temp = data[i]
            temp = temp.data.numpy()
            temp = temp * std + mean
            temp = np.exp(temp)
            #plt.imshow(temp.T,origin='lower')
            log_norm_data.append(temp)
    result = np.array(log_norm_data)
    return result

def de_normalize(data,mean,std):
    log_norm_data = []
    for i in range(data.data.shape[0]):
            temp = data[i]
            temp = temp.data.numpy()
            temp = temp * std + mean
            log_norm_data.append(temp)
    result = np.array(log_norm_data)
    return result
def my_lstm_mse_loss(output,dic,target,sequence,masking,mean,std):
    dic = torch.from_numpy(dic).float()
    dic = Variable(dic)
    temp_e = torch.matmul(output,dic)
    #error =  Variable(torch.from_numpy(temp_e1).float())- target
    error = (temp_e - target)*masking
    error = (torch.sum(error ** 2) ) / sum(sequence)
    return error

def Combine_coef_dic(output_coef,dic):
    dic = torch.from_numpy(dic).float()
    dic = Variable(dic)
    temp = torch.matmul(output_coef, dic)
    return temp


# load data and split to train and val dataset
TRAIN = sio.loadmat('data/f001_STFT_TRAINSET')
TRAIN_DIC = sio.loadmat('data/f001_sparse_Dic')# f001_STFT_Dic

AC = TRAIN['STFT_ac'] # change stft to log
BC = TRAIN['STFT_bc']
AC_Dic = TRAIN_DIC['dic']
AC_Dic = normalize_Dic(AC_Dic)

AC_orig_Train = AC

dataInfo = sio.loadmat('data/f001_datainfo.mat')
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
AC_mean_nolog,AC_std_nolog = dataInfo['STFT_ac_mean'],dataInfo['STFT_ac_var']
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']

# normalize data
featDim = 129
AC = log_and_normalize(AC,AC_mean,AC_std)
AC_orig_Train = normalize_AC(AC_orig_Train,AC_mean_nolog,AC_std_nolog)

## compasating dicinary
# Comps_Dic =  np.eye(featDim)
# Comps_Dic_Neg = (-1)*np.eye(featDim)
# AC_Dic = np.vstack((AC_Dic,Comps_Dic,Comps_Dic_Neg))

BC = log_and_normalize(BC,BC_mean,BC_std)
testdata = sio.loadmat('data/f001_STFT_TESTSET')
t_ac,t_bc = np.array(testdata['STFT_ac']),np.array(testdata['STFT_bc'])
AC_orig_Val = t_ac
t_ac = log_and_normalize(t_ac,AC_mean,AC_std)
t_bc = log_and_normalize(t_bc,BC_mean,BC_std)

AC_orig_Val = normalize_AC(AC_orig_Val,AC_mean_nolog,AC_std_nolog)

Num = AC.shape[0]
train_ac,train_bc = AC[:],BC[:]
val_ac,val_bc = t_ac,t_bc
train_num = train_ac.shape[0]
val_num = val_ac.shape[0]

train_batchsize = 1
val_batchsize = 1
num_epochs =  100 ## <---

hidden_dim_encoder =  [129,256,256,200]# feedforwa[256,458,458,458]rd 458 hidden, lstm hidden1, lstm hidden2, feedforward hidden
hidden_dim_decoder =  [129,256,129]


num_train_batch = int(train_num/train_batchsize)
num_val_batch = int(val_num/val_batchsize)

EncoderModel = MyEncoder(featDim, hidden_dim_encoder,train_batchsize)
DecoderModel = MyDecoder(hidden_dim_decoder,train_batchsize)


# initial weight----> load the paras.
# for name, param in EncoderModel.named_parameters():
#   if 'bias' in name:
#      nn.init.constant(param, 0.0)
#   elif 'weight' in name:
#      nn.init.xavier_normal(param)

# load model
EncoderModel.load_state_dict(torch.load('data/params-encoder.pkl'))

#initial weight----> load the paras.
for name, param in DecoderModel.named_parameters():
  if 'bias' in name:
     nn.init.constant(param, 0.0)
  elif 'weight' in name:
     nn.init.xavier_normal(param)

EncoderModel.cpu()
DecoderModel.cpu()

#criterion = nn.MSELoss() #nn.MSELoss()

def criterion(output,target,target_length):
    error = (output - target)
    error = (torch.sum(error ** 2) ) / target_length
    return error

encoder_optimizer = optim.Adam(EncoderModel.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
decoder_optimizer = optim.Adam(DecoderModel.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


encoder_scheduler = lr_scheduler.ReduceLROnPlateau(encoder_optimizer,'min',patience=2,factor=0.5,min_lr=0.000001)
decoder_scheduler = lr_scheduler.ReduceLROnPlateau(decoder_optimizer,'min',patience=2,factor=0.5,min_lr=0.000001)


num_iteration_train = 0
num_iteration_test = 0
best_model_wts = copy.deepcopy(EncoderModel.state_dict())
best_loss = 1000

notimproveNum = 0
clip = 5

# tensor = torch.FloatTensor([0])
# loss = Variable(tensor, requires_grad=True)


for epoch in range(num_epochs):
  # shuffle the dataorder
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    if notimproveNum>5:
        print('Valloss do not improve at {} epochs,so break'.format(notimproveNum))
        break
    for phase in ['train', 'val']:
        if phase == 'train':
            #scheduler.step()
            EncoderModel.train()  # Set model to training mode
            EncoderModel.batch_size = train_batchsize
            num_batch = num_train_batch
            batchsize = train_batchsize
            AC,BC,AC_orig = shuffle_data(train_ac,train_bc,AC_orig_Train,train_num)

        else:
            EncoderModel.eval() # Set model to evaluate mode
            EncoderModel.batch_size = val_batchsize
            num_batch = num_val_batch
            batchsize = val_batchsize
            AC,BC,AC_orig = shuffle_data(val_ac,val_bc,AC_orig_Val,val_num)

        running_loss = 0.0



        for j in range(1):#num_batch
            DATA,LABEL,LABEL_de_log_norm,Sequence,Masking = prepare_data(BC[j*batchsize:(j+1)*batchsize],AC[j*batchsize:(j+1)*batchsize],AC_orig[j*batchsize:(j+1)*batchsize],batchsize,featDim) #prepare_data(bc,ac,batchsize,featDim):
            DATA,LABEL,LABEL_de_log_norm,Masking = torch.from_numpy(DATA).float(),torch.from_numpy(LABEL).float(),torch.from_numpy(LABEL_de_log_norm).float(),torch.from_numpy(Masking).float() # Pa
            DATA,LABEL,LABEL_de_log_norm,Masking= Variable(DATA.cpu()),Variable(LABEL.cpu()),Variable(LABEL_de_log_norm.cpu()),Variable(Masking.cpu())
            EncoderModel.zero_grad()
            [EncoderModel.hidden1,EncoderModel.hidden] = EncoderModel.init_hidden_state()

            encoder_outputs = EncoderModel(DATA,Sequence)
            encoder_outputs = Combine_coef_dic(encoder_outputs,AC_Dic)

            target_length = LABEL_de_log_norm.data.shape[1]
            #-----decoder process
            DecoderModel.zero_grad()
            [last_hidden, last_state] = DecoderModel.init_hidden_state()

            loss = np.zeros(1)
            for current_index in range(target_length):
                #print('{}:current_index'.format(current_index))
                # gc.collect()
                # max_mem_used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                # print("{:.2f} MB".format(max_mem_used / 1024))

                decoder_output, last_hidden, last_state = DecoderModel(last_hidden, last_state, encoder_outputs, current_index)
                temp_loss = criterion(decoder_output, LABEL_de_log_norm[0,current_index,:], target_length)
                loss += temp_loss.data[0]
            #-----end of decoder process

            #loss = my_lstm_mse_loss(encoder_outputs,AC_Dic,LABEL_de_log_norm,Sequence,Masking,AC_mean,AC_std)
            loss = torch.from_numpy(loss)
            loss = Variable(loss, requires_grad=True)
            if phase == 'train':
                loss.backward()
                utils.clip_grad_norm(EncoderModel.parameters(), clip)
                utils.clip_grad_norm(DecoderModel.parameters(), clip)
                encoder_optimizer.step()
                decoder_optimizer.step()

                num_iteration_train = num_iteration_train+1
                #writer.add_scalar('TrainLoss', loss.data[0], num_iteration_train)
                print('{}: {} Average_BatchLoss: {:.4f} '.format(j, phase, loss.data[0]/target_length))
            else:
                num_iteration_test = num_iteration_test+1
                #writer.add_scalar('VALLoss', loss.data[0], num_iteration_test)
                print('{}: {} Average_BatchLoss: {:.4f} '.format(j, phase, loss.data[0]/target_length))

            running_loss += loss


        epoch_loss = running_loss.data[0]/(num_batch)  ##????????????????????????????????shuju geshi

        if phase == 'val':
            former_lr = encoder_optimizer.param_groups[0]['lr']
            encoder_scheduler.step(epoch_loss)
            current_lr = encoder_optimizer.param_groups[0]['lr']
            writer.add_scalar('Epoch_VALLoss', epoch_loss, epoch)
            print('learning rate is {}'.format(encoder_optimizer.param_groups[0]['lr']))
            if  epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(EncoderModel.state_dict())
                EncoderModel.load_state_dict(best_model_wts)
                print('BestLoss: {:.4f} is Epoch{} '.format(best_loss,epoch+1))
                notimproveNum = 0
            else:
                notimproveNum = notimproveNum +1
        else:
            writer.add_scalar('Epoch_TrainLoss', epoch_loss, epoch)


        print('{} EpochLoss: {} '.format(phase,epoch_loss))

EncoderModel.load_state_dict(best_model_wts)
torch.save(EncoderModel.state_dict(),'data/params.pkl')


# ##------------evaluate----------------------------------------------##
# load data
TRAIN1 = sio.loadmat('data/f001_STFT_TESTSET')
AC,BC = TRAIN1['STFT_ac'],TRAIN1['STFT_bc']
dataInfo = sio.loadmat('data/f001_datainfo.mat')
BC_mean,BC_std = dataInfo['log_STFT_bc_mean'],dataInfo['log_STFT_bc_var']
AC_mean,AC_std = dataInfo['log_STFT_ac_mean'],dataInfo['log_STFT_ac_var']
AC_mean_nolog,AC_std_nolog = dataInfo['STFT_ac_mean'],dataInfo['STFT_ac_var']
# normalize data
BC = log_and_normalize(BC,BC_mean,BC_std)
# load model
EncoderModel.load_state_dict(torch.load('data/params.pkl'))

# start to evaluate
testnum = BC.shape[0]
result = []

dic_val = torch.from_numpy(AC_Dic).float()
dic_val = Variable(dic_val)

for i in range(testnum):
    EncoderModel.batch_size = 1  # this should be write before hidden_init
    [EncoderModel.hidden1, EncoderModel.hidden] = EncoderModel.init_hidden_state()
    DATA = BC[i]
    sequence = DATA.shape[0]
    DATA = DATA[np.newaxis,:,:]
    DATA = torch.from_numpy(DATA).float()
    DATA = Variable(DATA.cpu())
    predict = EncoderModel(DATA,[sequence])
    predict_val = torch.matmul(predict, dic_val)
    predict_val = de_normalize(predict_val,AC_mean_nolog,AC_std_nolog)
    predict_val = np.array(predict_val[0,:,:])
    result.append(predict_val)

sio.savemat('data/pytorch_lstm_one_to_one.mat',{'result':result})