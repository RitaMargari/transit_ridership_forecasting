from scipy import io
import numpy as np
import networkx as nx
import time
import torch
import torch.optim as optim
import torch.nn as nn
#torch.distributions.normal.Normal(loc, scale, validate_args=None)
torch.set_printoptions(sci_mode=False)
import networkx as nx
from GNNAPI import NNmodel

import copy


class NetReprLearning(NNmodel):
    def __init__(self,city,fts,GNNConfigs, VNNConfig, model='p',  
                 directembedding = False,attention=False,VNNattraction = False):
        super().__init__()
        self.city = city
        self.N = fts.shape[0]
#         self.fts = fts.to(torch.device("cuda"))
        self.directembedding = directembedding
        self.attention = attention
        self.VNNattraction = VNNattraction
        self.initNNs(fts,GNNConfigs, VNNConfig)
        self.model = model

    def embed(self,AL,X):
        if X is None:
            X = torch.FloatTensor(np.eye(self.N))
#         print('X.shape',X.shape)
        X = X.to(torch.device("cuda"))
        for i in range(self.GNNLayerNum):
            X = self.GNNLayers[i].forward(AL,X)
#             print(X)
        self.nodeEmbed = X
        return X    


    def params(self):
        return self.OutVNN.params() + sum([self.GNNLayers[i].params() for i in range(self.GNNLayerNum)], []) 
    
    def batch(self, X, Y, batchsize = 0): #default batching method; could be overriden
        # mh edited for situation batchsize = 0
        print('in batch function')
        batchind = range(X.shape[0])
        if batchsize > 0:
            batchind = np.random.choice(batchind, batchsize) #batching
            return X[batchind, :], Y[batchind, :]
        else:
            print('return X,Y')
            return X,Y

    def fitEpochHandle(self):
        #BCE = netloss(fitstate['Ytrue'], fitstate['Ytrue'], self.train_, style = 'BCE')
        #MSE = netloss(itstate['Ytrue'], fitstate['Ytrue'], self.train_, style = 'MSE')
        #acc = netloss(itstate['Ytrue'], fitstate['Ytrue'], self.train_, style = 'acc')

        print('Epoch: {:04d} of {:04d}'.format(self.fitstate['epoch'] + 1, self.fitstate['n_epochs']), 'batch loss: {:.8f}'.format(self.fitstate['loss']), 'full loss: {:.8f}'.format(self.fitstate['full_loss']),
              'best loss: {:.8f}'.format(self.fitstate['best_loss']),
              'time elapsed: {:.4f}s'.format(time.time() - self.fitstate['start_time']), self.param_checksum(previous = self.init_checksum,
                cumulative = True))

    # X is nodefts
    def fit(self,city,A,X,OD,between_fts,Avalid=None,Xvalid=None,ODvalid=None,between_fts_valid=None, 
        Atest=None,Xtest=None,ODtest=None,between_fts_test=None, n_epochs = 1000, 
            lr = 0.005, SEED = 1, interim_output_freq = 200):
#         vis = visdom.Visdom()
        self.Y = torch.FloatTensor(OD)
        self.Y = self.Y.to(torch.device("cuda"))
        # self.init_checksum = self.out_params()
        self.reloadBestAfterEpochs = 0
        self.iteratelearning = 0
        self.brokenGradientReload = False
        self.fitstate = {}
#         print('X.shape in fit',X.shape)
        #self.cs = copy.deepcopy(self.init_checksum)
        super().fit(city,A, X,self.Y,between_fts,
                    Avalid=Avalid,Xvalid=Xvalid,ODvalid=ODvalid,between_fts_valid=between_fts_valid,
                    Atest=Atest,Xtest=Xtest,ODtest=ODtest,between_fts_test=between_fts_test,
                     n_epochs = n_epochs, 
                    lr = lr, SEED = SEED, interim_output_freq = interim_output_freq)
    
    
    def param_checksum(self, previous = {}, cumulative = True):
        return dmerge({'OutVNN': self.OutVNN.param_checksum(previous.get('OutVNN', {}), cumulative)},
                      {'GNNLayer{}'.format(i): self.GNNLayers[i].param_checksum(previous.get('GNNLayer{}'.format(i), {}), cumulative) for i in range(self.GNNLayerNum)})

    def out_params(self):
        return dmerge({'OutVNN': self.OutVNN.out_params()},
                      {'GNNLayer{}'.format(i): self.GNNLayers[i].out_params() for i in range(self.GNNLayerNum)})

